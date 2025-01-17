from __future__ import annotations

import abc
import parallelproj
import array_api_compat.numpy as np
from array_api_compat import get_namespace, device

from types import ModuleType
from typing import TypeAlias, Callable

Array: TypeAlias = np.ndarray
try:
    import array_api_compat.cupy as cp

    Array: TypeAlias = Array | cp.ndarray
except:
    pass

from copy import copy
from rdp import SmoothFunction, SmoothFunctionWithDiagonalHessian


class SmoothSubsetFunction(abc.ABC):

    def __init__(
        self,
        in_shape: tuple[int],
        num_subsets: int,
        xp: ModuleType,
        dev: str,
    ) -> None:

        self._in_shape = in_shape
        self._num_subsets = num_subsets
        self._xp = xp
        self._dev = dev

        self._subset_gradients = None

    @property
    def in_shape(self) -> tuple[int, ...]:
        return self._in_shape

    @property
    def num_subsets(self) -> int:
        return self._num_subsets

    @property
    def xp(self):
        return self._xp

    @property
    def dev(self):
        return self._dev

    @abc.abstractmethod
    def call_subset(self, x: Array, subset: int) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def subset_gradient(self, x: Array, subset: int) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def all_subset_gradients(self, x: Array) -> list[Array]:
        raise NotImplementedError

    def __call__(self, x: Array) -> float:
        x = self._xp.asarray(x, device=self._dev)

        flat_input = x.ndim == 1
        if flat_input:
            x = self._xp.reshape(x, self._in_shape)

        res = sum([self.call_subset(x, subset) for subset in range(self._num_subsets)])

        return res

    def gradient(self, x: Array) -> Array:
        dev_input = device(x)

        if dev_input != self._dev:
            x = self._xp.asarray(x, device=self._dev)

        flat_input = x.ndim == 1
        if flat_input:
            x = self._xp.reshape(x, self._in_shape)

        res = sum(self.all_subset_gradients(x))

        if flat_input:
            res = self._xp.reshape(res, (res.size,))

        if dev_input != self._dev:
            res = self._xp.to_device(res, dev_input)

        return res


class SubsetNegPoissonLogLWithPrior(SmoothSubsetFunction):
    def __init__(
        self,
        data: Array,
        subset_fwd_operators: parallelproj.LinearOperatorSequence,
        contamination: Array,
        subset_slices: list[tuple[slice, ...]],
        prior: None | SmoothFunction = None,
    ) -> None:
        self._data = data
        self._subset_fwd_operators = subset_fwd_operators
        self._contamination = contamination
        self._subset_slices = subset_slices
        self._prior = prior
        self._adjoint_ones = None

        super().__init__(
            in_shape=subset_fwd_operators[0].in_shape,
            num_subsets=len(subset_fwd_operators),
            xp=get_namespace(data),
            dev=device(data),
        )

    @property
    def data(self) -> Array:
        return self._data

    @property
    def subset_fwd_operators(self) -> parallelproj.LinearOperatorSequence:
        return self._subset_fwd_operators

    @property
    def contamination(self) -> Array:
        return self._contamination

    @property
    def subset_slices(self) -> list[slice]:
        return self._subset_slices

    @property
    def prior(self) -> None | SmoothFunction:
        return self._prior

    def call_subset(self, x: Array, subset: int) -> float:
        sl = self._subset_slices[subset]

        exp = self._subset_fwd_operators[subset](x) + self._contamination[sl]

        res = float(self.xp.sum(exp - self._data[sl] * self.xp.log(exp)))

        if self._prior is not None:
            res += self._prior(x) / self._num_subsets

        return res

    def subset_gradient(self, x: Array, subset: int) -> Array:
        sl = self._subset_slices[subset]
        exp = self._subset_fwd_operators[subset](x) + self._contamination[sl]

        res = self._subset_fwd_operators[subset].adjoint((exp - self._data[sl]) / exp)

        if self._prior is not None:
            res += self._prior.gradient(x) / self._num_subsets

        return res

    def all_subset_gradients(self, x: Array) -> list[Array]:
        """dedicated implementation to avoid multiple evaluations of the prior"""

        if self._prior is not None:
            prior_grad = self._prior.gradient(x)

        subset_grads = []

        for subset in range(self._num_subsets):
            sl = self._subset_slices[subset]
            exp = self._subset_fwd_operators[subset](x) + self._contamination[sl]

            subset_grad = self._subset_fwd_operators[subset].adjoint(
                (exp - self._data[sl]) / exp
            )

            if self._prior is not None:
                subset_grad += prior_grad / self._num_subsets

            subset_grads.append(subset_grad)

        return subset_grads


class OSEM:
    def __init__(self, data_fidelity: SubsetNegPoissonLogLWithPrior) -> None:
        self._data_fidelity = data_fidelity

        self._num_subsets = data_fidelity.num_subsets

        self._adjoint_ones = []

        for i in range(self._num_subsets):
            self._adjoint_ones.append(
                data_fidelity.subset_fwd_operators[i].adjoint(
                    data_fidelity.xp.ones(
                        data_fidelity.subset_fwd_operators[i].out_shape,
                        device=data_fidelity.dev,
                    )
                )
            )

    def update(self, x: Array, subset: int) -> Array:
        step = x / self._adjoint_ones[subset]
        return x - step * self._data_fidelity.subset_gradient(x, subset)

    def run(self, x: Array, num_epochs: int) -> Array:
        for _ in range(num_epochs):
            for subset in range(self._num_subsets):
                x = self.update(x, subset)

        return x


def split_fwd_model(
    pet_lin_op: parallelproj.CompositeLinearOperator,
    num_subsets: int,
):
    """split PET forward model into a sequence of subset forward operators"""
    att_sino: Array = pet_lin_op[0].values
    proj: parallelproj.RegularPolygonPETProjector = pet_lin_op[1]
    res_model: parallelproj.LinearOperator = pet_lin_op[2]

    subset_views, subset_slices = proj.lor_descriptor.get_distributed_views_and_slices(
        num_subsets, len(proj.out_shape)
    )

    _, subset_slices_non_tof = proj.lor_descriptor.get_distributed_views_and_slices(
        num_subsets, 3
    )

    # clear the cached LOR endpoints since we will create many copies of the projector
    proj.clear_cached_lor_endpoints()
    pet_subset_linop_seq = []

    # we setup a sequence of subset forward operators each constisting of
    # (1) image-based resolution model
    # (2) subset projector
    # (3) multiplication with the corresponding subset of the attenuation sinogram
    for i in range(num_subsets):
        # make a copy of the full projector and reset the views to project
        subset_proj = copy(proj)
        subset_proj.views = subset_views[i]

        if subset_proj.tof:
            subset_att_op = parallelproj.TOFNonTOFElementwiseMultiplicationOperator(
                subset_proj.out_shape, att_sino[subset_slices_non_tof[i]]
            )
        else:
            subset_att_op = parallelproj.ElementwiseMultiplicationOperator(
                att_sino[subset_slices_non_tof[i]]
            )

        # add the resolution model and multiplication with a subset of the attenuation sinogram
        pet_subset_linop_seq.append(
            parallelproj.CompositeLinearOperator(
                [
                    subset_att_op,
                    subset_proj,
                    res_model,
                ]
            )
        )

    pet_subset_lin_op_seq = parallelproj.LinearOperatorSequence(pet_subset_linop_seq)

    return pet_subset_lin_op_seq, subset_slices


class SVRG:
    def __init__(
        self,
        subset_neglogL: SmoothSubsetFunction,
        prior: SmoothFunctionWithDiagonalHessian,
        x_init: Array,
        complete_gradient_epochs: None | list[int] = None,
        precond_update_epochs: None | list[int] = None,
        verbose: bool = False,
        seed: int = 1,
        precond_version: int = 2,
    ):

        np.random.seed(seed)

        self._verbose = verbose
        self._subset = 0
        self._x = x_init

        self._subset_neglogL = subset_neglogL
        self._prior = prior

        self._num_subsets = subset_neglogL.num_subsets

        self._xp = get_namespace(x_init)
        self._dev = device(x_init)

        self._adjoint_ones = self._xp.zeros_like(x_init)

        # calculate adjoint ones of data operator
        for i in range(self._num_subsets):
            self._adjoint_ones += self._subset_neglogL.subset_fwd_operators[i].adjoint(
                self._xp.ones(
                    self._subset_neglogL.subset_fwd_operators[i].out_shape,
                    device=self._dev,
                )
            )

        if complete_gradient_epochs is None:
            self._complete_gradient_epochs: list[int] = [x for x in range(0, 1000, 2)]
        else:
            self._complete_gradient_epochs = complete_gradient_epochs

        if precond_update_epochs is None:
            self._precond_update_epochs: list[int] = [1, 2, 3]
        else:
            self._precond_update_epochs = precond_update_epochs

        self._precond_filter = parallelproj.GaussianFilterOperator(
            x_init.shape, sigma=2.0
        )

        self._update = 0
        self._step_size_factor = 1.0
        self._subset_number_list = []
        self._precond_version = precond_version
        self._precond = self.calc_precond(self._x)
        self._step_size = 1.0
        self._prior_diag_hess = None

    @property
    def epoch(self) -> int:
        return self._update // self._num_subsets

    @property
    def x(self) -> Array:
        return self._x

    def update_step_size(self):
        if self.epoch <= 4:
            self._step_size = self._step_size_factor * 2.0
        elif self.epoch > 4 and self.epoch <= 8:
            self._step_size = self._step_size_factor * 1.5
        elif self.epoch > 8 and self.epoch <= 12:
            self._step_size = self._step_size_factor * 1.0
        else:
            self._step_size = self._step_size_factor * 0.5

        if self._verbose:
            print(self._update, self.epoch, self._step_size)

    def calc_precond(
        self,
        x: Array,
        delta_rel: float = 1e-6,
    ) -> Array:

        # generate a smoothed version of the input image
        # to avoid high values, especially in first and last slices
        x_sm = self._precond_filter(x)
        delta = delta_rel * x_sm.max()

        if self._precond_version == 1:
            precond = (x_sm + delta) / self._adjoint_ones
        elif self._precond_version == 2:
            self._prior_diag_hess = self._prior.diag_hessian(x_sm)

            precond = (x_sm + delta) / (
                self._adjoint_ones + 2 * self._prior_diag_hess * x_sm
            )
        else:
            raise ValueError(f"Unknown preconditioner version {self._precond_version}")

        return precond

    def update_all_subset_gradients(self) -> None:

        self._subset_gradients = []
        subset_prior_gradient = self._prior.gradient(self._x) / self._num_subsets

        for i in range(self._num_subsets):
            self._subset_gradients.append(
                self._subset_neglogL.subset_gradient(self._x, i) + subset_prior_gradient
            )

        self._summed_subset_gradients = sum(self._subset_gradients)

    def update(self):

        update_all_subset_gradients = (
            self._update % self._num_subsets == 0
        ) and self.epoch in self._complete_gradient_epochs

        update_precond = (
            self._update % self._num_subsets == 0
        ) and self.epoch in self._precond_update_epochs

        if self._update % self._num_subsets == 0:
            self.update_step_size()

        if update_precond:
            if self._verbose:
                print(f"  {self._update}, updating preconditioner")
            self._precond = self.calc_precond(self._x)

        if update_all_subset_gradients:
            if self._verbose:
                print(
                    f"  {self._update}, {self._subset}, recalculating all subset gradients"
                )
            self.update_all_subset_gradients()
            approximated_gradient = self._summed_subset_gradients
        else:
            if self._subset_number_list == []:
                self.create_subset_number_list()

            self._subset = self._subset_number_list.pop()
            if self._verbose:
                print(f" {self._update}, {self._subset}, subset gradient update")

            subset_prior_gradient = self._prior.gradient(self._x) / self._num_subsets

            approximated_gradient = (
                self._num_subsets
                * (
                    (
                        self._subset_neglogL.subset_gradient(self._x, self._subset)
                        + subset_prior_gradient
                    )
                    - self._subset_gradients[self._subset]
                )
                + self._summed_subset_gradients
            )

        self._x = self._x - self._step_size * self._precond * approximated_gradient

        # enforce non-negative constraint
        self._xp.clip(self._x, 0, None, out=self._x)

        self._update += 1

    def run(
        self, num_updates: int, callback: None | Callable[[Array], float] = None
    ) -> list:

        callback_res = []

        for _ in range(num_updates):
            if callback is not None:
                callback_res.append(callback(self._x))
            self.update()

        return callback_res

    def create_subset_number_list(self):
        tmp = np.arange(self._num_subsets)
        np.random.shuffle(tmp)
        self._subset_number_list = tmp.tolist()


class ProxRDP:
    def __init__(
        self,
        prior: SmoothFunctionWithDiagonalHessian,
        init_step: float = 1.0,
        niter: int = 5,
        adaptive_step_size: bool = True,
        up: float = 1.1,
        down: float = 2.0,
    ):
        self._prior = prior
        self._step = init_step
        self._niter = niter
        self._up = up
        self._down = down
        self._adaptive_step_size = adaptive_step_size

    @property
    def prior(self) -> SmoothFunctionWithDiagonalHessian:
        return self._prior

    def __call__(
        self, z, tau, T: float | Array = 1.0, precond: float | Array = 1.0
    ) -> Array:

        xp = get_namespace(z)

        u = xp.clip(z, 0, None)

        for k in range(self._niter):
            # compute gradient step
            grad = (u - z) / T + tau * self._prior.gradient(u)
            tmp = u - self._step * precond * grad

            u_new = xp.clip(tmp, 0, None)

            # update step size

            if self._adaptive_step_size:
                diff_new = xp.linalg.norm(u_new - u)

                if k == 0:
                    u = u_new
                    diff = diff_new
                else:
                    if diff_new <= diff:
                        self._step *= self._up
                        u = u_new
                        diff = diff_new
                    else:
                        self._step /= self._down
            else:
                u = u_new

        return u


class ProxSVRG:
    def __init__(
        self,
        subset_neglogL: SmoothSubsetFunction,
        prior_prox,
        x_init: Array,
        complete_gradient_epochs: None | list[int] = None,
        precond_update_epochs: None | list[int] = None,
        verbose: bool = False,
        seed: int = 1,
        precond_version: int = 2,
        **kwargs,
    ):

        np.random.seed(seed)

        self._verbose = verbose
        self._subset = 0
        self._x = x_init

        self._subset_neglogL = subset_neglogL

        self._num_subsets = subset_neglogL.num_subsets

        self._xp = get_namespace(x_init)
        self._dev = device(x_init)

        self._adjoint_ones = self._xp.zeros_like(x_init)

        # calculate adjoint ones of data operator
        for i in range(self._num_subsets):
            self._adjoint_ones += self._subset_neglogL.subset_fwd_operators[i].adjoint(
                self._xp.ones(
                    self._subset_neglogL.subset_fwd_operators[i].out_shape,
                    device=self._dev,
                )
            )

        if complete_gradient_epochs is None:
            self._complete_gradient_epochs: list[int] = [x for x in range(0, 1000, 2)]
        else:
            self._complete_gradient_epochs = complete_gradient_epochs

        if precond_update_epochs is None:
            self._precond_update_epochs: list[int] = [1, 2, 3]
        else:
            self._precond_update_epochs = precond_update_epochs

        self._precond_filter = parallelproj.GaussianFilterOperator(
            x_init.shape, sigma=2.0
        )

        self._update = 0
        self._step_size_factor = 1.0
        self._subset_number_list = []
        self._prior_prox = prior_prox
        self._prior_diag_hess = None
        self._precond_version = precond_version
        self._precond = self.calc_precond(self._x)
        self._step_size = 1.0

        self._subset_gradients = []
        self._summed_subset_gradients = None

    @property
    def epoch(self) -> int:
        return self._update // self._num_subsets

    @property
    def x(self) -> Array:
        return self._x

    def update_step_size(self):
        if self.epoch <= 4:
            self._step_size = self._step_size_factor * 2.0
        elif self.epoch > 4 and self.epoch <= 8:
            self._step_size = self._step_size_factor * 1.5
        elif self.epoch > 8 and self.epoch <= 12:
            self._step_size = self._step_size_factor * 1.0
        else:
            self._step_size = self._step_size_factor * 0.5

        if self._verbose:
            print(self._update, self.epoch, self._step_size)

    def calc_precond(
        self,
        x: Array,
        delta_rel: float = 1e-6,
    ) -> Array:

        # generate a smoothed version of the input image
        # to avoid high values, especially in first and last slices
        x_sm = self._precond_filter(x)
        delta = delta_rel * x_sm.max()

        if self._precond_version == 1:
            precond = (x_sm + delta) / self._adjoint_ones
        elif self._precond_version == 2:
            self._prior_diag_hess = self._prior_prox.prior.diag_hessian(x_sm)

            precond = (x_sm + delta) / (
                self._adjoint_ones + 2 * self._prior_diag_hess * x_sm
            )
        else:
            raise ValueError(f"Unknown preconditioner version {self._precond_version}")

        return precond

    def update_all_subset_gradients(self) -> None:

        self._subset_gradients = self._subset_neglogL.all_subset_gradients(self._x)
        self._summed_subset_gradients = sum(self._subset_gradients)

    def update(self):

        update_all_subset_gradients = (
            self._update % self._num_subsets == 0
        ) and self.epoch in self._complete_gradient_epochs

        update_precond = (
            self._update % self._num_subsets == 0
        ) and self.epoch in self._precond_update_epochs

        if self._update % self._num_subsets == 0:
            self.update_step_size()

        if update_precond:
            if self._verbose:
                print(f"  {self._update}, updating preconditioner")
            self._precond = self.calc_precond(self._x)

        if update_all_subset_gradients:
            if self._verbose:
                print(
                    f"  {self._update}, {self._subset}, recalculating all subset gradients"
                )
            self.update_all_subset_gradients()
            approximated_gradient = self._summed_subset_gradients
        else:
            if self._subset_number_list == []:
                self.create_subset_number_list()

            self._subset = self._subset_number_list.pop()
            if self._verbose:
                print(f" {self._update}, {self._subset}, subset gradient update")

            approximated_gradient = (
                self._num_subsets
                * (
                    self._subset_neglogL.subset_gradient(self._x, self._subset)
                    - self._subset_gradients[self._subset]
                )
                + self._summed_subset_gradients
            )

        tmp = self._x - self._step_size * self._precond * approximated_gradient

        tau = self._step_size
        T = self._precond

        if self._precond_version == 1:
            pc = 1 / T
        elif self._precond_version == 2:
            pc = 1 / (1 / T + tau * self._prior_diag_hess)

        self._x = self._prior_prox(tmp, tau=tau, T=T, precond=pc)

        self._update += 1

    def run(
        self, num_updates: int, callback: None | Callable[[Array], float] = None
    ) -> list:

        callback_res = []

        for _ in range(num_updates):
            if callback is not None:
                callback_res.append(callback(self._x))
            self.update()

        return callback_res

    def create_subset_number_list(self):
        tmp = np.arange(self._num_subsets)
        np.random.shuffle(tmp)
        self._subset_number_list = tmp.tolist()
