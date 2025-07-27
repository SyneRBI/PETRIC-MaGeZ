from __future__ import annotations

from typing import Union, Callable, TYPE_CHECKING
from types import ModuleType
from collections.abc import Generator
from time import time
from tqdm import tqdm
import math

import abc
import parallelproj
import array_api_compat.numpy as np
from array_api_compat import get_namespace, device

if TYPE_CHECKING:
    import cupy as cp

    Array = Union[np.ndarray, cp.ndarray]  # Used for type checking
else:
    Array = np.ndarray  # Default at runtime


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

        res = float(
            self.xp.sum(exp - self._data[sl] * self.xp.log(exp), dtype=self.xp.float64)
        )

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


class StochasticGradientDescent:
    def __init__(
        self,
        subset_neglogL: SmoothSubsetFunction,
        prior: SmoothFunction,
        x_init: Array,
        diag_precond_func: Callable[[Array], Array],
        subset_generator: Generator[int, None, None],
        method: str = "SVRG",
        step_size_func: Callable[[int], float] = lambda z: 1.0,
        complete_gradient_epochs: None | list[int] = None,
        precond_update_epochs: None | list[int] = None,
        verbose: bool = False,
        gradient_norm_based_sampling: bool = False,
        barzilai_borwein: bool = False,
    ):
        if method not in ["SVRG", "SGD", "SAGA"]:
            raise ValueError("Unknown optimization method")

        self._method = method

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

        self._step_size_func = step_size_func

        if complete_gradient_epochs is None:
            self._complete_gradient_epochs: list[int] = [x for x in range(0, 1000, 2)]
        else:
            self._complete_gradient_epochs = complete_gradient_epochs

        if precond_update_epochs is None:
            self._precond_update_epochs: list[int] = [1, 2, 3]
        else:
            self._precond_update_epochs = precond_update_epochs

        self._update = 0
        self._subset_number_list = []
        self._subset_gradients = self._xp.zeros(
            (self._num_subsets,) + self._x.shape, device=self._dev, dtype=self._x.dtype
        )
        self._summed_subset_gradients = None

        self._diag_precond_func = diag_precond_func
        self._diag_precond = self._diag_precond_func(self._x)
        self._step_size = 1.0

        ##################################################################
        ##################################################################
        ##################################################################
        self._gradient_norm_based_sampling = gradient_norm_based_sampling
        print(f"Gradient norm based sampling: {self._gradient_norm_based_sampling}")
        if self._gradient_norm_based_sampling:
            self._subset_generator = ProbabilisticSampler(
                p=np.ones(self._num_subsets) / self._num_subsets,
                seed=1,
                with_replacement=True,
            )
            ##################################################################
            ##################################################################
            ##################################################################
        else:
            self._subset_generator = subset_generator

        self._barzilai_borwein = barzilai_borwein
        print(f"Barzilai-Borwein: {self._barzilai_borwein}")
        self._t0 = time()

    @property
    def epoch(self) -> int:
        return self._update // self._num_subsets

    @property
    def x(self) -> Array:
        return self._x

    @property
    def step_size_func(self) -> Callable[[int], float]:
        return self._step_size_func

    def update_all_subset_gradients(self) -> None:

        self._subset_gradients = self._xp.zeros(
            (self._num_subsets,) + self._x.shape, device=self._dev, dtype=self._x.dtype
        )
        subset_prior_gradient = self._prior.gradient(self._x) / self._num_subsets

        for i in range(self._num_subsets):
            self._subset_gradients[i, ...] = (
                self._subset_neglogL.subset_gradient(self._x, i) + subset_prior_gradient
            )

        self._summed_subset_gradients = self._xp.sum(self._subset_gradients, axis=0)

    def subset_gradient_norms(self) -> np.ndarray:
        return np.array(
            [float(self._xp.linalg.norm(g)) for g in self._subset_gradients]
        )

    def update(self):

        # update the step size according to the step size function
        # that maps the update_number to the step size (int -> float)
        if self._barzilai_borwein:
            if self._update <= 10:
                self._step_size = 3.0
            elif (self._update > 10) and (self.epoch <= 2):
                self._step_size = 2.2
            elif self.epoch >= 10:
                self._step_size = 1.0
        else:
            self._step_size = self._step_size_func(self._update)
        # print(f"Update:{self._update}, Step size: {self._step_size}")
        update_precond = (
            self._update % self._num_subsets == 0
        ) and self.epoch in self._precond_update_epochs

        if update_precond:
            if self._verbose:
                print(f"  {self._update}, updating preconditioner")
            self._diag_precond = self._diag_precond_func(self._x)

        # choose the subset to update
        self._subset = next(self._subset_generator)
        # print(f"epoch: {self.epoch}, iter:{self._update}, subset:{self._subset}, stepsize:{self._step_size}")
        if self._verbose:
            print(f" {self._update}, {self._subset}, subset gradient update")

        # calculate the stochastic gradient
        if self._method == "SGD":
            approximated_gradient = (
                self._num_subsets
                * self._subset_neglogL.subset_gradient(self._x, self._subset)
                + self._prior.gradient(self._x)
            )
        elif self._method == "SVRG":
            update_all_subset_gradients = (
                self._update % self._num_subsets == 0
            ) and self.epoch in self._complete_gradient_epochs

            if update_all_subset_gradients:
                if self._verbose:
                    print(
                        f"  {self._update}, {self._subset}, recalculating all subset gradients"
                    )
                self.update_all_subset_gradients()
                if self._barzilai_borwein:
                    if self._update == 0:
                        self.old_x = self._x
                        self.old_g = self._summed_subset_gradients
                    else:
                        self.old_x -= self._x
                        self.old_g -= self._summed_subset_gradients
                        self.tmp = self._diag_precond * self.old_g
                        bb_step = self._xp.sum(
                            self._xp.multiply(self.old_g, self.old_x)
                        ) / self._xp.sum(self._xp.multiply(self.old_g, self.tmp))
                        self.old_x = self._x
                        self.old_g = self._summed_subset_gradients
                        max_ss = 2.5 if self.epoch <= 2 else self._step_size
                        self._step_size = self._xp.clip(bb_step, 0.01, max_ss)

                #####################################################
                #####################################################
                #####################################################
                # update the probabilities for the probabilistic sampling
                if self._gradient_norm_based_sampling:
                    sgns = self.subset_gradient_norms()
                    pis = sgns / sgns.sum()
                    # print(pis)
                    # print(pis.min(), pis.max(), (pis.max() - pis.min()) / pis.mean())
                    self._subset_generator.p = pis
                #####################################################
                #####################################################
                #####################################################

                approximated_gradient = self._summed_subset_gradients
            else:
                approximated_gradient = (
                    self._num_subsets
                    * (
                        (
                            self._subset_neglogL.subset_gradient(self._x, self._subset)
                            + self._prior.gradient(self._x) / self._num_subsets
                        )
                        - self._subset_gradients[self._subset]
                    )
                    + self._summed_subset_gradients
                )
        elif self._method == "SAGA":
            subset_grad = (
                self._subset_neglogL.subset_gradient(self._x, self._subset)
                + self._prior.gradient(self._x) / self._num_subsets
            )

            approximated_gradient = self._num_subsets * (
                subset_grad - self._subset_gradients[self._subset]
            ) + self._xp.sum(self._subset_gradients, axis=0)

            self._subset_gradients[self._subset, ...] = subset_grad

        else:
            raise ValueError("Unknown optimization method")

        # update the image
        self._x = self._x - self._step_size * self._diag_precond * approximated_gradient

        # enforce non-negative constraint
        self._xp.clip(self._x, 0.0, None, out=self._x)
        self._update += 1

    def run(
        self, num_updates: int, callback: None | Callable[[Array], float] = None
    ) -> list:

        callback_res = []

        progress_bar = tqdm(range(num_updates))
        for _ in progress_bar:
            if callback is not None:
                callback_value = callback(self._x)
                callback_res.append([callback_value, time() - self._t0])
                progress_bar.set_postfix({"cb": callback_value})
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

        u = xp.clip(z, 0.0, None)

        for k in range(self._niter):
            # compute gradient step
            grad = (u - z) / T + tau * self._prior.gradient(u)
            tmp = u - self._step * precond * grad

            u_new = xp.clip(tmp, 0.0, None)

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


class MLEMPreconditioner:
    def __init__(self, adjoint_ones: Array, delta_rel: float = 1e-4):
        self._adjoint_ones = adjoint_ones
        self._delta_rel = delta_rel

    def __call__(self, x: Array) -> Array:
        return (x + self._delta_rel * x.max()) / self._adjoint_ones


class HarmonicPreconditioner:
    def __init__(
        self,
        adjoint_ones: Array,
        prior: SmoothFunctionWithDiagonalHessian,
        delta_rel: float = 1e-4,
        factor: float = 2.0,
        filter_function: None | Callable[[Array], Array] = None,
    ):
        self._adjoint_ones = adjoint_ones
        self._delta_rel = delta_rel
        self._factor = factor
        self._prior = prior
        self._filter_function = filter_function

    def __call__(self, x: Array) -> Array:

        if self._filter_function is not None:
            x_sm = self._filter_function(x)
        else:
            x_sm = x

        return (x + self._delta_rel * x.max()) / (
            self._adjoint_ones + self._factor * self._prior.diag_hessian(x_sm) * x
        )


def subset_generator_without_replacement(
    num_subsets: int, seed: int
) -> Generator[int, None, None]:
    """subset generator without replacement - every subset is yielded once before repeating"""
    indices = np.arange(num_subsets)  # Create an array of subset indices
    rng = np.random.default_rng(seed)  # NumPy random generator with optional seed
    while True:
        rng.shuffle(indices)  # Shuffle the indices in-place
        yield from indices  # Yield each index one by one


def subset_generator_with_replacement(
    num_subsets: int, seed: int
) -> Generator[int, None, None]:
    """subset generator with replacement"""
    indices = np.arange(num_subsets)  # Create an array of subset indices
    rng = np.random.default_rng(seed)  # NumPy random generator with optional seed
    while True:
        yield rng.choice(indices)  # Yield a random index with replacement


import numpy as np


class ProbabilisticSampler:
    def __init__(self, p: np.ndarray, seed: int, with_replacement=True):
        self._p = p
        self._N = len(p)
        self._with_replacement = with_replacement
        self._rng = np.random.default_rng(seed)

        self._reset_remaining()

    @property
    def p(self) -> np.ndarray:
        return self._p

    @p.setter
    def p(self, value: np.ndarray) -> None:
        self._p = value
        self._reset_remaining()

    def _reset_remaining(self):
        self._remaining = list(self._rng.choice(self._N, size=self._N, replace=False))

    def __iter__(self):
        return self

    def __next__(self):
        if self._with_replacement:
            return self._rng.choice(self._N, replace=True, p=self._p)
        else:
            if len(self._remaining) == 0:
                self._reset_remaining()
            return self._remaining.pop(0)


# Herman-Meyer order for subset selection
def subset_generator_herman_meyer(num_subsets: int) -> Generator[int, None, None]:
    """subset generator with replacement"""
    indices = herman_meyer_order(num_subsets)  # Create an array of subset indices
    while True:
        indices = herman_meyer_order(num_subsets)
        yield from indices  # Yield a herman-meyer generated index


def herman_meyer_order(n):
    order = [0] * n
    factors = []
    len_order = len(order)

    while n % 2 == 0:
        factors.append(2)
        n //= 2

    # Check for odd factors
    for factor in range(3, int(n**0.5) + 1, 2):
        while n % factor == 0:
            factors.append(factor)
            n //= factor

    # If n is a prime number greater than 2
    if n > 2:
        factors.append(n)

    n_factors = len(factors)
    value = 0
    for factor_n in range(n_factors):
        n_change_value = 1 if factor_n == 0 else math.prod(factors[:factor_n])
        n_rep_value = 0

        for element in range(len_order):
            mapping = value
            n_rep_value += 1
            if n_rep_value >= n_change_value:
                value += 1
                n_rep_value = 0
            if value == factors[factor_n]:
                value = 0
            order[element] += math.prod(factors[factor_n + 1 :]) * mapping
    return order


# cofactor sampling


def find_coprimes(n):
    coprimes = []
    for i in range(2, n):
        if math.gcd(n, i) == 1:
            coprimes.append(i)
    return coprimes


def subset_generator_cofactor(num_subsets: int) -> Generator[int, None, None]:
    """subset generator with replacement"""
    coprimes = find_coprimes(num_subsets)
    if not coprimes:
        indices = [0] * 10000
    else:
        sorted_coprimes = sorted(
            coprimes,
            key=lambda x: min(
                abs(x - int(np.round(0.3 * num_subsets))),
                abs(x - int(np.round(0.7 * num_subsets))),
            ),
        )
    while True:
        if not sorted_coprimes:
            sorted_coprimes = sorted(
                coprimes,
                key=lambda x: min(
                    abs(x - int(np.round(0.3 * num_subsets))),
                    abs(x - int(np.round(0.7 * num_subsets))),
                ),
            )
        generator = sorted_coprimes.pop(0)
        indices = [(generator * k) % num_subsets for k in range(num_subsets)]
        yield from indices  # Yield a cofactor generated index


if __name__ == "__main__":
    p1 = np.array([0.5, 0.5, 0.0, 0.0])
    sampler = ProbabilisticSampler(p1, seed=1, with_replacement=True)
    for i in range(len(p1)):
        print(next(sampler))

    print()

    p2 = np.array([0.0, 0.0, 1.0, 0.0])
    sampler.p = p2
    for i in range(len(p2)):
        print(next(sampler))
