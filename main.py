"""Main file to modify for submissions.

Once renamed or symlinked as `main.py`, it will be used by `petric.py` as follows:

>>> from main import Submission, submission_callbacks
>>> from petric import data, metrics
>>> algorithm = Submission(data)
>>> algorithm.run(np.inf, callbacks=metrics + submission_callbacks)
"""

import math
import sirf.STIR as STIR
from cil.optimisation.algorithms import Algorithm
from cil.optimisation.utilities import callbacks
from sirf.contrib.partitioner import partitioner

from collections.abc import Callable

import numpy as np
import array_api_compat.cupy as xp
from array_api_compat import to_device

# import pure python re-implementation of the RDP -> only used to get diagonal of the RDP Hessian!
from rdp import RDP

from petric import Dataset

import math

def find_coprimes(n):
    coprimes = []
    for i in range(2, n):
        if math.gcd(n, i) == 1:
            coprimes.append(i)
    return coprimes

def neighbor_difference_and_sum(x, padding= "edge", _dev='cpu'):
    """get differences and sums with nearest neighbors for an n-dimensional array x
    using padding (by default in edge mode)
    a x.ndim*(3,) neighborhood around each element is used
    """
    x_padded = xp.pad(x, 1, mode=padding)
    # number of nearest neighbors
    num_neigh = 3**x.ndim - 1
    # array for differences and sums with nearest neighbors
    d = xp.zeros((num_neigh,) + x.shape, dtype=x.dtype, device=_dev)
    s = xp.zeros_like(d)

    for i, ind in enumerate(xp.ndindex(x.ndim * (3,))):
        if i != (num_neigh // 2):
            sl = []
            for j in ind:
                if j - 2 < 0:
                    sl.append(slice(j, j - 2))
                else:
                    sl.append(slice(j, None))
            sl = tuple(sl)

            if i < num_neigh // 2:
                d[i, ...] = x - x_padded[sl]
                s[i, ...] = x + x_padded[sl]
            else:
                d[i - 1, ...] = x - x_padded[sl]
                s[i - 1, ...] = x + x_padded[sl]
    return d, s

def neighbor_product(x, padding = "edge", _dev='cpu'):
    """get backward and forward neighbor products for each dimension of an array x
    using padding (by default in edge mode)
    """
    x_padded = xp.pad(x, 1, mode=padding)
    # number of nearest neighbors
    num_neigh = 3**x.ndim - 1

    # array for differences and sums with nearest neighbors
    p = xp.zeros((num_neigh,) + x.shape, dtype=x.dtype, device=_dev)

    for i, ind in enumerate(xp.ndindex(x.ndim * (3,))):
        if i != (num_neigh // 2):
            sl = []
            for j in ind:
                if j - 2 < 0:
                    sl.append(slice(j, j - 2))
                else:
                    sl.append(slice(j, None))
            sl = tuple(sl)

            if i < num_neigh // 2:
                p[i, ...] = x * x_padded[sl]
            else:
                p[i - 1, ...] = x * x_padded[sl]
    return p

def get_weights(voxel_sizes, in_shape, _dev='cpu'):
    ndim = len(voxel_sizes)
    num_neigh = 3**ndim-1
    voxel_size_weights = xp.zeros((num_neigh,) + in_shape, dtype=xp.float64, device=_dev)

    for i, ind in enumerate(xp.ndindex(ndim * (3,))):
        if i != (num_neigh // 2):
            offset = xp.asarray(ind) - 1
            vw = voxel_sizes[2] / xp.linalg.norm(offset * voxel_sizes)

            if i < num_neigh // 2:
                voxel_size_weights[i, ...] = vw
            else:
                voxel_size_weights[i - 1, ...] = vw

    return voxel_size_weights


def get_divisors(n):
    """Returns a sorted list of all divisors of a positive integer n."""
    divisors = set()
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            divisors.add(i)
            divisors.add(n // i)
    return sorted(divisors)


def step_size_rule_1(update: int) -> float:
    if update >= 10 and update <= 13:
        new_step_size = 1.0
    elif update > 13:
        new_step_size = 0.5
    return new_step_size


class MaxIteration(callbacks.Callback):
    """
    The organisers try to `Submission(data).run(inf)` i.e. for infinite iterations (until timeout).
    This callback forces stopping after `max_iteration` instead.
    """

    def __init__(self, max_iteration: int, verbose: int = 1):
        super().__init__(verbose)
        self.max_iteration = max_iteration

    def __call__(self, algorithm: Algorithm):
        if algorithm.iteration >= self.max_iteration:
            raise StopIteration


class Submission(Algorithm):
    """
    OSEM algorithm example.
    NB: In OSEM, the multiplicative term cancels in the back-projection of the quotient of measured & estimated data
    (so this is used here for efficiency). Note that a similar optimisation can be used for all algorithms using the Poisson log-likelihood.
    NB: OSEM does not use `data.prior` and thus does not converge to the MAP reference used in PETRIC.
    NB: this example does not use the `sirf.STIR` Poisson objective function.
    NB: see https://github.com/SyneRBI/SIRF-Contribs/tree/master/src/Python/sirf/contrib/BSREM
    """

    def __init__(
        self,
        data: Dataset,
        approx_num_subsets: float = 24.2,  # approximate number of subsets, closest divisor of num_views will be used
        update_objective_interval: int | None = None,
        initial_stepsize: float = 3,
        precond_filter_fwhm_mm: float = 5.0,
        seed: int = 1,
        **kwargs,
    ):
        """
        Initialisation function, setting up data & (hyper)parameters.
        NB: in practice, `num_subsets` should likely be determined from the data.
        This is just an example. Try to modify and improve it!
        """

        np.random.seed(seed)

        self.subset = 0

        # --- setup the number of subsets
        num_views = data.mult_factors.dimensions()[2]
        num_views_divisors = np.array(get_divisors(num_views))
        self._num_subsets = num_views_divisors[
            np.argmin(np.abs(num_views_divisors - approx_num_subsets))
        ]
        self.coprimes = find_coprimes(self._num_subsets)
        if not self.coprimes:
            self.sorted_coprimes = [0] * 100
        else:
            self.sorted_coprimes = sorted(self.coprimes, key=lambda x: min(abs(x - int(0.3*self._num_subsets)), abs(x - int(0.7*self._num_subsets)))) 

        if self._num_subsets not in num_views_divisors:
            raise ValueError(
                f"Number of subsets {self._num_subsets} is not a divisor of {num_views}. Divisors are {num_views_divisors}"
            )
        print(self._num_subsets)

        # --- setup the initial image as a slightly smoothed version of the OSEM image
        self.x = data.OSEM_image.clone()
        self.deltamax = 1e-6*self.x.max()
        self.delta = self.x.get_uniform_copy(self.deltamax)
        self.tmp_im = self.x.get_uniform_copy(1)
        self.tmp_grad = self.tmp_im.clone()
        self.old_x = self.x.clone()
        self.old_g = self.tmp_im.clone()

        self._step_size = initial_stepsize 
        self._subset_number_list = []

        _, _, self._obj_funs = (
            partitioner.data_partition(
                data.acquired_data,
                data.additive_term,
                data.mult_factors,
                self._num_subsets,
                initial_image=data.OSEM_image,
                mode="staggered",
            )
        )

        penalisation_factor = data.prior.get_penalisation_factor()

        # WARNING: modifies prior strength with 1/num_subsets (as currently needed for BSREM implementations)
        data.prior.set_penalisation_factor(penalisation_factor / self._num_subsets)
        data.prior.set_up(data.OSEM_image)
        for f in self._obj_funs:  # add prior evenly to every objective function
            f.set_prior(data.prior)


        self._adjoint_ones = self.x.get_uniform_copy(0)
        for i in range(self._num_subsets):
            self._adjoint_ones += self._obj_funs[i].get_subset_sensitivity(0)
        self.update_filter = STIR.TruncateToCylinderProcessor()
        self.update_filter.apply(self.tmp_im)
        self.tmp_im -= self.x.get_uniform_copy(1)
        self._adjoint_ones.sapyb(1.0, self.tmp_im, -1e-6, out=self._adjoint_ones)

        # self._fov_mask = data.FOV_mask
        # initialize list / ImageData for all subset gradients and sum of gradients
        self._summed_subset_gradients = self.x.get_uniform_copy(0)
        self._subset_gradients = []

        # change these to correspond to final things
        self._complete_gradient_epochs: list[int] = [x for x in range(0, 1000, 2)]
        self._precond_update_epochs: list[int] = [1, 2, 4, 6]
        self._bb_update_epochs: list[int] = [2, 4, 6]
        if "cupy" in xp.__name__:
            self._dev = xp.cuda.Device(0)
        else:
            self._dev = "cpu"

        # what is the difference between voxel sizes and spacing?
        self.kappa = data.kappa.as_array()
        self.epsilon = data.prior.get_epsilon()
        self.penalty_strength = penalisation_factor
        self.gamma = data.prior.get_gamma()
        self.weights = get_weights(xp.asarray(data.OSEM_image.voxel_sizes(), device=self._dev), self.x.shape, _dev=self._dev)
        self.weights *=neighbor_product(xp.asarray(self.kappa, device=self._dev, dtype = xp.float64), _dev=self._dev)


        self._precond_filter = STIR.SeparableGaussianImageFilter()
        self._precond_filter.set_fwhms(
            [precond_filter_fwhm_mm, precond_filter_fwhm_mm, precond_filter_fwhm_mm]
        )
        self._precond_filter.set_up(data.OSEM_image)

        # calculate the initial preconditioner based on the initial image
        self._precond = self.calc_precond(self.x)

        if update_objective_interval is None:
            update_objective_interval = self._num_subsets

        super().__init__(update_objective_interval=update_objective_interval, **kwargs)
        self.configured = True  # required by Algorithm

    @property
    def epoch(self):
        return self.iteration // self._num_subsets

    def calc_precond(
        self,
        x: STIR.ImageData,
    ) -> STIR.ImageData:
        x_sm = self._precond_filter.process(x)

        x_xp = xp.asarray(x_sm.as_array(), device=self._dev, dtype=xp.float64)        
        d, s = neighbor_difference_and_sum(x_xp, padding='edge')
        phi = s + self.gamma * xp.abs(d) + self.epsilon
        tmp = ((s - d + self.epsilon) ** 2) / (phi**3) 
        tmp *= self.weights
        tmp = 3*self.penalty_strength*tmp.sum(axis=0)
        tmp *= x_xp
        self.tmp_im.fill(to_device(tmp, 'cpu'))
        self.tmp_im += self._adjoint_ones
        # x_sm += self.delta
        x_sm.divide(self.tmp_im, out=self.tmp_im)
        # self.tmp_im.multiply(self._fov_mask)
        self.update_filter.apply(self.tmp_im)
        return self.tmp_im.clone()


    def update_all_subset_gradients(self) -> None:
        self._subset_gradients = [lkhd_func.gradient(self.x) for lkhd_func in self._obj_funs]
        self._summed_subset_gradients.fill(np.sum(self._subset_gradients))

    def update(self):

        update_all_subset_gradients = (
            self.iteration % self._num_subsets == 0
        ) and self.epoch in self._complete_gradient_epochs

        update_precond = (
            self.iteration % self._num_subsets == 0
        ) and self.epoch in self._precond_update_epochs

        if self.iteration <= 10:
            self._step_size = 3.
        elif (self.iteration > 10) and (self.epoch <= 2):
            self._step_size = 2.2
        elif self.epoch >= 10:
            self._step_size = 1.0

        if update_precond:
            self._precond.fill(self.calc_precond(self.x))

        if update_all_subset_gradients:
            self.update_all_subset_gradients()
            self.tmp_grad.fill(self._summed_subset_gradients)
            if self.iteration == 0:
                self.old_x.fill(self.x)
                self.old_g.fill(self.tmp_grad)
            
            elif self.epoch in self._bb_update_epochs:
                self.old_x -= self.x
                self.old_g -= self.tmp_grad
                self._precond.multiply(self.old_g, out=self.tmp_im)
                shbb_step = np.abs(self.old_x.dot(self.old_g)/self.old_g.dot(self.tmp_im))
                self.old_x.fill(self.x)
                self.old_g.fill(self.tmp_grad)
                max_ss = 2.5 if self.epoch <= 2 else self._step_size
                self._step_size = np.clip(shbb_step, 0.01, max_ss)
        else:
            if not self._subset_number_list:
                self.create_subset_number_list()

            self.subset = self._subset_number_list.pop()
            self._obj_funs[self.subset].gradient(self.x, out = self.tmp_grad)
            self.tmp_grad -= self._subset_gradients[self.subset]
            self.tmp_grad.sapyb(self._num_subsets, self._summed_subset_gradients, 1, out=self.tmp_grad)
        self.tmp_grad.multiply(self._precond, out=self.tmp_grad)
        self.x.sapyb(1, self.tmp_grad, self._step_size, out=self.x)
        self.x.maximum(0, out=self.x)

    def update_objective(self) -> None:
        """
        NB: The objective value is not required by OSEM nor by PETRIC, so this returns `0`.
        NB: It should be `sum(prompts * log(acq_model.forward(self.x)) - self.x * sensitivity)` across all subsets.
        """

        self.loss.append(0)

    def create_subset_number_list(self):
        if not self.sorted_coprimes:
            self.sorted_coprimes =sorted(self.coprimes, key=lambda x: min(abs(x - int(0.3*self._num_subsets)), abs(x - int(0.7*self._num_subsets)))) 
        generator = self.sorted_coprimes.pop(0)
        self._subset_number_list = [(generator*k)%self._num_subsets for k in range(self._num_subsets)]
        if not self.coprimes:
            self._subset_number_list = [0] * 100

submission_callbacks = []
