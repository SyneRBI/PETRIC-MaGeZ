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

import numpy as np
import array_api_compat.cupy as xp
from array_api_compat import to_device

# import pure python re-implementation of the RDP -> only used to get diagonal of the RDP Hessian!
from rdp import RDP

from petric import Dataset


def get_divisors(n):
    """Returns a sorted list of all divisors of a positive integer n."""
    divisors = set()
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            divisors.add(i)
            divisors.add(n // i)
    return sorted(divisors)


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
        step_size_factor: float = 1.0,  # multiplicative factor to increase / decrease default step sizes
        num_subsets: int | None = None,
        update_objective_interval: int | None = None,
        complete_gradient_epochs: None | list[int] = None,
        precond_update_epochs: None | list[int] = None,
        precond_hessian_factor: float = 32.0,
        verbose: bool = False,
        **kwargs,
    ):
        """
        Initialisation function, setting up data & (hyper)parameters.
        NB: in practice, `num_subsets` should likely be determined from the data.
        This is just an example. Try to modify and improve it!
        """

        self._verbose = verbose
        self.subset = 0

        # --- setup the number of subsets

        num_views = data.mult_factors.dimensions()[2]
        num_views_divisors = np.array(get_divisors(num_views))
        if num_subsets is None:
            self._num_subsets = num_views_divisors[
                np.argmin(np.abs(num_views_divisors - 25))
            ]
        else:
            self._num_subsets = num_subsets

        if self._num_subsets not in num_views_divisors:
            raise ValueError(
                f"Number of subsets {self._num_subsets} is not a divisor of {num_views}. Divisors are {num_views_divisors}"
            )

        if self._verbose:
            print(f"num_subsets: {self._num_subsets}")

        # --- setup the initial image as a slightly smoothed version of the OSEM image
        self.x = data.OSEM_image.clone()

        self._update = 0
        self._step_size_factor = step_size_factor
        self._step_size = self._step_size_factor * 2.0
        self._subset_number_list = []
        self._precond_hessian_factor = precond_hessian_factor

        self._data_sub, self._acq_models, self._subset_likelihood_funcs = (
            partitioner.data_partition(
                data.acquired_data,
                data.additive_term,
                data.mult_factors,
                self._num_subsets,
                initial_image=data.OSEM_image,
                mode="staggered",
            )
        )

        # WARNING: modifies prior strength with 1/num_subsets (as currently needed for BSREM implementations)
        data.prior.set_penalisation_factor(
            data.prior.get_penalisation_factor() / self._num_subsets
        )
        data.prior.set_up(data.OSEM_image)

        self._subset_prior_fct = data.prior

        self._adjoint_ones = self.x.get_uniform_copy(0)

        for i in range(self._num_subsets):
            if self._verbose:
                print(f"Calculating subset {i} sensitivity")
            self._adjoint_ones += self._subset_likelihood_funcs[
                i
            ].get_subset_sensitivity(0)

        self._fov_mask = self.x.get_uniform_copy(0)
        # tmp = 1.0 * (self._adjoint_ones.as_array() > 0)
        tmp = 1.0 * (data.OSEM_image.as_array() > 0)
        self._fov_mask.fill(tmp)

        # add a small number in the adjoint ones outside the FOV to avoid NaN in division
        self._adjoint_ones += 1e-6 * (-self._fov_mask + 1.0)

        # initialize list / ImageData for all subset gradients and sum of gradients
        self._summed_subset_gradients = self.x.get_uniform_copy(0)
        self._subset_gradients = []

        if complete_gradient_epochs is None:
            self._complete_gradient_epochs: list[int] = [x for x in range(0, 1000, 2)]
        else:
            self._complete_gradient_epochs = complete_gradient_epochs

        if precond_update_epochs is None:
            self._precond_update_epochs: list[int] = [1, 2, 3]
        else:
            self._precond_update_epochs = precond_update_epochs

        # setup python re-implementation of the RDP
        # only used to get the diagonal of the RDP Hessian for preconditioning!
        # (diag of RDP Hessian is not available in SIRF yet)
        if "cupy" in xp.__name__:
            self._dev = xp.cuda.Device(0)
        else:
            self._dev = "cpu"

        self._python_prior = RDP(
            data.OSEM_image.shape,
            xp,
            self._dev,
            xp.asarray(data.OSEM_image.spacing, device=self._dev),
            eps=data.prior.get_epsilon(),
            gamma=data.prior.get_gamma(),
        )
        self._python_prior.kappa = xp.asarray(data.kappa.as_array(), device=self._dev)
        self._python_prior.scale = data.prior.get_penalisation_factor()

        self._precond_filter = STIR.SeparableGaussianImageFilter()
        self._precond_filter.set_fwhms([5.0, 5.0, 5.0])
        self._precond_filter.set_up(data.OSEM_image)

        # calculate the initial preconditioner based on the initial image
        self._precond = self.calc_precond(self.x)

        if update_objective_interval is None:
            update_objective_interval = self._num_subsets

        super().__init__(update_objective_interval=update_objective_interval, **kwargs)
        self.configured = True  # required by Algorithm

    @property
    def epoch(self):
        return self._update // self._num_subsets

    def update_step_size(self):
        if self.epoch <= 3:
            self._step_size = self._step_size_factor * 2.0
        elif self.epoch > 3 and self.epoch <= 8:
            self._step_size = self._step_size_factor * 1.5
        elif self.epoch > 8 and self.epoch <= 12:
            self._step_size = self._step_size_factor * 1.0
        else:
            self._step_size = self._step_size_factor * 0.5

        if self._verbose:
            print(self._update, self.epoch, self._step_size)

    def calc_precond(
        self,
        x: STIR.ImageData,
        delta_rel: float = 1e-6,
    ) -> STIR.ImageData:

        # generate a smoothed version of the input image
        # to avoid high values, especially in first and last slices
        x_sm = self._precond_filter.process(x)
        delta = delta_rel * x_sm.max()

        prior_diag_hess = x_sm.get_uniform_copy(0)
        prior_diag_hess.fill(
            to_device(
                self._python_prior.diag_hessian(
                    xp.asarray(x_sm.as_array(), device=self._dev)
                ),
                "cpu",
            )
        )

        precond = (
            self._fov_mask
            * (x_sm + delta)
            / (
                self._adjoint_ones
                + self._precond_hessian_factor * prior_diag_hess * x_sm
            )
        )

        return precond

    def update_all_subset_gradients(self) -> None:

        self._summed_subset_gradients = self.x.get_uniform_copy(0)
        self._subset_gradients = []

        subset_prior_gradient = self._subset_prior_fct.gradient(self.x)

        # remember that the objective has to be maximized
        # posterior = log likelihood - log prior ("minus" instead of "plus"!)
        for i in range(self._num_subsets):
            self._subset_gradients.append(
                self._subset_likelihood_funcs[i].gradient(self.x)
                - subset_prior_gradient
            )
            self._summed_subset_gradients += self._subset_gradients[i]

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
            self._precond = self.calc_precond(self.x)

        if update_all_subset_gradients:
            if self._verbose:
                print(
                    f"  {self._update}, {self.subset}, recalculating all subset gradients"
                )
            self.update_all_subset_gradients()
            approximated_gradient = self._summed_subset_gradients
        else:
            if self._subset_number_list == []:
                self.create_subset_number_list()

            self.subset = self._subset_number_list.pop()
            if self._verbose:
                print(f" {self.update}, {self.subset}, subset gradient update")

            subset_prior_gradient = self._subset_prior_fct.gradient(self.x)

            # remember that the objective has to be maximized
            # posterior = log likelihood - log prior ("minus" instead of "plus"!)
            approximated_gradient = (
                self._num_subsets
                * (
                    (
                        self._subset_likelihood_funcs[self.subset].gradient(self.x)
                        - subset_prior_gradient
                    )
                    - self._subset_gradients[self.subset]
                )
                + self._summed_subset_gradients
            )

        ### Objective has to be maximized -> "+" for gradient ascent
        self.x = self.x + self._step_size * self._precond * approximated_gradient

        # enforce non-negative constraint
        self.x.maximum(0, out=self.x)

        self._update += 1

    def update_objective(self) -> None:
        """
        NB: The objective value is not required by OSEM nor by PETRIC, so this returns `0`.
        NB: It should be `sum(prompts * log(acq_model.forward(self.x)) - self.x * sensitivity)` across all subsets.
        """

        self.loss.append(0)

    def create_subset_number_list(self):
        tmp = np.arange(self._num_subsets)
        np.random.shuffle(tmp)
        self._subset_number_list = tmp.tolist()


submission_callbacks = [MaxIteration(300)]
