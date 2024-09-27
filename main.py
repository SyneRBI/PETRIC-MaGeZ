"""Main file to modify for submissions.

Once renamed or symlinked as `main.py`, it will be used by `petric.py` as follows:

>>> from main import Submission, submission_callbacks
>>> from petric import data, metrics
>>> algorithm = Submission(data)
>>> algorithm.run(np.inf, callbacks=metrics + submission_callbacks)
"""

import math
from cil.optimisation.algorithms import Algorithm
from cil.optimisation.utilities import callbacks
from petric import Dataset

# from sirf.contrib.partitioner.partitioner import partition_indices
from sirf.contrib.partitioner import partitioner

import numpy as np


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
    (so this is used here for efficiency).
    A similar optimisation can be used for all algorithms using the Poisson log-likelihood.
    NB: OSEM does not use `data.prior` and thus does not converge to the MAP reference used in PETRIC.
    NB: this example does not use the `sirf.STIR` Poisson objective function.
    NB: see https://github.com/SyneRBI/SIRF-Contribs/tree/master/src/Python/sirf/contrib/BSREM
    """

    def __init__(
        self,
        data: Dataset,
        approx_num_subsets: int = 28,
        update_objective_interval: int | None = None,
        base_gamma: float = 1.0,
        rho: float = 1.0,
        seed: int = 1,
        **kwargs,
    ):
        """
        Initialisation function, setting up data & (hyper)parameters.
        NB: in practice, `num_subsets` should likely be determined from the data.
        This is just an example. Try to modify and improve it!
        """

        np.random.seed(seed)

        self.acquisition_models = []
        self.prompts = []
        self.sensitivities = []
        self.subset = 0
        self.x = data.OSEM_image.clone()
        self.prior = data.prior

        num_views = data.mult_factors.dimensions()[2]
        num_views_divisors = np.array(get_divisors(num_views))
        self.num_subsets = num_views_divisors[
            np.argmin(np.abs(num_views_divisors - approx_num_subsets))
        ]

        self.y = []

        self.prompts, self.acquisition_models, _ = partitioner.data_partition(
            data.acquired_data,
            data.additive_term,
            data.mult_factors,
            self.num_subsets,
            initial_image=data.OSEM_image,
            mode="staggered",
        )

        self.z = self.x.get_uniform_copy(0)

        self.rho = rho
        self.gamma = base_gamma / self.x.max()

        self.S_As = []

        Ts_np = np.zeros((self.num_subsets,) + self.x.shape)

        for i in range(self.num_subsets):
            # we need to use the linear part of the acq. subset model
            # otherwise forward() includes already the additive term
            acqm = self.acquisition_models[i].get_linear_acquisition_model()
            ones_subset_sino = self.prompts[i].get_uniform_copy(1)

            self.y.append(
                -(self.prompts[i] / self.acquisition_models[i].forward(self.x)) + 1
            )

            self.z += acqm.backward(self.y[i])

            # calculate step sizes S_As
            ones_image = self.x.get_uniform_copy(1)
            tmp = acqm.forward(ones_image)

            S_A = tmp.power(-1) * (self.gamma * self.rho)

            # clip inf values
            max_S_A = S_A.as_array()[S_A.as_array() != np.inf].max()
            S_A.minimum(max_S_A, out=S_A)
            # np.save(f"S_A_{i}", S_A.as_array())
            self.S_As.append(S_A)

            # calcualte Ts
            tmp2 = acqm.backward(ones_subset_sino)
            Ts_np[i, ...] = (
                self.rho / (self.gamma * self.num_subsets)
            ) / tmp2.as_array()

        Ts_np = np.nan_to_num(Ts_np, posinf=0)
        self.T = self.x.get_uniform_copy(0)
        self.T.fill(Ts_np.min(0))
        # clip inf values
        max_T = self.T.as_array()[self.T.as_array() != np.inf].max()
        self.T.minimum(max_T, out=self.T)

        # derive FOV mask and multiply step size T with it
        self.fov_mask = self.x.get_uniform_copy(0)
        tmp = 1.0 * (data.OSEM_image.as_array() > 0)
        self.fov_mask.fill(tmp)
        self.T *= self.fov_mask
        np.save("T", self.T.as_array())

        self.zbar = self.z.clone()
        self.grad_h = None

        self.subset_number_list = []

        if update_objective_interval is None:
            update_objective_interval = self.num_subsets

        super().__init__(update_objective_interval=update_objective_interval, **kwargs)
        self.configured = True  # required by Algorithm

    def update(self):
        include_prior = True

        if self.subset_number_list == []:
            self.create_subset_number_list()

        self.subset = self.subset_number_list.pop()

        if self.grad_h is None:
            self.grad_h = self.fov_mask * self.prior.gradient(self.x)

        if include_prior:
            q = self.zbar + self.grad_h
        else:
            q = self.zbar

        self.x = self.x - self.T * q
        self.x.maximum(0, out=self.x)

        np.save(
            f"x_{self.num_subsets - len(self.subset_number_list)}",
            self.x.as_array(),
        )

        if include_prior:
            grad_h_new = self.fov_mask * self.prior.gradient(self.x)
            xbar = self.x + self.T * (self.grad_h - grad_h_new)
            self.grad_h = grad_h_new
        else:
            xbar = self.x

        # forward step, remember that acq_model.forward includes the additive term
        y_plus = self.y[self.subset] + self.S_As[self.subset] * (
            self.acquisition_models[self.subset].forward(xbar)
        )

        # prox of convex conjugate of negative Poisson logL
        tmp = (y_plus - 1) * (y_plus - 1) + 4 * self.S_As[self.subset] * self.prompts[
            self.subset
        ]
        tmp.sqrt(out=tmp)
        y_plus = 0.5 * (y_plus + 1 - tmp)

        delta_z = self.acquisition_models[self.subset].backward(
            y_plus - self.y[self.subset]
        )

        np.save(
            f"delta_z_{self.num_subsets - len(self.subset_number_list)}",
            delta_z.as_array(),
        )

        np.save(
            f"y_plus_{self.num_subsets - len(self.subset_number_list)}",
            y_plus.as_array(),
        )

        self.y[self.subset] = y_plus

        self.z = self.z + delta_z
        self.zbar = self.z + self.num_subsets * delta_z

        print(self.x.min(), self.x.max())

    def update_objective(self):
        """
        NB: The objective value is not required by OSEM nor by PETRIC, so this returns `0`.
        NB: It should be `sum(prompts * log(acq_model.forward(self.x)) - self.x * sensitivity)` across all subsets.
        """
        return 0

    def create_subset_number_list(self):
        tmp = np.arange(self.num_subsets)
        np.random.shuffle(tmp)
        self.subset_number_list = tmp.tolist()


submission_callbacks = [MaxIteration(300)]
