"""Main file to modify for submissions.

Once renamed or symlinked as `main.py`, it will be used by `petric.py` as follows:

>>> from main import Submission, submission_callbacks
>>> from petric import data, metrics
>>> algorithm = Submission(data)
>>> algorithm.run(np.inf, callbacks=metrics + submission_callbacks)
"""

import sirf.STIR as STIR
from cil.optimisation.algorithms import Algorithm
from cil.optimisation.utilities import callbacks
from petric import Dataset

# from sirf.contrib.partitioner.partitioner import partition_indices
from sirf.contrib.partitioner import partitioner

import numpy as np


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
        num_subsets: int = 28,
        update_objective_interval: int = 10,
        base_gamma: float = 1.0,
        rho: float = 1.0,
        **kwargs
    ):
        """
        Initialisation function, setting up data & (hyper)parameters.
        NB: in practice, `num_subsets` should likely be determined from the data.
        This is just an example. Try to modify and improve it!
        """
        self.acquisition_models = []
        self.prompts = []
        self.sensitivities = []
        self.subset = 0
        self.x = data.OSEM_image.clone()
        self.prior = data.prior

        self.num_subsets = num_subsets

        # find views in each subset
        # (note that SIRF can currently only do subsets over views)
        views = data.mult_factors.dimensions()[2]

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

        Ts = []
        self.S_As = []

        ones_image = self.x.get_uniform_copy(1)

        Ts_np = np.zeros((self.num_subsets,) + self.x.shape)

        for i in range(num_subsets):
            # we need to use the linear part of the acq. subset model
            # otherwise forward() includes already the additive term
            acqm = self.acquisition_models[i].get_linear_acquisition_model()
            ones_subset_sino = self.prompts[i].get_uniform_copy(1)

            self.y.append(-(self.prompts[i] / acqm.forward(self.x)) + 1)

            self.z += acqm.backward(self.y[i])

            # calculate step sizes S_As
            tmp = acqm.forward(ones_image)

            self.S_As.append(tmp.power(-1) * (self.gamma * self.rho))

            # calcualte Ts
            tmp2 = acqm.backward(ones_subset_sino)
            Ts_np[i, ...] = (
                self.rho / (self.gamma * self.num_subsets)
            ) / tmp2.as_array()

        self.T = self.x.get_uniform_copy(0)
        self.T.fill(Ts_np.min(0))

        self.zbar = self.z.clone()
        self.grad_h = None

        super().__init__(update_objective_interval=update_objective_interval, **kwargs)
        self.configured = True  # required by Algorithm

    def update(self):
        if self.grad_h is None:
            self.grad_h = self.prior.gradient(self.x)

        q = self.zbar + self.grad_h

        self.x = self.x - self.T * q
        self.x.maximum(0, out=self.x)

        grad_h_new = self.prior.gradient(self.x)

        xbar = self.x + self.T * (self.grad_h - grad_h_new)
        self.grad_h = grad_h_new

        # forward step, remember that acq_model.forward includes the additive term
        y_plus = self.y[self.subset] + self.S_As[self.subset] * (
            self.acquisition_models[self.subset].forward(xbar)
        )

        # prox of convex conjugate of negative Poisson logL
        tmp = (y_plus - 1) * (y_plus - 1) + 4 * self.S_As[self.subset] * self.data[
            self.subset
        ]
        tmp.sqrt(out=tmp)
        y_plus = 0.5 * (y_plus + 1 - tmp)

        delta_z = self.acquisition_models[self.subset].backward(
            y_plus - self.y[self.subset]
        )

        self.z = self.z + delta_z
        self.zbar = self.z + self.num_subsets * delta_z

        self.subset = (self.subset + 1) % len(self.prompts)

    def update_objective(self):
        """
        NB: The objective value is not required by OSEM nor by PETRIC, so this returns `0`.
        NB: It should be `sum(prompts * log(acq_model.forward(self.x)) - self.x * sensitivity)` across all subsets.
        """
        return 0


submission_callbacks = [MaxIteration(660)]
