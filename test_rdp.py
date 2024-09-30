#!/usr/bin/env python
"""
ANY CHANGES TO THIS FILE ARE IGNORED BY THE ORGANISERS.
Only the `main.py` file may be modified by participants.

This file is not intended for participants to use, except for
the `get_data` function (and possibly `QualityMetrics` class).
It is used by the organisers to run the submissions in a controlled way.
It is included here purely in the interest of transparency.

Usage:
  petric.py [options]

Options:
  --log LEVEL  : Set logging level (DEBUG, [default: INFO], WARNING, ERROR, CRITICAL)
"""
import csv
import logging
import os
from dataclasses import dataclass
from pathlib import Path, PurePath
from time import time

import numpy as np
from skimage.metrics import mean_squared_error as mse
from tensorboardX import SummaryWriter

import sirf.STIR as STIR
from cil.optimisation.algorithms import Algorithm
from cil.optimisation.utilities import callbacks as cil_callbacks
from img_quality_cil_stir import ImageQualityCallback

log = logging.getLogger("petric")
TEAM = os.getenv("GITHUB_REPOSITORY", "SyneRBI/PETRIC-").split("/PETRIC-", 1)[-1]
VERSION = os.getenv("GITHUB_REF_NAME", "")
OUTDIR = Path(f"/o/logs/{TEAM}/{VERSION}" if TEAM and VERSION else "./output")
if not (SRCDIR := Path("/mnt/share/petric")).is_dir():
    SRCDIR = Path("./data")


class Callback(cil_callbacks.Callback):
    """
    CIL Callback but with `self.skip_iteration` checking `min(self.interval, algo.update_objective_interval)`.
    TODO: backport this class to CIL.
    """

    def __init__(self, interval: int = 1 << 31, **kwargs):
        super().__init__(**kwargs)
        self.interval = interval

    def skip_iteration(self, algo: Algorithm) -> bool:
        return (
            algo.iteration % min(self.interval, algo.update_objective_interval) != 0
            and algo.iteration != algo.max_iteration
        )


class SaveIters(Callback):
    """Saves `algo.x` as "iter_{algo.iteration:04d}.hv" and `algo.loss` in `csv_file`"""

    def __init__(self, outdir=OUTDIR, csv_file="objectives.csv", **kwargs):
        super().__init__(**kwargs)
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.csv = csv.writer((self.outdir / csv_file).open("w", buffering=1))
        self.csv.writerow(("iter", "objective"))

    def __call__(self, algo: Algorithm):
        if not self.skip_iteration(algo):
            log.debug("saving iter %d...", algo.iteration)
            algo.x.write(str(self.outdir / f"iter_{algo.iteration:04d}.hv"))
            self.csv.writerow((algo.iteration, algo.get_last_loss()))
            log.debug("...saved")
        if algo.iteration == algo.max_iteration:
            algo.x.write(str(self.outdir / "iter_final.hv"))


class StatsLog(Callback):
    """Log image slices & objective value"""

    def __init__(
        self,
        transverse_slice=None,
        coronal_slice=None,
        vmax=None,
        logdir=OUTDIR,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.transverse_slice = transverse_slice
        self.coronal_slice = coronal_slice
        self.vmax = vmax
        self.x_prev = None
        self.tb = (
            logdir
            if isinstance(logdir, SummaryWriter)
            else SummaryWriter(logdir=str(logdir))
        )

    def __call__(self, algo: Algorithm):
        if self.skip_iteration(algo):
            return
        t = getattr(self, "__time", None) or time()
        log.debug("logging iter %d...", algo.iteration)
        # initialise `None` values
        self.transverse_slice = (
            algo.x.dimensions()[0] // 2
            if self.transverse_slice is None
            else self.transverse_slice
        )
        self.coronal_slice = (
            algo.x.dimensions()[1] // 2
            if self.coronal_slice is None
            else self.coronal_slice
        )
        self.vmax = algo.x.max() if self.vmax is None else self.vmax

        self.tb.add_scalar("objective", algo.get_last_loss(), algo.iteration, t)
        if self.x_prev is not None:
            normalised_change = (algo.x - self.x_prev).norm() / algo.x.norm()
            self.tb.add_scalar(
                "normalised_change", normalised_change, algo.iteration, t
            )
        self.x_prev = algo.x.clone()
        x_arr = algo.x.as_array()
        self.tb.add_image(
            "transverse",
            np.clip(
                x_arr[self.transverse_slice : self.transverse_slice + 1] / self.vmax,
                0,
                1,
            ),
            algo.iteration,
            t,
        )
        self.tb.add_image(
            "coronal",
            np.clip(x_arr[None, :, self.coronal_slice] / self.vmax, 0, 1),
            algo.iteration,
            t,
        )
        log.debug("...logged")


class QualityMetrics(ImageQualityCallback, Callback):
    """From https://github.com/SyneRBI/PETRIC/wiki#metrics-and-thresholds"""

    def __init__(
        self,
        reference_image,
        whole_object_mask,
        background_mask,
        interval: int = 1 << 31,
        **kwargs,
    ):
        # TODO: drop multiple inheritance once `interval` included in CIL
        Callback.__init__(self, interval=interval)
        ImageQualityCallback.__init__(self, reference_image, **kwargs)
        self.whole_object_indices = np.where(whole_object_mask.as_array())
        self.background_indices = np.where(background_mask.as_array())
        self.ref_im_arr = reference_image.as_array()
        self.norm = self.ref_im_arr[self.background_indices].mean()

    def __call__(self, algo: Algorithm):
        if self.skip_iteration(algo):
            return
        t = getattr(self, "__time", None) or time()
        for tag, value in self.evaluate(algo.x).items():
            self.tb_summary_writer.add_scalar(tag, value, algo.iteration, t)

    def evaluate(self, test_im: STIR.ImageData) -> dict[str, float]:
        assert not any(self.filter.values()), "Filtering not implemented"
        test_im_arr = test_im.as_array()
        whole = {
            "RMSE_whole_object": np.sqrt(
                mse(
                    self.ref_im_arr[self.whole_object_indices],
                    test_im_arr[self.whole_object_indices],
                )
            )
            / self.norm,
            "RMSE_background": np.sqrt(
                mse(
                    self.ref_im_arr[self.background_indices],
                    test_im_arr[self.background_indices],
                )
            )
            / self.norm,
        }
        local = {
            f"AEM_VOI_{voi_name}": np.abs(
                test_im_arr[voi_indices].mean() - self.ref_im_arr[voi_indices].mean()
            )
            / self.norm
            for voi_name, voi_indices in sorted(self.voi_indices.items())
        }
        return {**whole, **local}

    def keys(self):
        return ["RMSE_whole_object", "RMSE_background"] + [
            f"AEM_VOI_{name}" for name in sorted(self.voi_indices)
        ]


class MetricsWithTimeout(cil_callbacks.Callback):
    """Stops the algorithm after `seconds`"""

    def __init__(
        self,
        seconds=600,
        outdir=OUTDIR,
        transverse_slice=None,
        coronal_slice=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._seconds = seconds
        self.callbacks = [
            cil_callbacks.ProgressCallback(),
            SaveIters(outdir=outdir),
            (
                tb_cbk := StatsLog(
                    logdir=outdir,
                    transverse_slice=transverse_slice,
                    coronal_slice=coronal_slice,
                )
            ),
        ]
        self.tb = tb_cbk.tb  # convenient access to the underlying SummaryWriter
        self.reset()

    def reset(self, seconds=None):
        self.limit = time() + (self._seconds if seconds is None else seconds)
        self.offset = 0

    def __call__(self, algo: Algorithm):
        if (now := time()) > self.limit + self.offset:
            log.warning("Timeout reached. Stopping algorithm.")
            raise StopIteration
        for c in self.callbacks:
            c.__time = (
                now - self.offset
            )  # privately inject walltime-excluding-petric-callbacks
            c(algo)
        self.offset += time() - now

    @staticmethod
    def mean_absolute_error(y, x):
        return np.mean(np.abs(y, x))


def construct_RDP(penalty_strength, initial_image, kappa, max_scaling=1e-3):
    """
    Construct a smoothed Relative Difference Prior (RDP)

    initial_image: used to determine a smoothing factor (epsilon).
    kappa: used to pass voxel-dependent weights.
    """
    prior = getattr(STIR, "CudaRelativeDifferencePrior", STIR.RelativeDifferencePrior)()
    # need to make it differentiable
    epsilon = initial_image.max() * max_scaling
    prior.set_epsilon(epsilon)
    prior.set_penalisation_factor(penalty_strength)
    prior.set_kappa(kappa)
    prior.set_up(initial_image)
    return prior


@dataclass
class Dataset:
    acquired_data: STIR.AcquisitionData
    additive_term: STIR.AcquisitionData
    mult_factors: STIR.AcquisitionData
    OSEM_image: STIR.ImageData
    prior: STIR.RelativeDifferencePrior
    kappa: STIR.ImageData
    reference_image: STIR.ImageData | None
    whole_object_mask: STIR.ImageData | None
    background_mask: STIR.ImageData | None
    voi_masks: dict[str, STIR.ImageData]
    path: PurePath


def get_data(srcdir=".", outdir=OUTDIR, sirf_verbosity=0):
    """
    Load data from `srcdir`, constructs prior and return as a `Dataset`.
    Also redirects sirf.STIR log output to `outdir`.
    """
    srcdir = Path(srcdir)
    outdir = Path(outdir)
    STIR.set_verbosity(sirf_verbosity)  # set to higher value to diagnose problems
    STIR.AcquisitionData.set_storage_scheme("memory")  # needed for get_subsets()

    _ = STIR.MessageRedirector(
        str(outdir / "info.txt"),
        str(outdir / "warnings.txt"),
        str(outdir / "errors.txt"),
    )
    acquired_data = STIR.AcquisitionData(str(srcdir / "prompts.hs"))
    additive_term = STIR.AcquisitionData(str(srcdir / "additive_term.hs"))
    mult_factors = STIR.AcquisitionData(str(srcdir / "mult_factors.hs"))
    OSEM_image = STIR.ImageData(str(srcdir / "OSEM_image.hv"))
    kappa = STIR.ImageData(str(srcdir / "kappa.hv"))
    if (penalty_strength_file := (srcdir / "penalisation_factor.txt")).is_file():
        penalty_strength = float(np.loadtxt(penalty_strength_file))
    else:
        penalty_strength = 1 / 700  # default choice
    prior = construct_RDP(penalty_strength, OSEM_image, kappa)

    def get_image(fname):
        if (source := srcdir / "PETRIC" / fname).is_file():
            return STIR.ImageData(str(source))
        return None  # explicit to suppress linter warnings

    reference_image = get_image("reference_image.hv")
    whole_object_mask = get_image("VOI_whole_object.hv")
    background_mask = get_image("VOI_background.hv")
    voi_masks = {
        voi.stem[4:]: STIR.ImageData(str(voi))
        for voi in (srcdir / "PETRIC").glob("VOI_*.hv")
        if voi.stem[4:] not in ("background", "whole_object")
    }

    return Dataset(
        acquired_data,
        additive_term,
        mult_factors,
        OSEM_image,
        prior,
        kappa,
        reference_image,
        whole_object_mask,
        background_mask,
        voi_masks,
        srcdir.resolve(),
    )


if SRCDIR.is_dir():
    # create list of existing data
    # NB: `MetricsWithTimeout` initialises `SaveIters` which creates `outdir`
    data_dirs_metrics = [
        (
            SRCDIR / "Siemens_mMR_NEMA_IQ",
            OUTDIR / "mMR_NEMA",
            [
                MetricsWithTimeout(
                    outdir=OUTDIR / "mMR_NEMA", transverse_slice=72, coronal_slice=109
                )
            ],
        ),
        (
            SRCDIR / "NeuroLF_Hoffman_Dataset",
            OUTDIR / "NeuroLF_Hoffman",
            [
                MetricsWithTimeout(
                    outdir=OUTDIR / "NeuroLF_Hoffman", transverse_slice=72
                )
            ],
        ),
        (
            SRCDIR / "Siemens_Vision600_thorax",
            OUTDIR / "Vision600_thorax",
            [MetricsWithTimeout(outdir=OUTDIR / "Vision600_thorax")],
        ),
        (
            SRCDIR / "Siemens_mMR_ACR",
            OUTDIR / "Siemens_mMR_ACR",
            [MetricsWithTimeout(outdir=OUTDIR / "Siemens_mMR_ACR")],
        ),
    ]
else:
    log.warning("Source directory does not exist: %s", SRCDIR)
    data_dirs_metrics = [(None, None, [])]  # type: ignore

if __name__ == "__main__":

    from rdp import RDP

    import array_api_compat.numpy as xp

    image_type = "osem"  # osem or ref

    for ds in range(0, 1):
        srcdir, outdir, metrics = data_dirs_metrics[ds]
        print(srcdir)
        data = get_data(srcdir=srcdir, outdir=outdir)

        if image_type == "osem":
            x = data.OSEM_image
        elif image_type == "ref":
            x = data.reference_image
        else:
            raise ValueError("image must be either 'osem' or 'ref'")

        # %%

        # the RDP implementation can use numpy or cupy. for the latter you have to adjust the device
        python_prior = RDP(
            x.shape,
            xp,
            "cpu",
            xp.asarray(x.spacing, device="cpu"),
            eps=data.prior.get_epsilon(),
            gamma=data.prior.get_gamma(),
        )
        python_prior.kappa = data.kappa.as_array()
        python_prior.scale = data.prior.get_penalisation_factor()

        prior_val = data.prior(x)
        print(f"data.prior({image_type}) = {prior_val}")

        # %%
        # get gradient
        prior_grad = data.prior.gradient(x)

        # %%
        # get the diagonal of the Hessian of the RDP (as numpy array)
        # precond_filter = STIR.SeparableGaussianImageFilter()
        # precond_filter.set_fwhms([5.0, 5.0, 5.0])
        # precond_filter.set_up(data.OSEM_image)
        # x_sm = precond_filter.process(x)

        x_sm = x

        h = x.get_uniform_copy(0)
        inv_h = x.get_uniform_copy(0)

        tmp = python_prior.diag_hessian(x_sm.as_array())

        h.fill(tmp)
        inv_h.fill(np.nan_to_num(1 / tmp, posinf=0))

        # %%

        from sirf.contrib.partitioner import partitioner

        data_sub, acq_models, likelihood_funcs = partitioner.data_partition(
            data.acquired_data,
            data.additive_term,
            data.mult_factors,
            1,
            initial_image=data.OSEM_image,
            mode="staggered",
        )

        logL_func = likelihood_funcs[0]

        logL_val = logL_func(x)
        logL_grad = logL_func.gradient(x)

        adjoint_ones = logL_func.get_subset_sensitivity(0)

        fov_mask = x.get_uniform_copy(0)
        tmp = 1.0 * (adjoint_ones.as_array() > 0)
        fov_mask.fill(tmp)
        adjoint_ones += 1e-6 * (-fov_mask + 1.0)

        P_logL = fov_mask * (x / adjoint_ones)

        P_harm = x / (adjoint_ones + 2 * h)

        num_iter = 200

        step_sizes = [1.0, 2.0]

        cost = np.zeros((len(step_sizes), num_iter))

        ref_cost = logL_val - prior_val

        for j, alpha in enumerate(step_sizes):
            print(f"alpha {alpha:.2E}")
            # xnew_prior = x - alpha * fov_mask * prior_grad * inv_h
            # xnew_prior.maximum(0, out=xnew_prior)
            # print(f"delta neg. prior {(prior_val - data.prior(xnew_prior)):.2E}")

            # xnew_logL = x + alpha * fov_mask * logL_grad * P_logL
            # xnew_logL.maximum(0, out=xnew_logL)
            # print(f"delta logL       {(logL_func(xnew_logL) - logL_val):.2E}")

            for i in range(num_iter):
                xnew = x + alpha * fov_mask * (logL_grad - prior_grad) * P_harm
                xnew.maximum(0, out=xnew)

                cost[j, i] = logL_func(xnew) - data.prior(xnew)

                delta_cost = cost[j, i] - ref_cost

                print(f"{i:04} delta cost {(delta_cost):.2E}")

                if delta_cost < 0:
                    break

                x = xnew