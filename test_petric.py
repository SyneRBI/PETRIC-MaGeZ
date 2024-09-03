#!/usr/bin/env python
"""
ANY CHANGES TO THIS FILE ARE IGNORED BY THE ORGANISERS.
Only the `main.py` file may be modified by participants.

This file is not intended for participants to use.
It is used by the organisers to run the submissions in a controlled way.
It is included here purely in the interest of transparency.

Usage:
  petric.py [options]

Options:
  --log LEVEL                           set logging level (DEBUG, [default: INFO], WARNING, ERROR, CRITICAL).
  --step_size=<ss>                      step size [default: 1.0].
  --data_set=<ds>                       number of data set ([default: 0], 1, 2).
  --num_iter=<ni>                       number of iterations [default: 100].
  --num_subsets=<ns>                    number of iterations [default: -1].
  --metric_period=<mf>                  period for updating metric [default: 10].
  --complete_gradient_epochs=<numbers>  space separated list of epochs to compute the complete gradient [default: None].
  --precond_update_epochs=<numbers>     space separated list of epochs to update the preconditioner [default: None].
  --batch_mode                          run in batch mode (auto loop over settings).        
"""
import csv
import logging
import os
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from time import time
from datetime import datetime

import numpy as np
from skimage.metrics import mean_squared_error as mse
from tensorboardX import SummaryWriter

import sirf.STIR as STIR
from cil.optimisation.algorithms import Algorithm
from cil.optimisation.utilities import callbacks as cbks
from img_quality_cil_stir import ImageQualityCallback

from main import Submission, submission_callbacks

log = logging.getLogger("petric")
TEAM = os.getenv("GITHUB_REPOSITORY", "SyneRBI/PETRIC-").split("/PETRIC-", 1)[-1]
VERSION = os.getenv("GITHUB_REF_NAME", "")
OUTDIR = Path(f"/o/logs/{TEAM}/{VERSION}" if TEAM and VERSION else "./output_magez")
if not (SRCDIR := Path("/mnt/share/petric")).is_dir():
    SRCDIR = Path("./data")


class SaveIters(cbks.Callback):
    """Saves `algo.x` as "iter_{algo.iteration:04d}.hv" and `algo.loss` in `csv_file`"""

    def __init__(self, verbose=1, outdir=OUTDIR, csv_file="objectives.csv"):
        super().__init__(verbose)
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.csv = csv.writer((self.outdir / csv_file).open("w", buffering=1))
        self.csv.writerow(("iter", "objective"))

    def __call__(self, algo: Algorithm):
        if (
            algo.iteration % algo.update_objective_interval == 0
            or algo.iteration == algo.max_iteration
        ):
            log.debug("saving iter %d...", algo.iteration)
            algo.x.write(str(self.outdir / f"iter_{algo.iteration:04d}.hv"))
            self.csv.writerow((algo.iteration, algo.get_last_loss()))
            log.debug("...saved")
        if algo.iteration == algo.max_iteration:
            algo.x.write(str(self.outdir / "iter_final.hv"))


class StatsLog(cbks.Callback):
    """Log image slices & objective value"""

    def __init__(
        self,
        verbose=1,
        transverse_slice=None,
        coronal_slice=None,
        vmax=None,
        logdir=OUTDIR,
    ):
        super().__init__(verbose)
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
        if (
            algo.iteration % algo.update_objective_interval != 0
            and algo.iteration != algo.max_iteration
        ):
            return
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

        self.tb.add_scalar("objective", algo.get_last_loss(), algo.iteration)
        if self.x_prev is not None:
            normalised_change = (algo.x - self.x_prev).norm() / algo.x.norm()
            self.tb.add_scalar("normalised_change", normalised_change, algo.iteration)
        self.x_prev = algo.x.clone()
        self.tb.add_image(
            "transverse",
            np.clip(
                algo.x.as_array()[self.transverse_slice : self.transverse_slice + 1]
                / self.vmax,
                0,
                1,
            ),
            algo.iteration,
        )
        self.tb.add_image(
            "coronal",
            np.clip(algo.x.as_array()[None, :, self.coronal_slice] / self.vmax, 0, 1),
            algo.iteration,
        )
        log.debug("...logged")


class QualityMetrics(ImageQualityCallback):
    """From https://github.com/SyneRBI/PETRIC/wiki#metrics-and-thresholds"""

    def __init__(self, reference_image, whole_object_mask, background_mask, **kwargs):
        super().__init__(reference_image, **kwargs)
        self.whole_object_indices = np.where(whole_object_mask.as_array())
        self.background_indices = np.where(background_mask.as_array())
        self.ref_im_arr = reference_image.as_array()
        self.norm = self.ref_im_arr[self.background_indices].mean()

    def __call__(self, algo: Algorithm):
        iteration = algo.iteration
        if (
            iteration % algo.update_objective_interval != 0
            and iteration != algo.max_iteration
        ):
            return
        for tag, value in self.evaluate(algo.x).items():
            self.tb_summary_writer.add_scalar(tag, value, iteration)

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


class MetricsWithTimeout(cbks.Callback):
    """Stops the algorithm after `seconds`"""

    def __init__(
        self,
        seconds=3000,
        outdir=OUTDIR,
        transverse_slice=None,
        coronal_slice=None,
        verbose=1,
    ):
        super().__init__(verbose)
        self._seconds = seconds
        self.callbacks = [
            cbks.ProgressCallback(),
            SaveIters(outdir=outdir),
            (
                tb_cbk := StatsLog(
                    logdir=outdir,
                    transverse_slice=transverse_slice,
                    coronal_slice=coronal_slice,
                )
            ),
        ]
        self.tb = tb_cbk.tb
        self.reset()

    def reset(self, seconds=None):
        self.limit = time() + (self._seconds if seconds is None else seconds)

    def __call__(self, algo: Algorithm):
        if (now := time()) > self.limit:
            log.warning("Timeout reached. Stopping algorithm.")
            raise StopIteration
        if self.callbacks:
            for c in self.callbacks:
                c(algo)
            self.limit += time() - now

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
    )


def test_petric(ds: int, num_iter: int, suffix: str = "", **kwargs):

    # get arguments and values such that we can dump them in the outdir
    frame = inspect.currentframe()
    arguments = inspect.getargvalues(frame).args
    values = inspect.getargvalues(frame).locals
    arg_dict = {arg: values[arg] for arg in arguments}

    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d-%H%M%S")

    # sdir_name = f"{formatted_datetime}_ss_{step_size}_n_{num_iter}_subs_{num_subsets}_phf_{precond_hessian_factor}"
    sdir_name = f"{formatted_datetime}_{suffix}"

    if ds == 0:
        srcdir = SRCDIR / "Siemens_mMR_NEMA_IQ"
        outdir = OUTDIR / "mMR_NEMA" / sdir_name
    elif ds == 1:
        srcdir = SRCDIR / "NeuroLF_Hoffman_Dataset"
        outdir = OUTDIR / "NeuroLF_Hoffman" / sdir_name
    elif ds == 2:
        srcdir = SRCDIR / "Siemens_Vision600_thorax"
        outdir = OUTDIR / "Vision600_thorax" / sdir_name
    elif ds == 3:
        srcdir = SRCDIR / "Siemens_mMR_ACR"
        outdir = OUTDIR / "Siemens_mMR_ACR" / sdir_name
    else:
        raise ValueError(f"Unknown data set {ds}")

    metrics = [MetricsWithTimeout(outdir=outdir)]

    #########################################################
    #########################################################
    print(f"\n{srcdir}")
    print("====================================")
    data = get_data(srcdir=srcdir, outdir=outdir)

    #########################################################
    #########################################################

    # dump arguments to json file
    with open(outdir / "args.json", "w") as f:
        json.dump(arg_dict, f, indent=4)

    ########################################################
    ########################################################
    metrics_with_timeout = metrics[0]
    if data.reference_image is not None:
        metrics_with_timeout.callbacks.append(
            QualityMetrics(
                data.reference_image,
                data.whole_object_mask,
                data.background_mask,
                tb_summary_writer=metrics_with_timeout.tb,
                voi_mask_dict=data.voi_masks,
            )
        )
    metrics_with_timeout.reset()  # timeout from now
    ########################################################
    ########################################################

    algo = Submission(data=data, **kwargs)
    algo.run(num_iter, callbacks=metrics + submission_callbacks)

    del algo
    del data


if __name__ == "__main__":
    from docopt import docopt

    args = docopt(__doc__)

    print(args)

    if args["--complete_gradient_epochs"] == "None":
        complete_gradient_epochs = None
    else:
        complete_gradient_epochs = list(
            map(int, args["--complete_gradient_epochs"].split(" "))
        )

    if args["--precond_update_epochs"] == "None":
        precond_update_epochs = None
    else:
        precond_update_epochs = list(
            map(int, args["--precond_update_epochs"].split(" "))
        )

    if not args["--batch_mode"]:
        logging.basicConfig(level=getattr(logging, args["--log"].upper()))

        test_petric(
            step_size=float(args["--step_size"]),
            ds=int(args["--data_set"]),
            num_iter=int(args["--num_iter"]),
            num_subsets=int(args["--num_subsets"]),
            complete_gradient_epochs=complete_gradient_epochs,
            precond_update_epochs=precond_update_epochs,
        )
    else:
        # for i in range(4):
        for i in [0, 1, 3, 2]:
            for ns in [25, 50, 10]:
                test_petric(
                    ds=i, num_iter=300, suffix=f"num_sub_{ns}", approx_num_subsets=ns
                )
