"""script to compare SPDHG, SGD and SVRG for PET reconstruction with quad. and RDP prior"""

#  CUDA_VISIBLE_DEVICES=1 python sim_ablation.py --true_counts 1e7 --beta_rel 16 --num_epochs 10 --num_subsets 27 --precond_type 2 --phantom_type 1 --method SVRG --step_size_rule bb --eta=0.01 --init_step_size 1.0
# %%
from __future__ import annotations

import sys
from pathlib import Path

if not str(Path("..").resolve()) in sys.path:
    sys.path.insert(0, str(Path("..").resolve()))

try:
    import array_api_compat.cupy as xp
except ImportError:
    import array_api_compat.numpy as xp

import json
import parallelproj
import array_api_compat.numpy as np
import matplotlib.pyplot as plt

from array_api_compat import to_device
from scipy.optimize import fmin_l_bfgs_b
from copy import copy

import argparse

from sim_utils import (
    SubsetNegPoissonLogLWithPrior,
    split_fwd_model,
    OSEM,
    StochasticGradientDescent,
    MLEMPreconditioner,
    HarmonicPreconditioner,
    subset_generator_with_replacement,
    subset_generator_without_replacement,
    subset_generator_herman_meyer,
    subset_generator_cofactor,
)

from rdp import RDP
from sim_phantoms import pet_phantom

# choose a device (CPU or CUDA GPU)
if "numpy" in xp.__name__:
    # using numpy, device must be cpu
    dev = "cpu"
elif "cupy" in xp.__name__:
    # using cupy, only cuda devices are possible
    dev = xp.cuda.Device(0)


def nrmse(vec_x, vec_y, mask, norm):
    return float(xp.sqrt(xp.mean((vec_x[mask] - vec_y[mask]) ** 2))) / norm


# %%

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--subset_seed", type=int, default=1)
parser.add_argument(
    "--subset_sampling_method", default="wor", choices=["wor", "wr", "hm", "cofactor"]
)
parser.add_argument("--true_counts", type=float, default=int(1e8))
parser.add_argument("--beta_rel", type=float, default=1.0)
parser.add_argument("--num_epochs", type=int, default=20)
parser.add_argument("--num_subsets", type=int, default=27)
parser.add_argument("--precond_type", type=int, default=2, choices=[1, 2])
parser.add_argument("--phantom_type", type=int, default=1, choices=[1, 2])
parser.add_argument("--tof", action="store_true")
parser.add_argument("--method", default="SVRG", choices=["SVRG", "SGD", "SAGA"])
parser.add_argument("--eta", type=float, default=0.0)
parser.add_argument("--init_step_size", type=float, default=1.0)
parser.add_argument("--precond_update_epochs", type=int, default=None, nargs="+")
parser.add_argument("--suffix", type=str, default="")
parser.add_argument("--gnbs", action="store_true")
parser.add_argument(
    "--step_size_rule", default="decaying", choices=["const", "decaying", "bb", "alg1"]
)

args = parser.parse_args()

# %%
# input parameters

seed = int(args.seed)
subset_seed = int(args.subset_seed)
subset_sampling_method = args.subset_sampling_method

# true counts, reasonable range
true_counts = int(args.true_counts)
# regularization weight, reasonable range: 5e-5 * (true_counts / 1e6) is medium regularization
beta_rel = args.beta_rel
beta = beta_rel * (2e-4) * (true_counts / 3e7)
# RDP gamma parameter
gamma_rdp = 2.0
# type of preconditioner: 1 MLEM, 2: Harmonic
precond_type = args.precond_type

# number of epochs / subsets for stochastic gradient algorithms
num_epochs = args.num_epochs
num_subsets = args.num_subsets

# max number of updates for reference L-BFGS-B solution
num_iter_bfgs_ref = 500

# %%
# number of rings of simulated PET scanner, should be odd in this example
num_rings = 17
# resolution of the simulated PET scanner in mm
fwhm_data_mm = 4.0
# simulated TOF or non-TOF system
tof = args.tof
# mean of contamination sinogram, relative to mean of trues sinogram, reasonable range: 0.5 - 1.0
contam_fraction = 0.5
# show the geometry of the scanner / image volume
show_geometry = False
# verbose output
verbose = True
# track cost function values after every update (slow)
track_cost = True

# number of epochs / subsets for intial OSEM
num_epochs_osem = 1
num_subsets_osem = 27

# phantom type (int)
phantom_type = args.phantom_type

# stochastic gradient method
method = args.method

# step size parameters
step_size_rule = args.step_size_rule
if step_size_rule == "const":
    eta = 0.0
    init_step_size = args.init_step_size
    step_size_func = lambda k: init_step_size
    barzilai_borwein = False
elif step_size_rule == "decaying":
    eta = args.eta
    init_step_size = args.init_step_size
    step_size_func = lambda k: init_step_size / (1 + eta * k / num_subsets)
    barzilai_borwein = False
elif step_size_rule == "bb":
    init_step_size = args.init_step_size
    eta = 0.0

    def step_size_func(update: int) -> float:
        if update <= 10:
            new_step_size = 3.0
        elif update > 10 and update <= (4 * 25):
            new_step_size = 2.0
        elif update > (4 * 25) and update <= (8 * 25):
            new_step_size = 1.5
        elif update > (8 * 25) and update <= (12 * 25):
            new_step_size = 1.0
        else:
            new_step_size = 0.5
        return new_step_size

    barzilai_borwein = True

elif step_size_rule == "alg1":
    init_step_size = args.init_step_size
    eta = 0.0

    def step_size_func(update: int) -> float:
        if update <= 10:
            new_step_size = 3.0
        elif update > 10 and update <= (4 * 25):
            new_step_size = 2.0
        elif update > (4 * 25) and update <= (8 * 25):
            new_step_size = 1.5
        elif update > (8 * 25) and update <= (12 * 25):
            new_step_size = 1.0
        else:
            new_step_size = 0.5
        return new_step_size

    barzilai_borwein = False
else:
    raise ValueError("invalid step size rule")

# eta = args.eta
# init_step_size = args.init_step_size
# step_size_func = lambda k: init_step_size / (1 + eta * k / num_subsets)

precond_update_epochs = args.precond_update_epochs
suffix = args.suffix

gradient_norm_based_sampling = args.gnbs

# %%
# random seed
np.random.seed(seed)
xp.random.seed(seed)

# %%
# Setup of the forward model :math:`\bar{y}(x) = A x + s`
# --------------------------------------------------------
#
# We setup a linear forward operator :math:`A` consisting of an
# image-based resolution model, a non-TOF PET projector and an attenuation model
#
# .. note::
#     The OSEM implementation below works with all linear operators that
#     subclass :class:`.LinearOperator` (e.g. the high-level projectors).

scanner = parallelproj.RegularPolygonPETScannerGeometry(
    xp,
    dev,
    radius=300.0,
    num_sides=36,
    num_lor_endpoints_per_side=12,
    lor_spacing=4.0,
    ring_positions=xp.linspace(
        -5 * (num_rings - 1) / 2, 5 * (num_rings - 1) / 2, num_rings
    ),
    symmetry_axis=2,
)

# %%
# setup the LOR descriptor that defines the sinogram

img_shape = (161, 161, 2 * num_rings - 1)
voxel_size = (2.5, 2.5, 2.5)

lor_desc = parallelproj.RegularPolygonPETLORDescriptor(
    scanner,
    radial_trim=40,
    sinogram_order=parallelproj.SinogramSpatialAxisOrder.RVP,
)

if lor_desc.num_views % num_subsets != 0:
    raise ValueError(
        f"num_subsets_sgd ({num_subsets}) must be a divisor of num_views ({lor_desc.num_views})"
    )
if lor_desc.num_views % num_subsets_osem != 0:
    raise ValueError(
        f"num_subsets_osem  ({num_subsets_osem}) must be a divisor of num_views ({lor_desc.num_views})"
    )

proj = parallelproj.RegularPolygonPETProjector(
    lor_desc, img_shape=img_shape, voxel_size=voxel_size
)

# %%
# setup of true emission and attenuation image
print("setting up phantom")

if phantom_type == 1:
    x_true, x_att = pet_phantom(img_shape, xp, dev, mu_value=0.01)
elif phantom_type == 2:
    x_true, x_att = pet_phantom(
        img_shape,
        xp,
        dev,
        mu_value=0.01,
        add_spheres=True,
        add_inner_cylinder=False,
        r0=0.25,
        r1=0.25,
    )
else:
    raise ValueError("invalid phantom type")

# %%
# calculate the attenuation sinogram
print("calculating attenuation sinogram")
att_sino = xp.exp(-proj(x_att))

# %%
# Complete PET forward model setup
# --------------------------------
#
# We combine an image-based resolution model,
# a non-TOF or TOF PET projector and an attenuation model
# into a single linear operator.

# enable TOF - comment if you want to run non-TOF
if tof is True:
    proj.tof_parameters = parallelproj.TOFParameters(
        num_tofbins=21, tofbin_width=25.0, sigma_tof=25.0
    )

# setup the attenuation multiplication operator which is different
# for TOF and non-TOF since the attenuation sinogram is always non-TOF
if proj.tof:
    att_op = parallelproj.TOFNonTOFElementwiseMultiplicationOperator(
        proj.out_shape, att_sino
    )
else:
    att_op = parallelproj.ElementwiseMultiplicationOperator(att_sino)

res_model = parallelproj.GaussianFilterOperator(
    proj.in_shape, sigma=fwhm_data_mm / (2.35 * proj.voxel_size)
)

# compose all 3 operators into a single linear operator
pet_lin_op = parallelproj.CompositeLinearOperator((att_op, proj, res_model))

# %%
# show_geometry = True
if show_geometry:
    fig = plt.figure(figsize=(8, 8), tight_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    proj.show_geometry(ax)
    lor_desc.show_views(ax, xp.asarray([0, lor_desc.num_views // 4]), xp.asarray([1]))
    fig.show()

# breakpoint()
# %%
# Simulation of projection data
# -----------------------------
#
# We setup an arbitrary ground truth :math:`x_{true}` and simulate
# noise-free and noisy data :math:`y` by adding Poisson noise.

# simulated noise-free data
noise_free_data = pet_lin_op(x_true)

if true_counts > 0:
    scale_fac = true_counts / float(xp.sum(noise_free_data))
    noise_free_data *= scale_fac
    x_true *= scale_fac

# generate a contant contamination sinogram
contamination = xp.full(
    noise_free_data.shape,
    contam_fraction * float(xp.mean(noise_free_data)),
    device=dev,
    dtype=xp.float32,
)

noise_free_data += contamination

# add Poisson noise
print("adding Poisson noise")
data = xp.random.poisson(noise_free_data)

# %%
# run quick OSEM with one iteration

print("running OSEM")
pet_subset_lin_op_seq_osem, subset_slices_osem = split_fwd_model(
    pet_lin_op, num_subsets_osem
)

data_fidelity = SubsetNegPoissonLogLWithPrior(
    data, pet_subset_lin_op_seq_osem, contamination, subset_slices_osem
)

x0 = xp.ones(pet_lin_op.in_shape, device=dev, dtype=xp.float32)
osem_alg = OSEM(data_fidelity)
x_osem = osem_alg.run(x0, num_epochs_osem)
x_init = x_osem.copy()

del pet_subset_lin_op_seq_osem

# %%
# setup of the cost function

print("setting up kappa")
fwd_ones = pet_lin_op(xp.ones(pet_lin_op.in_shape, device=dev, dtype=xp.float32))
fwd_osem = pet_lin_op(x_osem) + contamination
kappa_img = xp.sqrt(pet_lin_op.adjoint((data * fwd_ones) / (fwd_osem**2)))

del fwd_ones
del fwd_osem

print("setting up RDP")
prior = RDP(
    img_shape,
    xp=xp,
    dev=dev,
    voxel_size=xp.asarray(voxel_size, device=dev),
    eps=float(xp.max(x_osem)) * 1e-3,
    gamma=gamma_rdp,
)

prior.kappa = kappa_img
prior.scale = beta

# %%
pet_subset_lin_op_seq, subset_slices = split_fwd_model(pet_lin_op, num_subsets)

cost_function = SubsetNegPoissonLogLWithPrior(
    data, pet_subset_lin_op_seq, contamination, subset_slices, prior=prior
)

# calculate cost of quick OSEM reconstruction
cost_osem = cost_function(x_osem)

# %%
# setup the preconditioner for stochastic gradient algorithms
print("setting up preconditioner")

adjoint_ones = pet_lin_op.adjoint(
    xp.ones(pet_lin_op.out_shape, device=dev, dtype=xp.float32)
)

if precond_type == 1:
    diag_precond = MLEMPreconditioner(adjoint_ones)
elif precond_type == 2:
    # add a 4mm FWHM Gaussian filter before calc. the diag Hess of the prior
    precond_pre_filter_func = parallelproj.GaussianFilterOperator(
        img_shape, sigma=4.0 / (2.35 * xp.asarray(voxel_size))
    )

    diag_precond = HarmonicPreconditioner(
        adjoint_ones, prior=prior, factor=2.0, filter_function=precond_pre_filter_func
    )
elif precond_type == 3:
    # same as precond_type 2 but without pre-filter
    diag_precond = HarmonicPreconditioner(
        adjoint_ones, prior=prior, factor=2.0, filter_function=None
    )
else:
    raise ValueError("invalid preconditioner version")

# %%
# run (pre-conditioned) L-BFGS-B without subsets as reference

ref_path = Path("sim_results_ref_recons")
ref_path.mkdir(exist_ok=True)

ref_file = (
    ref_path
    / f"rdp_t_{true_counts:.1E}_b_{beta_rel:.2f}_g_{gamma_rdp:.2f}_n_{num_iter_bfgs_ref}_nr_{num_rings}_tof_{tof}_cf_{contam_fraction}_s_{seed}_ph_{phantom_type}.npy"
)

if ref_file.exists():
    print("loading L-BFGS-B reference")
    x_ref = xp.asarray(np.load(ref_file), device=dev)
else:
    print("running L-BFGS-B reference")
    # the preconditioner is taken from
    # Tsai et al: "Benefits of Using a Spatially-Variant Penalty Strength With Anatomical Priors in PET Reconstruction"
    # doi: 10.1109/TMI.2019.2913889

    # run a 1st L-BFGS-B to get a preconditioner based on the initial OSEM solution
    # calculate data term preconditioner
    precond_lbfgs0 = 1 / xp.sqrt(kappa_img**2 + prior.diag_hessian(x_init))
    precond_lbfgs_np0 = xp.asnumpy(precond_lbfgs0).ravel()

    def precond_cf0(z):
        return cost_function(z * precond_lbfgs_np0)

    def precond_grad0(z):
        return cost_function.gradient(z * precond_lbfgs_np0) * precond_lbfgs_np0

    z0_bfgs = to_device((x_init / precond_lbfgs0).ravel(), "cpu")
    bounds = x0.size * [(0, None)]

    res0 = fmin_l_bfgs_b(
        precond_cf0,
        z0_bfgs,
        precond_grad0,
        disp=1,
        maxiter=50,
        bounds=bounds,
        m=20,
        factr=1e-16,
        pgtol=1e-16,
    )

    x_ref0 = precond_lbfgs0 * xp.asarray(res0[0].reshape(img_shape), device=dev)

    # run 2nd L-BFGS-B with updated preconditioner
    # calculate data term preconditioner
    precond_lbfgs1 = 1 / xp.sqrt(kappa_img**2 + prior.diag_hessian(x_ref0))
    precond_lbfgs_np1 = xp.asnumpy(precond_lbfgs1).ravel()

    def precond_cf1(z):
        return cost_function(z * precond_lbfgs_np1)

    def precond_grad1(z):
        return cost_function.gradient(z * precond_lbfgs_np1) * precond_lbfgs_np1

    z1_bfgs = to_device((x_ref0 / precond_lbfgs1).ravel(), "cpu")

    res1 = fmin_l_bfgs_b(
        precond_cf1,
        z1_bfgs,
        precond_grad1,
        disp=1,
        maxiter=num_iter_bfgs_ref,
        bounds=bounds,
        m=20,
        factr=1e-16,
        pgtol=1e-16,
    )

    x_ref = precond_lbfgs1 * xp.asarray(res1[0].reshape(img_shape), device=dev)
    xp.save(ref_file, x_ref)


# %%
cost_ref = cost_function(x_ref)
x_osem_scale = float(xp.mean(x_init))

print(cost_ref)


# %%
# define NRMSE callback
def nrmse_callback(x):
    """calculate NRMSE and time since start"""
    return nrmse(x, x_ref, x_true > 0, norm=scale_fac)


# %%
nrmse_osem = nrmse_callback(x_osem)
nrmse_init = nrmse_callback(x_init)

# %%
# run stochastic gradient descent
# setup subset negative Poisson log-likelihood without prior
subset_neglogL = SubsetNegPoissonLogLWithPrior(
    data, pet_subset_lin_op_seq, contamination, subset_slices, prior=None
)

# setup subset sampling generator
if subset_sampling_method == "wor":
    subset_generator = subset_generator_without_replacement(num_subsets, subset_seed)
elif subset_sampling_method == "wr":
    subset_generator = subset_generator_with_replacement(num_subsets, subset_seed)
elif subset_sampling_method == "hm":
    subset_generator = subset_generator_herman_meyer(num_subsets)
elif subset_sampling_method == "cofactor":
    subset_generator = subset_generator_cofactor(num_subsets)
else:
    raise ValueError("invalid subset sampling method")


print(f"running stochastic gradient descent {method}")
stochastic_alg = StochasticGradientDescent(
    subset_neglogL,
    prior,
    x_init,
    diag_precond_func=diag_precond,
    subset_generator=subset_generator,
    method=method,
    verbose=False,
    step_size_func=step_size_func,
    precond_update_epochs=precond_update_epochs,
    gradient_norm_based_sampling=gradient_norm_based_sampling,
    barzilai_borwein=barzilai_borwein,
)  # add barzilai_borwein=barzilai_borwein

callback_res_stochastic = stochastic_alg.run(
    num_epochs * num_subsets, callback=nrmse_callback
)

nrmse_stochastic = [x[0] for x in callback_res_stochastic]
walltime_stochastic = [x[1] for x in callback_res_stochastic]

# %%
cost_stochastic = cost_function(stochastic_alg.x)

print(f"cost ref                                  .: {cost_ref:.8E}")
print(f"cost stochastic                           .: {cost_stochastic:.8E}")
print(
    f"(cost_stochastic-cost_ref)/abs(cost_ref)  .: {(cost_stochastic-cost_ref)/abs(cost_ref)}"
)

if cost_stochastic < cost_ref:
    print(
        "WARNING: stochastic cost is lower than reference cost. Find better reference."
    )

# %%
# save the results
res_dict = vars(copy(args))

res_dict["scale_fac"] = scale_fac
res_dict["beta"] = beta
res_dict["cost_ref"] = cost_ref
res_dict["cost_osem"] = cost_osem
res_dict["cost_stochastic"] = cost_stochastic
res_dict["nrmse_stochastic"] = nrmse_stochastic
res_dict["walltime_stochastic"] = walltime_stochastic
res_dict["nrmse_osem"] = nrmse_osem
res_dict["nrmse_init"] = nrmse_init

recon_path = Path("sim_results_ablation")
recon_path.mkdir(exist_ok=True)

print(f"ssl{step_size_rule}, eta{eta:.2E}, s0{init_step_size:.2E}")
res_file = (
    recon_path
    / f"{ref_file.stem}_ne_{num_epochs}_ns_{num_subsets}_m_{method}_pc_{precond_type}_s0_{init_step_size:.2E}_eta_{eta:.2E}_ss_{subset_seed}_ssm_{subset_sampling_method}_ssl_{step_size_rule}_gnbs{gradient_norm_based_sampling}_{suffix}.json"
)

with open(res_file, "w", encoding="utf-8") as f:
    json.dump(res_dict, f)

recon_file = res_file.with_suffix(".npy")
xp.save(recon_file, stochastic_alg.x)

# %%
fig, ax = plt.subplots(
    5,
    4,
    figsize=(12, 9),
    tight_layout=True,
    gridspec_kw={"height_ratios": [4, 1, 1, 0.07, 3]},
)

sl0 = x_true.shape[1] // 2
sl1 = x_true.shape[2] // 2
sl2 = x_true.shape[0] // 2

ims = dict(cmap="Greys", vmin=0, vmax=1.2 * float(xp.max(x_true)))
ims_diff = dict(cmap="seismic", vmin=-0.05, vmax=0.05)

ax[0, 0].imshow(to_device(x_true[..., sl1], "cpu"), **ims)
ax[1, 0].imshow(to_device(x_true[sl2, ...].T, "cpu"), **ims)
im0 = ax[2, 0].imshow(to_device(x_true[:, sl0, :].T, "cpu"), **ims)
ax[0, 0].set_title(f"ground truth", fontsize="medium")

ax[0, 1].imshow(to_device(x_ref[..., sl1], "cpu"), **ims)
ax[1, 1].imshow(to_device(x_ref[sl2, ...].T, "cpu"), **ims)
im1 = ax[2, 1].imshow(to_device(x_ref[:, sl0, :].T, "cpu"), **ims)
ax[0, 1].set_title(f"reference (L-BFGS-B)", fontsize="medium")

ax[0, 2].imshow(to_device(stochastic_alg.x[..., sl1], "cpu"), **ims)
ax[1, 2].imshow(to_device(stochastic_alg.x[sl2, ...].T, "cpu"), **ims)
im2 = ax[2, 2].imshow(to_device(stochastic_alg.x[:, sl0, :].T, "cpu"), **ims)
ax[0, 2].set_title(
    f"{method} {num_epochs} epochs, {num_subsets} subsets", fontsize="medium"
)

ax[0, 3].imshow(
    to_device((stochastic_alg.x[..., sl1] - x_ref[..., sl1]) / scale_fac, "cpu"),
    **ims_diff,
)
ax[1, 3].imshow(
    to_device((stochastic_alg.x[sl2, ...] - x_ref[sl2, ...]).T / scale_fac, "cpu"),
    **ims_diff,
)
im3 = ax[2, 3].imshow(
    to_device((stochastic_alg.x[:, sl0, :].T - x_ref[:, sl0, :].T) / scale_fac, "cpu"),
    **ims_diff,
)
ax[0, 3].set_title(f"({method} - ref.) / bg. activity", fontsize="medium")

fig.colorbar(im0, ax=ax[-2, 0], location="bottom", fraction=1, aspect=70)
fig.colorbar(im1, ax=ax[-2, 1], location="bottom", fraction=1, aspect=70)
fig.colorbar(im2, ax=ax[-2, 2], location="bottom", fraction=1, aspect=70)
fig.colorbar(im3, ax=ax[-2, 3], location="bottom", fraction=1, aspect=70)

for axx in ax[-2, :]:
    axx.set_axis_off()

update_arr = np.arange(num_subsets * num_epochs)

ax[-1, 0].semilogy(update_arr / num_subsets, nrmse_stochastic, label=method)
ax[-1, 0].axhline(0.01, color="black", ls="--")
ax[-1, 0].set_title("NRMSE", fontsize="medium")
ax[-1, 0].set_xlabel("epoch")
ax[-1, 0].grid(ls=":")
ax[-1, 0].legend()

ax[-1, 1].plot(
    update_arr / num_subsets, [stochastic_alg.step_size_func(x) for x in update_arr]
)
ax[-1, 1].set_title("step size", fontsize="medium")
ax[-1, 1].set_xlabel("epoch")
ax[-1, 1].grid(ls=":")

# add simulation parameters as text
params = {key: value for key, value in vars(args).items() if key != "step_size_func"}
params_text = "\n".join([f"{key}: {value}" for key, value in params.items()])
params_text += (
    f"\n(cost_stoch-cost_ref)/cost_ref: {(cost_stochastic-cost_ref)/abs(cost_ref):.2E}"
)

color = "red" if cost_stochastic < cost_ref else "black"
ax[-1, -1].text(
    0.5,
    1.0,
    params_text,
    ha="center",
    va="top",
    transform=ax[-1, -1].transAxes,
    fontsize="xx-small",
    color=color,
)


ax[-1, 2].set_axis_off()
ax[-1, 3].set_axis_off()

fig_file = res_file.with_suffix(".png")

fig.savefig(fig_file, dpi=300)
# fig.show()
