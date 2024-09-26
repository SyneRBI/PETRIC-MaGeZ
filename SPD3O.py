"""script to compare SPDHG, SGD and SVRG for PET reconstruction with quad. and RDP prior"""

# %%
from __future__ import annotations

try:
    import array_api_compat.cupy as xp
except ImportError:
    import array_api_compat.numpy as xp

import parallelproj
import array_api_compat.numpy as np
import matplotlib.pyplot as plt

from array_api_compat import to_device
from scipy.optimize import fmin_l_bfgs_b
from pathlib import Path

from utils import (
    SubsetNegPoissonLogLWithPrior,
    split_fwd_model,
    OSEM,
    SGD,
    SVRG,
    rdp_preconditioner,
)

import sys

sys.path.append("../")
from rdp import RDP, neighbor_product

# choose a device (CPU or CUDA GPU)
if "numpy" in xp.__name__:
    # using numpy, device must be cpu
    dev = "cpu"
elif "cupy" in xp.__name__:
    # using cupy, only cuda devices are possible
    dev = xp.cuda.Device(0)

# %%
# set up

# input parameters
seed = 1

# true counts, reasonable range: 1e6, 1e7 (high counts), 1e5 (low counts)
true_counts = 1e6
# regularization weight, reasonable range: 5e-5 * (true_counts / 1e6) is medium regularization
beta = 5e-3 * (true_counts / 1e6)
# RDP gamma parameter
gamma_rdp = 2.0

# number of epochs / subsets for stochastic gradient algorithms
num_epochs = 2
num_subsets = 108

# max number of updates for reference L-BFGS-B solution
num_iter_bfgs_ref = 400

# number of rings of simulated PET scanner, should be odd in this example
num_rings = 1
# resolution of the simulated PET scanner in mm
fwhm_data_mm = 4.5
# simulated TOF or non-TOF system
tof = False
# mean of contamination sinogram, relative to mean of trues sinogram, reasonable range: 0.5 - 1.0
contam_fraction = 0.5
# verbose output
verbose = False
# track cost function values after every update (slow)
track_cost = False

# number of epochs / subsets for intial OSEM
num_epochs_osem = 1
num_subsets_osem = 27

nrmse_limit = 5e-3

# random seed
np.random.seed(seed)

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

# setup the LOR descriptor that defines the sinogram

img_shape = (100, 100, 2 * num_rings - 1)
voxel_size = (2.0, 2.0, 2.0)

lor_desc = parallelproj.RegularPolygonPETLORDescriptor(
    scanner,
    radial_trim=140,
    sinogram_order=parallelproj.SinogramSpatialAxisOrder.RVP,
)

if lor_desc.num_views % num_subsets != 0:
    raise ValueError(
        f"num_subsets ({num_subsets}) must be a divisor of num_views ({lor_desc.num_views})"
    )
if lor_desc.num_views % num_subsets_osem != 0:
    raise ValueError(
        f"num_subsets_osem  ({num_subsets_osem}) must be a divisor of num_views ({lor_desc.num_views})"
    )

proj = parallelproj.RegularPolygonPETProjector(
    lor_desc, img_shape=img_shape, voxel_size=voxel_size
)

# setup a simple test image containing a few "hot rods"
x_true = xp.ones(proj.in_shape, device=dev, dtype=xp.float32)
c0 = proj.in_shape[0] // 2
c1 = proj.in_shape[1] // 2
x_true[(c0 - 4) : (c0 + 4), (c1 - 4) : (c1 + 4), :] = 3.0

x_true[28:32, c1 : (c1 + 4), :] = 5.0
x_true[c0 : (c0 + 4), 20:24, :] = 5.0

x_true[-32:-28, c1 : (c1 + 4), :] = 0.1
x_true[c0 : (c0 + 4), -24:-20, :] = 0.1

x_true[:25, :, :] = 0
x_true[-25:, :, :] = 0
x_true[:, :10, :] = 0
x_true[:, -10:, :] = 0

# Attenuation image and sinogram setup
# ------------------------------------

# setup an attenuation image
x_att = 0.01 * xp.astype(x_true > 0, xp.float32)
# calculate the attenuation sinogram
att_sino = xp.exp(-proj(x_att))

# Complete PET forward model setup
# --------------------------------
#
# We combine an image-based resolution model,
# a non-TOF or TOF PET projector and an attenuation model
# into a single linear operator.

# enable TOF - comment if you want to run non-TOF
if tof is True:
    proj.tof_parameters = parallelproj.TOFParameters(
        num_tofbins=13, tofbin_width=12.0, sigma_tof=12.0
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
data = xp.asarray(
    np.random.poisson(parallelproj.to_numpy_array(noise_free_data)),
    device=dev,
    dtype=xp.float32,
)

# run quick OSEM with one iteration
pet_subset_lin_op_seq_osem, subset_slices_osem = split_fwd_model(
    pet_lin_op, num_subsets_osem
)

data_fidelity = SubsetNegPoissonLogLWithPrior(
    data, pet_subset_lin_op_seq_osem, contamination, subset_slices_osem
)

x0 = xp.ones(pet_lin_op.in_shape, device=dev, dtype=xp.float32)
osem_alg = OSEM(data_fidelity)
x_osem = osem_alg.run(x0, num_epochs_osem)

#### post filter osem to get better initial recon
osem_filter = parallelproj.GaussianFilterOperator(
    proj.in_shape, sigma=4.7 / (2.35 * proj.voxel_size)
)
x_init = osem_filter(x_osem)

# setup of the cost function

fwd_ones = pet_lin_op(xp.ones(pet_lin_op.in_shape, device=dev, dtype=xp.float32))
fwd_osem = pet_lin_op(x_osem) + contamination
kappa_img = xp.sqrt(pet_lin_op.adjoint((data * fwd_ones) / (fwd_osem**2)))

prior = RDP(
    img_shape,
    xp=xp,
    dev=dev,
    voxel_size=xp.asarray(voxel_size, device=dev),
    eps=float(xp.max(x_osem)) / 100,
    gamma=gamma_rdp,
)

prior.kappa = kappa_img
prior.scale = beta

adjoint_ones = pet_lin_op.adjoint(
    xp.ones(pet_lin_op.out_shape, device=dev, dtype=xp.float32)
)

pet_subset_lin_op_seq, subset_slices = split_fwd_model(pet_lin_op, num_subsets)

cost_function = SubsetNegPoissonLogLWithPrior(
    data, pet_subset_lin_op_seq, contamination, subset_slices, prior=prior
)

# run L-BFGS-B without subsets as reference
x0_bfgs = to_device(x_init.ravel(), "cpu")

bounds = x0.size * [(0, None)]

ref_file = Path(
    f"rdp_t_{true_counts:.2E}_b_{beta:.2E}_g_{gamma_rdp:.2E}_n_{num_iter_bfgs_ref}_nr_{num_rings}_tof_{tof}_cf_{contam_fraction}_s_{seed}.npy"
)

if ref_file.exists():
    print("ref solution exists: load it")
    x_ref = xp.asarray(np.load(ref_file), device=dev)
else:
    print("ref solution does not exist: compute it")
    res = fmin_l_bfgs_b(
        cost_function,
        x0_bfgs,
        cost_function.gradient,
        disp=True,
        maxiter=num_iter_bfgs_ref,
        bounds=bounds,
        m=10,
        factr=10.0,
    )

    x_ref = xp.asarray(res[0].reshape(img_shape), device=dev)
    xp.save(ref_file, x_ref)

cost_ref = cost_function(x_ref)

x_osem_scale = float(xp.mean(x_init))

cost_osem = cost_function(x_osem)
nrmse_osem = xp.sqrt(xp.mean((x_ref - x_osem) ** 2)) / scale_fac
nrmse_init = xp.sqrt(xp.mean((x_ref - x_init) ** 2)) / scale_fac

# %%
# SPD3O

# Test a rough value for L
x1 = x_init
x1 = np.random.rand(*x1.shape)
x2 = 1.1 * x1
x2 = np.random.rand(*x1.shape)
g1 = prior.gradient(x1)
g2 = prior.gradient(x2)

Lest = np.linalg.norm(g1 - g2) / np.linalg.norm(x1 - x2)
print(Lest)

L = 60

# %%
# SPD3O
# parameters
rhos = np.array([1.5])  # up to rho = 3 seems also to work for some gammas
# array of gamma values to try for SPDHG - these get divided by the "scale" of the OSEM image
gammas = np.array([0.8 / 5])

num_epochs = 20

run_spd3o = True


params = gammas

if run_spd3o:
    print("run spd3o")

    nrmse_spd3o = np.zeros((len(params), num_epochs * num_subsets), dtype=xp.float32)

    # list for all recons using different gamma values
    x_spd3os = []

    for ig, param in enumerate(params):

        rho = param
        gamma = 1 / x_osem_scale
        print("run gamma", gamma)

        # initialize primal and dual variables
        x_spd3o = x_init.copy()
        # initialize dual variable for the negative Poisson logL
        y = 1 - data / (pet_lin_op(x_spd3o) + contamination)

        # y = 0*data

        # initialize z and zbar
        z = pet_lin_op.adjoint(y)
        # z = 0 * x_init
        zbar = 1.0 * z

        # calculate SPHDG step sizes
        S_As = []
        T_As = []

        for lin_op in pet_subset_lin_op_seq:
            tmp = lin_op(xp.ones(lin_op.in_shape, dtype=xp.float32, device=dev))
            # replace zeros by smallest non-zero value, to avoid division by zero
            tmp = xp.where(tmp == 0, xp.min(tmp[tmp > 0]), tmp)
            S_As.append(gamma * rho / tmp)

            T_As.append(
                (1 / (gamma * num_subsets))
                * rho
                / lin_op.adjoint(
                    xp.ones(lin_op.out_shape, dtype=xp.float64, device=dev)
                )
            )

        # element-wise minimum of the data T's
        T = xp.min(xp.asarray(T_As), axis=0)

        print(T.max() * L / 2)

        num_updates = num_epochs * num_subsets

        for i in range(num_updates):
            subset = np.random.randint(num_subsets)
            sl = subset_slices[subset]

            if i == 0:
                grad_h = prior.gradient(x_spd3o)

            q = zbar + grad_h
            x_spd3o = xp.clip(x_spd3o - T * q, 0, None)

            grad_h_new = prior.gradient(x_spd3o)

            xbar = x_spd3o + T * (grad_h - grad_h_new)
            # xbar = x_spd3o

            grad_h = grad_h_new

            # forward step
            y_plus = y[sl] + S_As[subset] * (
                pet_subset_lin_op_seq[subset](xbar) + contamination[sl]
            )
            # prox of convex conjugate of negative Poisson logL
            y_plus = 0.5 * (
                y_plus + 1 - xp.sqrt((y_plus - 1) ** 2 + 4 * S_As[subset] * data[sl])
            )
            delta_z = pet_subset_lin_op_seq[subset].adjoint(y_plus - y[sl])
            y[sl] = y_plus

            z = z + delta_z
            zbar = z + num_subsets * delta_z

            nrmse_spd3o[ig, i] = xp.sqrt(xp.mean((x_ref - x_spd3o) ** 2)) / scale_fac

            if (i + 1) % num_subsets == 0:
                print(
                    f"SPD3O epoch {((i+1)//num_subsets):04} / {num_epochs} NRMSE: {nrmse_spd3o[ig, i]:.2E}",
                    end="\r",
                )

        x_spd3os.append(x_spd3o)

    # %%
    # SPD3O plots

    vmax = 1.2 * float(xp.max(x_true))
    sl = img_shape[2] // 2
    num_rows = 3
    num_cols = len(params) + 1

    fig, ax = plt.subplots(
        num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3), tight_layout=True
    )
    ax[0, -1].imshow(
        to_device(x_ref[:, :, img_shape[2] // 2], "cpu"),
        cmap="Greys",
        vmin=0,
        vmax=1.2 * float(xp.max(x_true)),
    )
    ax[0, -1].set_title(f"ref. (L-BFGS-B)", fontsize="medium")
    ax[2, -1].set_axis_off()

    for ig, param in enumerate(params):
        ax[0, ig].imshow(
            to_device(x_spd3os[ig][:, :, img_shape[2] // 2], "cpu"),
            cmap="Greys",
            vmin=0,
            vmax=1.2 * float(xp.max(x_true)),
        )
        ax[1, ig].imshow(
            to_device(
                (x_spd3os[ig][:, :, sl] - x_ref[:, :, sl]) / (x_ref[:, :, sl] + 1e-3),
                "cpu",
            ),
            cmap="seismic",
            vmin=-0.2,
            vmax=0.2,
        )
        ax[2, ig].imshow(
            to_device(
                ((x_spd3os[ig][:, :, sl] - x_ref[:, :, sl]) / (x_ref[:, :, sl] + 1e-3))
                > 0.01,
                "cpu",
            ),
            cmap="Greys",
        )
        ax[0, ig].set_title(f"SPDHG, param {param}, {num_subsets}ss", fontsize="small")
        ax[1, ig].set_title(f"rel. bias", fontsize="small")
        ax[2, ig].set_title(f"rel. bias > 1%", fontsize="small")

        ax[1, -1].plot(
            np.arange(num_updates) / num_subsets,
            nrmse_spd3o[ig],
            label=f"param {param}",
        )

    ax[1, -1].set_title(f"NRMSE", fontsize="medium")
    ax[1, -1].set_xlabel(f"epoch")
    ax[1, -1].axhline(nrmse_limit, color="black", ls="--")
    ax[1, -1].legend()
    ax[1, -1].set_ylim(0, nrmse_init)
    ax[1, -1].grid(ls=":")

    fig.suptitle(
        f"True counts {true_counts:.2E}, prior RDP, beta {beta:.2E}, seed {seed}"
    )
    fig.savefig("fig_spdhg.png")
    fig.show()


# %%
# Run stochastic gradient descent

num_epochs_sgd = 5
run_sgd = True
decrease_step_size = False
step_sizes = np.array([1e-1])

# epoch numbers (starting at 0) where preconditioner is updated (use [] for no updates)
precond_update_epochs = [
    1,
    2,
]
# type of preconditioner (1: "P_logL": x / adjoint_ones, 2: 1 / (1/P_logL + 1/diag_Hess_RDP)
precond_version = 2
step_size_decay_factor = 0.75


num_updates_sgd = num_epochs_sgd * num_subsets

init_precond = rdp_preconditioner(
    x_init,
    adjoint_ones,
    prior,
    precond_version,
)


if run_sgd:
    print(f"cost init: {cost_osem}")
    print(f"cost ref: {cost_ref}")
    print(f"nrmse init: {nrmse_init}")
    print()

    cost_sgd = np.zeros((len(step_sizes), num_updates_sgd))
    nrmse_sgd = np.zeros((len(step_sizes), num_updates_sgd))

    x_sgds = []

    # SGD
    for i, step_size in enumerate(step_sizes):
        print(f"SGD  {i}, init step size: {step_size}")
        sgd_alg = SGD(cost_function, x_init)
        sgd_alg.step_size = step_size
        sgd_alg.precond = init_precond

        x_cur = x_init.copy()

        for j in range(num_updates_sgd):
            epoch = j // num_subsets
            subset = j % num_subsets

            if (epoch in precond_update_epochs) and (subset == 0):
                print("  updating preconditioner")
                sgd_alg.precond = rdp_preconditioner(
                    x_cur,
                    adjoint_ones,
                    prior,
                    precond_version,
                )

            x_cur = sgd_alg.update(x_cur)

            if track_cost:
                cost_sgd[i, j] = cost_function(x_cur)
            nrmse_sgd[i, j] = xp.sqrt(xp.mean((x_ref - x_cur) ** 2)) / scale_fac

            if decrease_step_size and (subset == 0) and (j > 0):
                sgd_alg.step_size *= step_size_decay_factor
                print(f"  decreasing step size {sgd_alg.step_size}")

        x_sgds.append(x_cur)

    # %%
    # SGD plots
    vmax = 1.2 * float(xp.max(x_true))
    sl = img_shape[2] // 2
    num_rows = 3
    num_cols = len(step_sizes) + 1

    fig, ax = plt.subplots(
        num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3), tight_layout=True
    )
    ax[0, -1].imshow(
        to_device(x_ref[:, :, img_shape[2] // 2], "cpu"),
        cmap="Greys",
        vmin=0,
        vmax=1.2 * float(xp.max(x_true)),
    )
    ax[0, -1].set_title(f"ref. (L-BFGS-B)", fontsize="medium")
    ax[2, -1].set_axis_off()

    for ig, step_size in enumerate(step_sizes):
        ax[0, ig].imshow(
            to_device(x_sgds[ig][:, :, img_shape[2] // 2], "cpu"),
            cmap="Greys",
            vmin=0,
            vmax=1.2 * float(xp.max(x_true)),
        )
        ax[1, ig].imshow(
            to_device(
                (x_sgds[ig][:, :, sl] - x_ref[:, :, sl]) / (x_ref[:, :, sl] + 1e-3),
                "cpu",
            ),
            cmap="seismic",
            vmin=-0.2,
            vmax=0.2,
        )
        ax[2, ig].imshow(
            to_device(
                ((x_sgds[ig][:, :, sl] - x_ref[:, :, sl]) / (x_ref[:, :, sl] + 1e-3))
                > 0.01,
                "cpu",
            ),
            cmap="Greys",
        )
        ax[0, ig].set_title(
            f"SGD, step size {step_size}, {num_subsets}ss", fontsize="small"
        )
        ax[1, ig].set_title(f"rel. bias", fontsize="small")
        ax[2, ig].set_title(f"rel. bias > 1%", fontsize="small")

        ax[1, -1].plot(
            np.arange(num_updates_sgd) / num_subsets,
            nrmse_sgd[ig],
            label=f"step size {step_size}",
        )

    ax[1, -1].set_title(f"NRMSE", fontsize="medium")
    ax[1, -1].set_xlabel(f"epoch")
    ax[1, -1].axhline(nrmse_limit, color="black", ls="--")
    ax[1, -1].legend()
    ax[1, -1].set_ylim(0, nrmse_init)
    ax[1, -1].grid(ls=":")
    fig.suptitle(
        f"True counts {true_counts:.2E}, prior RDP, beta {beta:.2E}, seed {seed}"
    )
    fig.savefig("fig_sgd.png")
    fig.show()


# %%
run_svrg = True
# update period for SVRG = epochs when all gradients are recalculated
svrg_gradient_recalc_periods = [x for x in range(0, num_epochs_sgd, 2)]

step_sizes = np.array([3e-1])


if run_svrg:
    x_svrgs = []

    cost_svrg = np.zeros((len(step_sizes), num_updates_sgd))
    nrmse_svrg = np.zeros((len(step_sizes), num_updates_sgd))

    # SVRG
    for i, step_size in enumerate(step_sizes):
        print(f"SVRG {i}, init step size: {step_size}")
        svrg_alg = SVRG(cost_function, x_init)
        svrg_alg.step_size = step_size
        svrg_alg.precond = init_precond

        x_cur = x_init.copy()

        for j in range(num_updates_sgd):
            epoch = j // num_subsets
            subset = j % num_subsets

            if subset == 0:
                print(f"  epoch {epoch}")

            line_search = False
            if epoch == 0 and subset < 4 and step_size == 0:
                line_search = True

            if (epoch in precond_update_epochs) and (subset == 0):
                print(f"  update {j}, updating preconditioner")
                svrg_alg.precond = rdp_preconditioner(
                    x_cur,
                    adjoint_ones,
                    prior,
                    precond_version,
                )

            if (epoch in svrg_gradient_recalc_periods) and (subset == 0):
                x_cur = svrg_alg.update(
                    x_cur, recalc_subset_gradients=True, line_search=line_search
                )
            else:
                x_cur = svrg_alg.update(
                    x_cur, recalc_subset_gradients=False, line_search=line_search
                )

            if track_cost:
                cost_svrg[i, j] = cost_function(x_cur)
            nrmse_svrg[i, j] = xp.sqrt(xp.mean((x_ref - x_cur) ** 2)) / scale_fac

            if decrease_step_size and (subset == 0) and (j > 0):
                svrg_alg.step_size *= step_size_decay_factor
                print(f"  update {j}, decreasing step size {svrg_alg.step_size}")

        x_svrgs.append(x_cur)

    # %%
    # SVRG plots

    vmax = 1.2 * float(xp.max(x_true))

    num_rows = 3
    num_cols = len(step_sizes) + 1

    sl = img_shape[2] // 2

    fig, ax = plt.subplots(
        num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3), tight_layout=True
    )
    ax[0, -1].imshow(
        to_device(x_ref[:, :, img_shape[2] // 2], "cpu"),
        cmap="Greys",
        vmin=0,
        vmax=1.2 * float(xp.max(x_true)),
    )
    ax[0, -1].set_title(f"ref. (L-BFGS-B)", fontsize="medium")
    ax[2, -1].set_axis_off()

    for ig, step_size in enumerate(step_sizes):
        ax[0, ig].imshow(
            to_device(x_svrgs[ig][:, :, img_shape[2] // 2], "cpu"),
            cmap="Greys",
            vmin=0,
            vmax=1.2 * float(xp.max(x_true)),
        )
        ax[1, ig].imshow(
            to_device(
                (x_svrgs[ig][:, :, sl] - x_ref[:, :, sl]) / (x_ref[:, :, sl] + 1e-3),
                "cpu",
            ),
            cmap="seismic",
            vmin=-0.2,
            vmax=0.2,
        )
        ax[2, ig].imshow(
            to_device(
                ((x_svrgs[ig][:, :, sl] - x_ref[:, :, sl]) / (x_ref[:, :, sl] + 1e-3))
                > 0.01,
                "cpu",
            ),
            cmap="Greys",
        )
        ax[0, ig].set_title(
            f"SVRG, step size {step_size}, {num_subsets}ss", fontsize="small"
        )
        ax[1, ig].set_title(f"rel. bias", fontsize="small")
        ax[2, ig].set_title(f"rel. bias > 1%", fontsize="small")

        ax[1, -1].plot(
            np.arange(num_updates_sgd) / num_subsets,
            nrmse_svrg[ig],
            label=f"step size {step_size}",
        )

    ax[1, -1].set_title(f"NRMSE", fontsize="medium")
    ax[1, -1].set_xlabel(f"epoch")
    ax[1, -1].axhline(nrmse_limit, color="black", ls="--")
    ax[1, -1].legend()
    ax[1, -1].set_ylim(0, float(nrmse_init))
    ax[1, -1].grid(ls=":")
    fig.suptitle(
        f"True counts {true_counts:.2E}, prior RDP, beta {beta:.2E}, seed {seed}"
    )
    fig.savefig(f"fig_svrg_{sl}.png")
    fig.show()

    # %%
    # compare
    t = np.arange(num_updates_sgd) / num_subsets
    plt.plot(t, nrmse_sgd[0], label=f"SGD")
    plt.plot(t, nrmse_spd3o[0], label=f"SPD3O")
    plt.plot(2 * t, nrmse_svrg[0], label=f"SVRG")
    plt.xlabel(f"epoch")
    plt.legend()
    plt.ylim(0, float(nrmse_init))
    plt.grid(ls=":")

# %%
