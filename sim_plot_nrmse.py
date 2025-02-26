import json
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

methods = ["SGD", "SAGA", "SVRG"]
true_counts_list = [1e7, 1e8]
beta_rels = [1.0, 4.0, 16.0]
precond_types = [1, 2]

sim_path = Path("sim_results")
sim_path.mkdir(exist_ok=True)

gamma_rdp = 2
num_iter_bfgs_ref = 500
num_rings = 17
tof = True
contam_fraction = 0.5
seed = 1
phantom_type = 2

num_epochs = 20
num_subsets = 27
step_size_func_str = "lambda_x__1.0"

nrows = len(true_counts_list)
ncols = len(beta_rels)

fig, ax = plt.subplots(
    nrows,
    ncols,
    figsize=(3 * ncols, 3 * nrows),
    tight_layout=True,
    sharex=True,
    sharey=True,
)

fig2, ax2 = plt.subplots(
    nrows,
    ncols,
    figsize=(3 * ncols, 3 * nrows),
    tight_layout=True,
    sharex=True,
    sharey=True,
)

for axx in ax.ravel():
    axx.axhline(1e-2, color="k")

for i, true_counts in enumerate(true_counts_list):
    for j, beta_rel in enumerate(beta_rels):
        beta = beta_rel * (2e-4) * (true_counts / 3e7)

        ref_file = (
            sim_path
            / f"rdp_t_{true_counts:.2E}_b_{beta:.2E}_g_{gamma_rdp:.2E}_n_{num_iter_bfgs_ref}_nr_{num_rings}_tof_{tof}_cf_{contam_fraction}_s_{seed}_ph_{phantom_type}.npy"
        )

        x_ref = np.load(ref_file)
        sl = x_ref.shape[2] // 2

        ax2[i, j].imshow(
            x_ref[..., sl], vmin=0, vmax=0.18 * true_counts / 1e7, cmap="Greys"
        )

        for im, method in enumerate(methods):
            for ip, precond_type in enumerate(precond_types):

                res_file = (
                    ref_file.parent
                    / f"{ref_file.stem}_ne_{num_epochs}_ns_{num_subsets}_m_{method}_pc_{precond_type}__pf_4.0_{step_size_func_str}.json"
                )

                # read nrmse_stochastic from the JSON file
                with open(res_file, "r") as f:
                    content = json.load(f)
                    nrmse_stochastic = content["nrmse_stochastic"]

                # plot the nrmse_stochastic values
                if ip == 0:
                    ls = ":"
                else:
                    ls = "-"
                ax[i, j].semilogy(
                    np.arange(num_epochs * num_subsets) / num_subsets,
                    nrmse_stochastic,
                    color=plt.cm.tab10(im),
                    linestyle=ls,
                    label=f"{method} pc={precond_type}",
                )

for axx in ax.ravel():
    axx.grid(True, ls=":")

for axx in ax[-1, :]:
    axx.set_xlabel("epoch")

for i, axx in enumerate(ax[0, :]):
    axx.set_title(f"beta = {beta_rels[i]:.1f}", fontsize="medium")
for i, axx in enumerate(ax2[0, :]):
    axx.set_title(f"beta = {beta_rels[i]:.1f}", fontsize="medium")

for i, axx in enumerate(ax[:, 0]):
    axx.set_ylabel(f"NRMSE true counts: {true_counts_list[i]:.1E}")
for i, axx in enumerate(ax2[:, 0]):
    axx.set_ylabel(f"true counts: {true_counts_list[i]:.1E}")

ax[0, 0].legend(fontsize="x-small", ncol=2, loc="lower right")

fig.show()
fig2.show()
