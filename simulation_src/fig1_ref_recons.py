import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from typing import List


true_counts_list: List[float] = [1e7, 1e8]
beta_rels: List[float] = [16.0, 4.0, 1.0]
gamma_rdp: float = 2
num_iter_bfgs_ref: int = 500
num_rings: int = 17
tof: bool = True
contam_fraction: float = 0.5
seed: int = 1
phantom_type: int = 1

crop: int = 30

# %%

sim_path = Path("sim_results")

nrows = len(true_counts_list)
ncols = len(beta_rels)

fig2, ax2 = plt.subplots(
    nrows,
    ncols,
    figsize=(0.875 * 3 * ncols, 3 * nrows),
    sharex=True,
    sharey=True,
    squeeze=False,
    layout="constrained",
)

for i, true_counts in enumerate(true_counts_list):

    vmax = 0.18 * true_counts / 1e7

    for j, beta_rel in enumerate(beta_rels):
        beta = beta_rel * (2e-4) * (true_counts / 3e7)

        ref_file = (
            sim_path
            / f"rdp_t_{true_counts:.1E}_b_{beta_rel:.2f}_g_{gamma_rdp:.2f}_n_{num_iter_bfgs_ref}_nr_{num_rings}_tof_{tof}_cf_{contam_fraction}_s_{seed}_ph_{phantom_type}.npy"
        )

        x_ref = np.load(ref_file)
        sl2 = x_ref.shape[2] // 2
        sl1 = x_ref.shape[1] // 2
        sl0 = x_ref.shape[0] // 2

        im = ax2[i, j].imshow(
            np.vstack(
                (x_ref[crop:-crop, :, sl2], x_ref[sl0, :, :].T, x_ref[:, sl1, :].T)
            ),
            vmin=0,
            vmax=vmax,
            cmap="Greys",
        )

        fig2.colorbar(im, ax=ax2[i, j], location="bottom", fraction=0.035)

for i, axx in enumerate(ax2[0, :]):
    axx.set_title(
        f"L-BFGS-B ref. solution $\\tilde{{ \\beta }}$ = {beta_rels[i]:.1f}",
        fontsize="medium",
    )

# split true_counts_list[i] into exponent and mantissa
exp_list = [int(np.log10(true_counts_list[i])) for i in range(len(true_counts_list))]
mant_list = [
    true_counts_list[i] / 10 ** exp_list[i] for i in range(len(true_counts_list))
]

for i, axx in enumerate(ax2[:, 0]):
    axx.set_ylabel(f"true counts = ${mant_list[i]:.0f} \\cdot 10^{exp_list[i]}$")

for axx in ax2.ravel():
    axx.set_xticks([])
    axx.set_yticks([])

# %%
fig2.show()

fig2.savefig(f"fig1_ref_recons_tof_{tof}.png", dpi=300)
