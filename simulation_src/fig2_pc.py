import json
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from typing import List


sim_path_str: str = "sim_results_paper"
ref_path_str: str = "sim_results_ref_recons"
methods: list[str] = ["SVRG", "SAGA", "SGD"]
num_subsets: int = 27
init_step_sizes: List[float] = [1.5, 1.0, 0.3]
true_counts_list: List[float] = [1e7, 1e8]
beta_rels: List[float] = [16.0, 4.0, 1.0]
precond_type_dict: dict[int, tuple[str, str]] = {1: ("MLEM", "--"), 2: ("harm", "-")}
gamma_rdp: float = 2
num_iter_bfgs_ref: int = 500
num_rings: int = 17
tof: bool = True
contam_fraction: float = 0.5
seed: int = 1
phantom_type: int = 1
num_epochs: int = 100
xmin: float | None = 2.0
xmax: float | None = 400.0
ymin: float = 1e-3
ymax: float = 6e-1
xaxis: str = "walltime"
eta: float = 0.02
subset_seed: int = 1
subset_sampling_method: str = "wor"
add_legend: bool = True

# %%
sim_path = Path(sim_path_str)

nrows = len(true_counts_list)
ncols = len(beta_rels)

for method in methods:
    fig, ax = plt.subplots(
        nrows,
        ncols,
        figsize=(8, 3.5),
        tight_layout=True,
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for axx in ax.ravel():
        axx.axhline(1e-2, color="k")

    lw = 1.0

    handles = []

    for i, true_counts in enumerate(true_counts_list):
        for j, beta_rel in enumerate(beta_rels):
            beta = beta_rel * (2e-4) * (true_counts / 3e7)

            ref_file = (
                Path(ref_path_str)
                / f"rdp_t_{true_counts:.1E}_b_{beta_rel:.2f}_g_{gamma_rdp:.2f}_n_{num_iter_bfgs_ref}_nr_{num_rings}_tof_{tof}_cf_{contam_fraction}_s_{seed}_ph_{phantom_type}.npy"
            )

            for i_s, init_step_size in enumerate(init_step_sizes):
                for i_p, (precond_type, ls) in precond_type_dict.items():
                    res_file = (
                        sim_path
                        / f"{ref_file.stem}_ne_{num_epochs}_ns_{num_subsets}_m_{method}_pc_{i_p}_s0_{init_step_size:.2E}_eta_{eta:.2E}_ss_{subset_seed}_ssm_{subset_sampling_method}.json"
                    )

                    # read nrmse_stochastic from the JSON file
                    with open(res_file, "r", encoding="utf-8") as f:
                        content = json.load(f)
                        nrmse_stochastic = content["nrmse_stochastic"]
                        walltime = content["walltime_stochastic"]

                    # the NRMSE callback is called after each update
                    # we select only the values at the end of each epoch
                    data_passes = np.arange(1, num_epochs + 1)

                    if method == "SVRG":
                        # the extra data passes that we perform in SVRG
                        # where every 2nd epoch is an extra full pass is needed
                        data_passes += np.repeat(data_passes, 2)[:num_epochs]

                    if xaxis == "data_passes":
                        x = data_passes
                    elif xaxis == "walltime":
                        x = walltime[(num_subsets - 1) :: num_subsets]

                    label = f"$\\tau_0$={init_step_size:.1f}, PC={precond_type}"

                    (line,) = ax[i, j].loglog(
                        x,
                        nrmse_stochastic[(num_subsets - 1) :: num_subsets],
                        color=plt.cm.tab10(i_s),
                        linestyle=ls,
                        linewidth=lw,
                        label=label,
                    )

                    if i == 0 and j == 0:
                        handles.append(line)

    for axx in ax.ravel():
        axx.grid(True, which="both", ls="-", lw=0.1)
        axx.set_xlim(xmin, xmax)
        axx.set_ylim(ymin, ymax)

    for axx in ax[-1, :]:
        if xaxis == "data_passes":
            axx.set_xlabel("data passes")
        elif xaxis == "walltime":
            axx.set_xlabel("walltime [s]")

    for i, axx in enumerate(ax[0, :]):
        axx.set_title(f"$\\tilde{{\\beta}}$ = {beta_rels[i]:.1f}", fontsize="medium")

    for i, axx in enumerate(ax[:, 0]):
        exp = int(np.log10(true_counts_list[i]))
        mant = true_counts_list[i] / 10**exp
        axx.set_ylabel(f"${mant:.0f} \\cdot 10^{exp}$ counts \n NRMSE")

    if add_legend:
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=3,
            bbox_to_anchor=(0.55, 0.43),
            fancybox=True,
            fontsize="x-small",
            framealpha=0.9,
            numpoints=2,
        )

    fig.show()
    fig.savefig(
        f"fig2_pc_{method}_{num_subsets}_{eta:.2E}_{subset_sampling_method}_tof_{tof}.pdf"
    )
