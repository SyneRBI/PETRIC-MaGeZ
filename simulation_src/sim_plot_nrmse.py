import json
import argparse
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from typing import List
from matplotlib.backends.backend_pdf import PdfPages


def create_figures(
    sim_path_str: str,
    methods_eta_ss: List[tuple[str, float, int, float, int, str]] = [
        ("SGD", 0.1, 27, 1.0, 1, "wor"),
        ("SAGA", 0.0, 27, 1.0, 1, "wor"),
        ("SVRG", 0.0, 27, 1.0, 1, "wor"),
    ],
    true_counts_list: List[float] = [1e7, 1e8],
    beta_rels: List[float] = [1.0, 4.0, 16.0],
    precond_types: List[int] = [1, 2],
    gamma_rdp: float = 2,
    num_iter_bfgs_ref: int = 500,
    num_rings: int = 17,
    tof: bool = False,
    contam_fraction: float = 0.5,
    seed: int = 1,
    phantom_type: int = 1,
    num_epochs: int = 100,
    xmin: int = 1,
    xmax: int | None = None,
    ymin: float = 1e-5,
    ymax: float = 1e0,
    xaxis: str = "walltime",
):
    sim_path = Path(sim_path_str)

    nrows = len(true_counts_list)
    ncols = len(beta_rels)

    fig, ax = plt.subplots(
        nrows,
        ncols,
        figsize=(3 * ncols, 3 * nrows),
        tight_layout=True,
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    fig2, ax2 = plt.subplots(
        nrows,
        ncols,
        figsize=(3 * ncols, 3 * nrows),
        tight_layout=True,
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    fig3, ax3 = plt.subplots(
        nrows,
        ncols,
        figsize=(3 * ncols, 1 * nrows),
        tight_layout=True,
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for axx in ax.ravel():
        axx.axhline(1e-2, color="k")

    for i, true_counts in enumerate(true_counts_list):
        for j, beta_rel in enumerate(beta_rels):
            beta = beta_rel * (2e-4) * (true_counts / 3e7)

            ref_file = (
                sim_path
                / f"rdp_t_{true_counts:.1E}_b_{beta_rel:.2f}_g_{gamma_rdp:.2f}_n_{num_iter_bfgs_ref}_nr_{num_rings}_tof_{tof}_cf_{contam_fraction}_s_{seed}_ph_{phantom_type}.npy"
            )

            x_ref = np.load(ref_file)
            sl2 = x_ref.shape[2] // 2
            sl0 = x_ref.shape[0] // 2

            ax2[i, j].imshow(
                x_ref[..., sl2], vmin=0, vmax=0.18 * true_counts / 1e7, cmap="Greys"
            )
            ax3[i, j].imshow(
                x_ref[sl0, ...].T, vmin=0, vmax=0.18 * true_counts / 1e7, cmap="Greys"
            )

            # loop over pairs of method and eta stored in methods_eta

            for im, (
                method,
                eta,
                num_subsets,
                init_step_size,
                subset_seed,
                subset_sampling_method,
            ) in enumerate(methods_eta_ss):
                for ip, precond_type in enumerate(precond_types):
                    res_file = (
                        ref_file.parent
                        / f"{ref_file.stem}_ne_{num_epochs}_ns_{num_subsets}_m_{method}_pc_{precond_type}_s0_{init_step_size:.2E}_eta_{eta:.2E}_ss_{subset_seed}_ssm_{subset_sampling_method}.json"
                    )

                    # read nrmse_stochastic from the JSON file
                    with open(res_file, "r", encoding="utf-8") as f:
                        content = json.load(f)
                        nrmse_stochastic = content["nrmse_stochastic"]
                        walltime = content["walltime_stochastic"]

                    # plot the nrmse_stochastic values
                    if ip == 0:
                        ls = (0, (1, 1))
                        lw = 1.5
                    else:
                        ls = "-"
                        lw = 1.5

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

                    ax[i, j].loglog(
                        x,
                        nrmse_stochastic[(num_subsets - 1) :: num_subsets],
                        color=plt.cm.tab10(im),
                        linestyle=ls,
                        linewidth=lw,
                        label=f"{method} $\\eta$={eta:.2f} pc={precond_type} ns={num_subsets} s0={init_step_size:.1f}",
                    )

    for axx in ax.ravel():
        axx.grid(True, which="both", ls="-", lw=0.1)
        # axx.set_xlim(xmin, xmax)
        axx.set_ylim(ymin, ymax)

    for axx in ax[-1, :]:
        if xaxis == "data_passes":
            axx.set_xlabel("data passes")
        elif xaxis == "walltime":
            axx.set_xlabel("walltime [s]")

    for i, axx in enumerate(ax[0, :]):
        axx.set_title(f"$\\beta$ = {beta_rels[i]:.1f}", fontsize="medium")
    for i, axx in enumerate(ax2[0, :]):
        axx.set_title(f"$\\beta$ = {beta_rels[i]:.1f}", fontsize="medium")
    for i, axx in enumerate(ax3[0, :]):
        axx.set_title(f"$\\beta$ = {beta_rels[i]:.1f}", fontsize="medium")

    for i, axx in enumerate(ax[:, 0]):
        axx.set_ylabel(f"NRMSE true counts: {true_counts_list[i]:.1E}")
    for i, axx in enumerate(ax2[:, 0]):
        axx.set_ylabel(f"true counts: {true_counts_list[i]:.1E}")
    for i, axx in enumerate(ax3[:, 0]):
        axx.set_ylabel(f"{true_counts_list[i]:.1E}")

    ax[0, 0].legend(fontsize="xx-small", ncol=2, loc="lower right")

    fig.suptitle(f"{sim_path_str}, TOF = {tof}")

    fig2.suptitle(f"{sim_path_str} TOF = {tof}")
    fig3.suptitle(f"{sim_path_str} TOF = {tof}")

    return fig, fig2, fig3


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot NRMSE for different methods and parameters."
    )

    parser.add_argument(
        "sim_path",
        type=str,
        help="Path to simulation results.",
    )

    parser.add_argument(
        "--methods_eta_ss",
        type=lambda s: [
            (
                method_eta.split(":")[0],
                float(method_eta.split(":")[1]),
                int(method_eta.split(":")[2]),
                float(method_eta.split(":")[3]),
                int(method_eta.split(":")[4]),
                str(method_eta.split(":")[5]),
            )
            for method_eta in s.split(",")
        ],
        default=[
            ("SGD", 0.1, 27, 1.0, 1, "wor"),
            ("SAGA", 0.0, 27, 1.0, 1, "wor"),
            ("SVRG", 0.0, 27, 1.0, 1, "wor"),
        ],
        help='List of method,eta,num_subsets,init_step,subset_seed,sampling_method values in the format "method:eta:num_ss:init_step:subset_seed:sampling_method,method:eta:num_ss:init_step:subset_seed,sampling_method...".',
    )
    parser.add_argument(
        "--true_counts_list",
        nargs="+",
        type=float,
        default=[1e7, 1e8],
        help="List of true counts.",
    )
    parser.add_argument(
        "--beta_rels",
        nargs="+",
        type=float,
        default=[1.0, 4.0, 16.0],
        help="List of beta relative values.",
    )
    parser.add_argument(
        "--precond_types",
        nargs="+",
        type=int,
        default=[1, 2],
        help="List of preconditioner types.",
    )
    parser.add_argument("--gamma_rdp", type=float, default=2, help="Gamma RDP value.")
    parser.add_argument(
        "--num_iter_bfgs_ref", type=int, default=500, help="Number of BFGS iterations."
    )
    parser.add_argument("--num_rings", type=int, default=17, help="Number of rings.")
    parser.add_argument("--tof", action="store_true", help="TOF flag")
    parser.add_argument(
        "--contam_fraction", type=float, default=0.5, help="Contamination fraction."
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument("--phantom_type", type=int, default=1, help="Phantom type.")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs.")
    parser.add_argument(
        "--show_every_update",
        action="store_true",
        help="Show NRMSE for every update instead of complete epochs",
    )
    parser.add_argument("--xmin", type=int, default=1, help="Minimum x-axis value.")
    parser.add_argument("--xmax", type=int, default=None, help="Maximum x-axis value.")
    parser.add_argument(
        "--ymin", type=float, default=1e-5, help="Minimum y-axis value."
    )
    parser.add_argument("--ymax", type=float, default=1e0, help="Maximum y-axis value.")

    args = parser.parse_args()

    # %%
    sim_path_str: str = args.sim_path
    methods_eta_ss: List[tuple[str, float, int, float, int, str]] = args.methods_eta_ss

    true_counts_list: List[float] = args.true_counts_list
    beta_rels: List[float] = args.beta_rels
    precond_types: List[int] = args.precond_types

    gamma_rdp: float = args.gamma_rdp
    num_iter_bfgs_ref: int = args.num_iter_bfgs_ref
    num_rings: int = args.num_rings
    tof: bool = args.tof
    contam_fraction: float = args.contam_fraction
    seed: int = args.seed
    phantom_type: int = args.phantom_type

    num_epochs: int = args.num_epochs

    show_complete_epochs_only: bool = not args.show_every_update

    xmin: int = args.xmin

    ymin: float = args.ymin
    ymax: float = args.ymax

    if args.xmax is None:
        xmax: int = num_epochs
    else:
        xmax: int = args.xmax

    fig, fig2, fig3 = create_figures(
        sim_path_str=sim_path_str,
        methods_eta_ss=methods_eta_ss,
        true_counts_list=true_counts_list,
        beta_rels=beta_rels,
        precond_types=precond_types,
        gamma_rdp=gamma_rdp,
        num_iter_bfgs_ref=num_iter_bfgs_ref,
        num_rings=num_rings,
        tof=tof,
        contam_fraction=contam_fraction,
        seed=seed,
        phantom_type=phantom_type,
        num_epochs=num_epochs,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
    )

    fig.show()
    fig2.show()
    fig3.show()

    output_pdf_path = Path(sim_path_str) / f"00_tof_{tof}.pdf"
    with PdfPages(output_pdf_path) as pdf:
        pdf.savefig(fig)
        pdf.savefig(fig2)
        pdf.savefig(fig3)
