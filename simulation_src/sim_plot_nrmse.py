import json
import argparse
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np

from pathlib import Path
from typing import List
from matplotlib.backends.backend_pdf import PdfPages


def create_figures(
    sim_path_str: str,
    methods: List[str] = ["SGD", "SAGA", "SVRG"],
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
    num_subsets: int = 27,
    init_step_size: float = 1.0,
    eta: float = 0.0,
    show_complete_epochs_only: bool = True,
    xmin: int = 0,
    xmax: int | None = None,
    ymin: float = 1e-5,
    ymax: float = 1e0,
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
    )

    fig2, ax2 = plt.subplots(
        nrows,
        ncols,
        figsize=(3 * ncols, 3 * nrows),
        tight_layout=True,
        sharex=True,
        sharey=True,
    )

    fig3, ax3 = plt.subplots(
        nrows,
        ncols,
        figsize=(3 * ncols, 1 * nrows),
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

            for im, method in enumerate(methods):
                for ip, precond_type in enumerate(precond_types):
                    res_file = (
                        ref_file.parent
                        / f"{ref_file.stem}_ne_{num_epochs}_ns_{num_subsets}_m_{method}_pc_{precond_type}_s0_{init_step_size:.2E}_eta_{eta:.2E}.json"
                    )

                    # read nrmse_stochastic from the JSON file
                    with open(res_file, "r", encoding="utf-8") as f:
                        content = json.load(f)
                        nrmse_stochastic = content["nrmse_stochastic"]

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

                    ax[i, j].loglog(
                        data_passes,
                        nrmse_stochastic[(num_subsets - 1) :: num_subsets],
                        color=plt.cm.tab10(im),
                        linestyle=ls,
                        linewidth=lw,
                        label=f"{method} pc={precond_type}",
                    )

    for axx in ax.ravel():
        # axx.xaxis.set_minor_locator(ticker.MultipleLocator(10))
        axx.grid(True, which="both", ls="-", lw=0.1)
        axx.set_xlim(xmin, xmax)
        axx.set_ylim(ymin, ymax)

    for axx in ax[-1, :]:
        axx.set_xlabel("data passes")

    for i, axx in enumerate(ax[0, :]):
        axx.set_title(f"beta = {beta_rels[i]:.1f}", fontsize="medium")
    for i, axx in enumerate(ax2[0, :]):
        axx.set_title(f"beta = {beta_rels[i]:.1f}", fontsize="medium")
    for i, axx in enumerate(ax3[0, :]):
        axx.set_title(f"beta = {beta_rels[i]:.1f}", fontsize="medium")

    for i, axx in enumerate(ax[:, 0]):
        axx.set_ylabel(f"NRMSE true counts: {true_counts_list[i]:.1E}")
    for i, axx in enumerate(ax2[:, 0]):
        axx.set_ylabel(f"true counts: {true_counts_list[i]:.1E}")
    for i, axx in enumerate(ax3[:, 0]):
        axx.set_ylabel(f"{true_counts_list[i]:.1E}")

    ax[0, 0].legend(fontsize="x-small", ncol=2, loc="lower right")

    fig.suptitle(
        f"{sim_path_str} num_subsets = {num_subsets}, eta = {eta:.2f}, s0 = {init_step_size:.2f}, TOF = {tof}"
    )

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
        "--methods", nargs="+", default=["SGD", "SAGA", "SVRG"], help="List of methods."
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
        "--num_subsets", type=int, default=27, help="Number of subsets."
    )
    parser.add_argument(
        "--init_step_size", type=float, default=1.0, help="Initial step size."
    )
    parser.add_argument("--eta", type=float, default=0.0, help="Eta value.")
    parser.add_argument(
        "--show_every_update",
        action="store_true",
        help="Show NRMSE for every update instead of complete epochs",
    )
    parser.add_argument("--xmin", type=int, default=0, help="Minimum x-axis value.")
    parser.add_argument("--xmax", type=int, default=None, help="Maximum x-axis value.")
    parser.add_argument(
        "--ymin", type=float, default=1e-5, help="Minimum y-axis value."
    )
    parser.add_argument("--ymax", type=float, default=1e0, help="Maximum y-axis value.")

    args = parser.parse_args()

    # %%
    methods: List[str] = args.methods
    true_counts_list: List[float] = args.true_counts_list
    beta_rels: List[float] = args.beta_rels
    precond_types: List[int] = args.precond_types

    sim_path_str: str = args.sim_path

    gamma_rdp: float = args.gamma_rdp
    num_iter_bfgs_ref: int = args.num_iter_bfgs_ref
    num_rings: int = args.num_rings
    tof: bool = args.tof
    contam_fraction: float = args.contam_fraction
    seed: int = args.seed
    phantom_type: int = args.phantom_type

    num_epochs: int = args.num_epochs
    num_subsets: int = args.num_subsets

    init_step_size: float = args.init_step_size
    eta: float = args.eta

    show_complete_epochs_only: bool = not args.show_every_update

    xmin: int = args.xmin

    ymin: float = args.ymin
    ymax: float = args.ymax

    if args.xmax is None:
        xmax: int = num_epochs
    else:
        xmax: int = args.xmax

    fig, fig2, fig3 = create_figures(
        sim_path_str,
        methods,
        true_counts_list,
        beta_rels,
        precond_types,
        gamma_rdp,
        num_iter_bfgs_ref,
        num_rings,
        tof,
        contam_fraction,
        seed,
        phantom_type,
        num_epochs,
        num_subsets,
        init_step_size,
        eta,
        show_complete_epochs_only,
        xmin,
        xmax,
        ymin,
        ymax,
    )

    fig.show()
    fig2.show()
    fig3.show()

    output_pdf_path = (
        Path(sim_path_str)
        / f"00_ns_{num_subsets}_tof_{tof}_s0_{init_step_size}_eta_{eta}.pdf"
    )
    with PdfPages(output_pdf_path) as pdf:
        pdf.savefig(fig)
        pdf.savefig(fig2)
        pdf.savefig(fig3)
