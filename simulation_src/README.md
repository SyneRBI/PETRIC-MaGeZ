# Project README

## Setup / Installation

To set up the environment required to run the scripts, use the provided `environment.yaml` file. Follow these steps:

1. Create the Conda environment:
   ```bash
   conda env create -f environment.yaml
   ```

## Running reconstructions of simulated data 

Execute

```
python run_fig2_4.py
python run_fig5.py
python run_fig6.py
```

Results will be saved into `sim_results` (first script) and `sim_results_ablation` (last two scripts). 
The first time you run `run_fig2_4.py`, the reference reconstruction will be saved into
``.

## Generate figures

Execute the plot scripts (after adjusting the `sim_path_str`) in the scripts.
To regenerate the plots of our paper use `sim_path_str = sim_results_paper` (default).

```
python fig1_ref_recons.py
python fig2_pc.py
python fig3_fig4_alg_vs_subsets.py
python fig5_sampling.py
python fig6_stepsize.py
```