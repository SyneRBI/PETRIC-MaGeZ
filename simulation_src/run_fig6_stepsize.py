import itertools
import subprocess

# Define the different parameter values
true_counts_values = [1e8, 1e7]
beta_rel_values = [16.0, 4.0, 1.0]
pc_types = [2]
etas = [0.02]
subset_values = [27]
stepsize_rules = ["decaying", "bb", "alg1", "const"]
subset_sampling_methods = ["wor"]
num_epochs = 100
phantom_type = 1


# Function to run simulations
def run_simulation(tof=False):
    for (
        beta_rel,
        true_counts,
        pc_type,
        eta,
        num_subsets,
        sampling,
        step_rule,
    ) in itertools.product(
        beta_rel_values,
        true_counts_values,
        pc_types,
        etas,
        subset_values,
        subset_sampling_methods,
        stepsize_rules,
    ):
        print(
            f"Running simulation with true_counts={true_counts}, beta_rel={beta_rel}, eta={eta}, sampling={sampling}, step_rule={step_rule}, TOF={tof}"
        )
        command = [
            "python",
            "sim_ablation.py",
            "--true_counts",
            str(true_counts),
            "--beta_rel",
            str(beta_rel),
            "--num_epochs",
            str(num_epochs),
            "--num_subsets",
            str(num_subsets),
            "--precond_type",
            str(pc_type),
            "--phantom_type",
            str(phantom_type),
            "--eta",
            str(eta),
            "--method",
            "SVRG",
            "--subset_sampling_method",
            sampling,
            "--step_size_rule",
            step_rule,
        ]
        if tof:
            command.append("--tof")
        subprocess.run(command)


# Run non-TOF simulations
run_simulation(tof=False)

# Run TOF simulations
run_simulation(tof=True)
