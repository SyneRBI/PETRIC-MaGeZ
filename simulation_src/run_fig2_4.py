import itertools
import subprocess

# Define the different parameter values
true_counts_values = [1e7, 1e8]
beta_rel_values = [16.0, 4.0, 1.0]
pc_types = [1, 2]
etas = [0.02]
subset_values = [8, 27, 54, 108]
init_step_size_values = [1.5, 1.0, 0.3]
phantom_type = 1
methods = ["SVRG", "SAGA", "SGD"]
num_epochs = 100
subset_seeds = [1, 2, 3, 4, 5]  # Run each scenario 5 times


# Helper function to run a command
def run_command(command):
    print("Running:", " ".join(command))
    subprocess.run(command)


# Iterate over all combinations of parameters
for (
    subset_seed,
    beta_rel,
    true_counts,
    pc_type,
    eta,
    num_subsets,
    init_step_size,
    method,
) in itertools.product(
    subset_seeds,
    beta_rel_values,
    true_counts_values,
    pc_types,
    etas,
    subset_values,
    init_step_size_values,
    methods,
):
    # Non-TOF simulation
    command = [
        "python",
        "sim_stochastic_grad.py",
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
        "--init_step_size",
        str(init_step_size),
        "--method",
        method,
        "--subset_seed",
        str(subset_seed),
    ]
    print(command)
    run_command(command)

    # TOF simulation
    command.append("--tof")
    print(command)
    run_command(command)
