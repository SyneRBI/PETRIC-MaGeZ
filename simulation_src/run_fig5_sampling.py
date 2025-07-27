import itertools
import subprocess

# Define the different parameter values
true_counts_values = [1e8, 1e7]
beta_rel_values = [16.0, 4.0, 1.0]
pc_types = [2]
etas = [0.02]
subset_values = [27]
subset_sampling_methods = ["wr", "wor", "hm", "cofactor"]
num_epochs = 100
phantom_type = 1


def run_command(command):
    """Helper function to run a command using subprocess."""
    print("Running:", " ".join(command))
    subprocess.run(command)


# Non-TOF simulations
print("Running non-TOF simulations")
print("---------------------------------")
for beta_rel, true_counts, pc_type, eta, num_subsets, sampling in itertools.product(
    beta_rel_values,
    true_counts_values,
    pc_types,
    etas,
    subset_values,
    subset_sampling_methods,
):
    print(
        f"Running simulation with true_counts={true_counts}, beta_rel={beta_rel}, sampling={sampling}"
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
    ]
    run_command(command)

    print(
        f"Running simulation with true_counts={true_counts}, beta_rel={beta_rel}, and weighted sampling"
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
        "--gnbs",
    ]
    run_command(command)

# TOF simulations
print("---------------------------------")
print("Running TOF simulations")
print("---------------------------------")
for beta_rel, true_counts, pc_type, eta, num_subsets, sampling in itertools.product(
    beta_rel_values,
    true_counts_values,
    pc_types,
    etas,
    subset_values,
    subset_sampling_methods,
):
    print(
        f"Running simulation with true_counts={true_counts}, beta_rel={beta_rel}, sampling={sampling}, TOF=True"
    )
    command = [
        "python",
        "sim_ablation.py",
        "--tof",
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
    ]
    run_command(command)

    print(
        f"Running simulation with true_counts={true_counts}, beta_rel={beta_rel}, and weighted sampling, TOF=True"
    )
    command = [
        "python",
        "sim_ablation.py",
        "--tof",
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
        "--gnbs",
    ]
    run_command(command)
