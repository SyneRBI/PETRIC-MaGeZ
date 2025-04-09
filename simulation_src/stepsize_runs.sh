#!/bin/bash

# Define the different true_counts values
trueCountsValues=(1e8 1e7)

# Define the different beta_rel values
betaRelValues=(16.0 4.0 1.0)

pc_types=(2)

etas=(0.01)

subset_values=(27)
stepsize_rule=(const)
subset_sampling_method=(wor)
num_epochs=100

phantom_type=1

# non TOF simulations
for betaRel in "${betaRelValues[@]}"; do
    for trueCounts in "${trueCountsValues[@]}"; do
        for pc_type in "${pc_types[@]}"; do
            for eta in "${etas[@]}"; do
                for num_subsets in "${subset_values[@]}"; do
                    for sampling in "${subset_sampling_method[@]}"; do
                        for stepsize_rule in "${stepsize_rule[@]}"; do
                            echo "Running simulation with true_counts=$trueCounts beta_rel=$betaRel etas=$eta and sampling=$sampling"
                            CUDA_VISIBLE_DEVICES=1 python sim_ablation.py --true_counts $trueCounts --beta_rel $betaRel --num_epochs $num_epochs --num_subsets $num_subsets --precond_type $pc_type --phantom_type $phantom_type --eta $eta --method SVRG --subset_sampling_method $sampling --step_size_rule $stepsize_rule
                        done
                    done
                done
            done
        done
    done
done

# TOF simulations
for betaRel in "${betaRelValues[@]}"; do
    for trueCounts in "${trueCountsValues[@]}"; do
        for pc_type in "${pc_types[@]}"; do
            for eta in "${etas[@]}"; do
                for num_subsets in "${subset_values[@]}"; do
                    for sampling in "${subset_sampling_method[@]}"; do
                        for stepsize_rule in "${stepsize_rule[@]}"; do
                            echo "Running simulation with true_counts=$trueCounts beta_rel=$betaRel etas=$eta and sampling=$sampling"
                            CUDA_VISIBLE_DEVICES=1 python sim_ablation.py --tof --true_counts $trueCounts --beta_rel $betaRel --num_epochs $num_epochs --num_subsets $num_subsets --precond_type $pc_type --phantom_type $phantom_type --eta $eta --method SVRG --subset_sampling_method $sampling --step_size_rule $stepsize_rule
                        done
                    done    
                done
            done
        done
    done
done

