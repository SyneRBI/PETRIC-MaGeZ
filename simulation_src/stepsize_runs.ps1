# Define the different true_counts values
$trueCountsValues = @(1e8, 1e7)

# Define the different beta_rel values
$betaRelValues = @(16.0, 4.0, 1.0)


$pc_types = @(2)
$etas = @(0.02)
$subset_values = @(27)
$stepsize_rule = @("decaying", "bb", "alg1", "const")
$subset_sampling_method = @("wor")
$num_epochs = 100
$phantom_type = 1

# non TOF simulations
foreach ($betaRel in $betaRelValues) {
    foreach ($trueCounts in $trueCountsValues) {
        foreach ($pc_type in $pc_types) {
            foreach ($eta in $etas) {
                foreach ($num_subsets in $subset_values) {
                    foreach ($sampling in $subset_sampling_method) {
                        foreach ($step_rule in $stepsize_rule) {
                            Write-Output "Running simulation with true_counts=$trueCounts beta_rel=$betaRel etas=$eta and sampling=$sampling"
                            python sim_ablation.py `
                                --true_counts $trueCounts `
                                --beta_rel $betaRel `
                                --num_epochs $num_epochs `
                                --num_subsets $num_subsets `
                                --precond_type $pc_type `
                                --phantom_type $phantom_type `
                                --eta $eta `
                                --method SVRG `
                                --subset_sampling_method $sampling `
                                --step_size_rule $step_rule
                        }
                    }
                }
            }
        }
    }
}

# TOF simulations
foreach ($betaRel in $betaRelValues) {
    foreach ($trueCounts in $trueCountsValues) {
        foreach ($pc_type in $pc_types) {
            foreach ($eta in $etas) {
                foreach ($num_subsets in $subset_values) {
                    foreach ($sampling in $subset_sampling_method) {
                        foreach ($step_rule in $stepsize_rule) {
                            Write-Output "Running simulation with true_counts=$trueCounts beta_rel=$betaRel etas=$eta and sampling=$sampling"
                            python sim_ablation.py `
                                --tof `
                                --true_counts $trueCounts `
                                --beta_rel $betaRel `
                                --num_epochs $num_epochs `
                                --num_subsets $num_subsets `
                                --precond_type $pc_type `
                                --phantom_type $phantom_type `
                                --eta $eta `
                                --method SVRG `
                                --subset_sampling_method $sampling `
                                --step_size_rule $step_rule
                        }
                    }
                }
            }
        }
    }
}
