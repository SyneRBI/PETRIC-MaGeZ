# Define the different parameter values
$trueCountsValues = @(1e8, 1e7)
$betaRelValues = @(16.0, 4.0, 1.0)
$pc_types = @(2)
$etas = @(0.01)
$subset_values = @(27)
$subset_sampling_method = @("wr", "wor", "hm", "cofactor")
$num_epochs = 100
$phantom_type = 1

Write-Output "Running non-TOF simulations"
Write-Output "---------------------------------"

foreach ($betaRel in $betaRelValues) {
    foreach ($trueCounts in $trueCountsValues) {
        foreach ($pc_type in $pc_types) {
            foreach ($eta in $etas) {
                foreach ($num_subsets in $subset_values) {
                    foreach ($sampling in $subset_sampling_method) {
                        Write-Output "Running simulation with true_counts=$trueCounts beta_rel=$betaRel and sampling=$sampling"
                        python sim_ablation.py `
                            --true_counts $trueCounts `
                            --beta_rel $betaRel `
                            --num_epochs $num_epochs `
                            --num_subsets $num_subsets `
                            --precond_type $pc_type `
                            --phantom_type $phantom_type `
                            --eta $eta `
                            --method SVRG `
                            --subset_sampling_method $sampling
                    }

                    Write-Output "Running simulation with true_counts=$trueCounts beta_rel=$betaRel and weighted sampling"
                    python sim_ablation.py `
                        --true_counts $trueCounts `
                        --beta_rel $betaRel `
                        --num_epochs $num_epochs `
                        --num_subsets $num_subsets `
                        --precond_type $pc_type `
                        --phantom_type $phantom_type `
                        --eta $eta `
                        --method SVRG `
                        --gnbs
                }
            }
        }
    }
}

Write-Output "---------------------------------"
Write-Output "Running TOF simulations"
Write-Output "---------------------------------"

foreach ($betaRel in $betaRelValues) {
    foreach ($trueCounts in $trueCountsValues) {
        foreach ($pc_type in $pc_types) {
            foreach ($eta in $etas) {
                foreach ($num_subsets in $subset_values) {
                    foreach ($sampling in $subset_sampling_method) {
                        Write-Output "Running simulation with true_counts=$trueCounts and beta_rel=$betaRel and sampling=$sampling"
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
                            --subset_sampling_method $sampling
                    }

                    Write-Output "Running simulation with true_counts=$trueCounts beta_rel=$betaRel and weighted sampling"
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
                        --gnbs
                }
            }
        }
    }
}
