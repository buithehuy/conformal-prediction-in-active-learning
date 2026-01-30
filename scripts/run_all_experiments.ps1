# Script to run all experiments across all datasets and strategies

Write-Host "========================================"
Write-Host "Running Complete AL Experiments"
Write-Host "All Datasets x All Strategies"
Write-Host "========================================"

# CIFAR-10 - All strategies
Write-Host ""
Write-Host "==== CIFAR-10 ===="
python src/train.py -m data=cifar10 `
  experiment=al_random,al_entropy,al_least_confidence,al_margin,al_cp_size,al_cp_v_shaped,al_combined,al_combined_v_shaped

# CIFAR-100 - All strategies  
Write-Host ""
Write-Host "==== CIFAR-100 ===="
python src/train.py -m data=cifar100 model=resnet18_cifar100 `
  experiment=al_random,al_entropy,al_least_confidence,al_margin,al_cp_size,al_cp_v_shaped,al_combined,al_combined_v_shaped

# STL-10 - All strategies (smaller dataset, adjusted rounds)
Write-Host ""
Write-Host "==== STL-10 ===="
python src/train.py -m data=stl10 model=resnet18_stl10 `
  al.num_rounds=3 al.budget_per_round=300 `
  experiment=al_random,al_entropy,al_least_confidence,al_margin,al_cp_size,al_cp_v_shaped,al_combined,al_combined_v_shaped

# SVHN - All strategies
Write-Host ""
Write-Host "==== SVHN ===="
python src/train.py -m data=svhn `
  experiment=al_random,al_entropy,al_least_confidence,al_margin,al_cp_size,al_cp_v_shaped,al_combined,al_combined_v_shaped

Write-Host ""
Write-Host "========================================"
Write-Host "All experiments completed!"
Write-Host "Results saved in logs/multiruns/"
Write-Host "========================================"
