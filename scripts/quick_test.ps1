# Quick Test Script - Test configuration works correctly

Write-Host "========================================"
Write-Host "Quick Test - 1 Round, Small Budget"
Write-Host "========================================"

Write-Host "`n==== Testing CIFAR-10 + Random ===="
python src/train.py `
  data=cifar10 `
  experiment=al_random `
  al.num_rounds=1 `
  al.budget_per_round=100 `
  trainer.max_epochs=2

Write-Host "`n==== Testing CIFAR-100 + Entropy ===="
python src/train.py `
  data=cifar100 `
  model=resnet18_cifar100 `
  experiment=al_entropy `
  al.num_rounds=1 `
  al.budget_per_round=100 `
  trainer.max_epochs=2

Write-Host "`n==== Testing STL-10 + CP Size ===="
python src/train.py `
  data=stl10 `
  model=resnet18_stl10 `
  experiment=al_cp_size `
  al.num_rounds=1 `
  al.budget_per_round=50 `
  trainer.max_epochs=2

Write-Host "`n==== Testing SVHN + Combined ===="
python src/train.py `
  data=svhn `
  experiment=al_combined `
  al.num_rounds=1 `
  al.budget_per_round=200 `
  trainer.max_epochs=2

Write-Host "`n========================================"
Write-Host "Quick test completed!"
Write-Host "If all passed, your setup is correct!"
Write-Host "========================================"
