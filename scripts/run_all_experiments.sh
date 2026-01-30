#!/bin/bash
# Script to run all experiments across all datasets and strategies

echo "========================================"
echo "Running Complete AL Experiments"
echo "All Datasets x All Strategies"
echo "========================================"

# CIFAR-10 - All strategies
echo ""
echo "==== CIFAR-10 ===="
python src/train.py -m data=cifar10 \
  experiment=al_random,al_entropy,al_least_confidence,al_margin,al_cp_size,al_cp_v_shaped,al_combined,al_combined_v_shaped

# CIFAR-100 - All strategies  
echo ""
echo "==== CIFAR-100 ===="
python src/train.py -m data=cifar100 model=resnet18_cifar100 \
  experiment=al_random,al_entropy,al_least_confidence,al_margin,al_cp_size,al_cp_v_shaped,al_combined,al_combined_v_shaped

# STL-10 - All strategies (smaller dataset, adjusted rounds)
echo ""
echo "==== STL-10 ===="
python src/train.py -m data=stl10 model=resnet18_stl10 \
  al.num_rounds=3 al.budget_per_round=300 \
  experiment=al_random,al_entropy,al_least_confidence,al_margin,al_cp_size,al_cp_v_shaped,al_combined,al_combined_v_shaped

# SVHN - All strategies
echo ""
echo "==== SVHN ===="
python src/train.py -m data=svhn \
  experiment=al_random,al_entropy,al_least_confidence,al_margin,al_cp_size,al_cp_v_shaped,al_combined,al_combined_v_shaped

echo ""
echo "========================================"
echo "All experiments completed!"
echo "Results saved in logs/multiruns/"
echo "========================================"
