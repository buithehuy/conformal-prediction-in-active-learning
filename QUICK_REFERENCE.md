# Quick Command Reference

## 🎯 Chạy Thực Nghiệm Nhanh

### CIFAR-10
```bash
# 1 strategy
python src/train.py data=cifar10 experiment=al_entropy

# Nhiều strategies
python src/train.py -m data=cifar10 experiment=al_random,al_entropy,al_cp_size
```

### CIFAR-100
```bash
# 1 strategy (cần model config cho 100 classes)
python src/train.py data=cifar100 model=resnet18_cifar100 experiment=al_entropy

# Nhiều strategies
python src/train.py -m data=cifar100 model=resnet18_cifar100 experiment=al_random,al_entropy,al_cp_v_shaped
```

### STL-10
```bash
# 1 strategy (96x96 images)
python src/train.py data=stl10 model=resnet18_stl10 experiment=al_cp_size

# Nhiều strategies
python src/train.py -m data=stl10 model=resnet18_stl10 experiment=al_random,al_entropy
```

### SVHN
```bash
# 1 strategy
python src/train.py data=svhn experiment=al_combined

# Nhiều strategies
python src/train.py -m data=svhn experiment=al_random,al_entropy,al_combined_v_shaped
```

## 🚀 Scripts Tự Động

```bash
# Test nhanh (1 round, budget nhỏ)
make quick-test

# Chạy TẤT CẢ datasets + TẤT CẢ strategies
make run-all

# Hoặc dùng PowerShell trực tiếp
powershell scripts/quick_test.ps1
powershell scripts/run_all_experiments.ps1
```

## 🎨 Tùy Chỉnh

```bash
# Thay đổi số rounds và budget
python src/train.py data=cifar10 experiment=al_entropy \
  al.num_rounds=10 al.budget_per_round=1000

# Thay đổi epochs
python src/train.py data=cifar10 experiment=al_random \
  trainer.max_epochs=15

# Thay đổi seed
python src/train.py data=cifar10 experiment=al_cp_size seed=123
```

## 📝 Danh Sách Strategy Codes

- `al_random` - Random sampling
- `al_entropy` - Entropy sampling  
- `al_least_confidence` - Least confidence
- `al_margin` - Margin sampling
- `al_cp_size` - CP set size
- `al_cp_v_shaped` - CP V-shaped
- `al_combined` - Combined (Entropy + CP)
- `al_combined_v_shaped` - Combined V-shaped

## 📚 Tài Liệu Chi Tiết

Xem [USAGE_GUIDE.md](USAGE_GUIDE.md) để có hướng dẫn đầy đủ với nhiều ví dụ.
