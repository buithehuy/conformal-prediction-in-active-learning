# Hướng Dẫn Chi Tiết Chạy Thực Nghiệm Active Learning

## 📋 Tổng Quan

Guide này hướng dẫn chi tiết cách chạy các thực nghiệm Active Learning với:
- **4 datasets**: CIFAR-10, CIFAR-100, STL-10, SVHN
- **8 strategies**: Random, Entropy, Least Confidence, Margin, CP Size, CP V-Shaped, Combined, Combined V-Shaped

---

## 🎯 Cấu Trúc Lệnh Cơ Bản

```bash
python src/train.py data=<DATASET> experiment=<STRATEGY> [OPTIONS]
```

### Các Thành Phần:
- `data=<DATASET>`: Chọn dataset (cifar10, cifar100, stl10, svhn)
- `experiment=<STRATEGY>`: Chọn AL strategy
- `[OPTIONS]`: Các tùy chọn bổ sung (epochs, rounds, budget, seed, etc.)

---

## 📊 Danh Sách Datasets

### 1. CIFAR-10
- **Classes**: 10
- **Train**: 50,000 images (32x32 RGB)
- **Test**: 10,000 images
- **Config**: `data=cifar10`

### 2. CIFAR-100
- **Classes**: 100
- **Train**: 50,000 images (32x32 RGB)
- **Test**: 10,000 images
- **Config**: `data=cifar100` + `model=resnet18_cifar100`

### 3. STL-10
- **Classes**: 10
- **Train**: 5,000 labeled images (96x96 RGB)
- **Test**: 8,000 images
- **Config**: `data=stl10` + `model=resnet18_stl10`

### 4. SVHN (Street View House Numbers)
- **Classes**: 10 (digits 0-9)
- **Train**: 73,257 images (32x32 RGB)
- **Test**: 26,032 images
- **Config**: `data=svhn`

---

## 🎲 Danh Sách Strategies

| Strategy | Tên Config | Mô Tả |
|----------|-----------|-------|
| Random | `al_random` | Baseline - chọn ngẫu nhiên |
| Entropy | `al_entropy` | Chọn samples có entropy cao nhất |
| Least Confidence | `al_least_confidence` | Chọn samples model kém tự tin nhất |
| Margin | `al_margin` | Chọn samples có margin nhỏ nhất giữa top-2 classes |
| CP Size | `al_cp_size` | Chọn samples có prediction set lớn nhất |
| CP V-Shaped | `al_cp_v_shaped` | Ưu tiên cả overconfident (size=0) và uncertain (size>1) |
| Combined | `al_combined` | Kết hợp Entropy + CP Size |
| Combined V-Shaped | `al_combined_v_shaped` | Kết hợp Entropy + CP V-Shaped |

---

## 💡 Ví Dụ Cụ Thể

### 1. Chạy 1 Strategy với 1 Dataset

#### CIFAR-10 + Entropy
```bash
python src/train.py data=cifar10 experiment=al_entropy
```

#### CIFAR-100 + CP V-Shaped
```bash
python src/train.py data=cifar100 model=resnet18_cifar100 experiment=al_cp_v_shaped
```

#### STL-10 + Combined
```bash
python src/train.py data=stl10 model=resnet18_stl10 experiment=al_combined
```

#### SVHN + Random
```bash
python src/train.py data=svhn experiment=al_random
```

---

### 2. Chạy Nhiều Strategies với 1 Dataset

#### Chạy 3 strategies trên CIFAR-10
```bash
python src/train.py -m data=cifar10 experiment=al_random,al_entropy,al_cp_size
```

#### Chạy tất cả 8 strategies trên CIFAR-100
```bash
python src/train.py -m \
  data=cifar100 \
  model=resnet18_cifar100 \
  experiment=al_random,al_entropy,al_least_confidence,al_margin,al_cp_size,al_cp_v_shaped,al_combined,al_combined_v_shaped
```

#### Chạy CP strategies trên STL-10
```bash
python src/train.py -m \
  data=stl10 \
  model=resnet18_stl10 \
  experiment=al_cp_size,al_cp_v_shaped,al_combined,al_combined_v_shaped
```

---

### 3. Chạy 1 Strategy trên Nhiều Datasets

#### Entropy strategy trên tất cả datasets
```bash
# CIFAR-10
python src/train.py data=cifar10 experiment=al_entropy

# CIFAR-100
python src/train.py data=cifar100 model=resnet18_cifar100 experiment=al_entropy

# STL-10
python src/train.py data=stl10 model=resnet18_stl10 experiment=al_entropy

# SVHN
python src/train.py data=svhn experiment=al_entropy
```

---

### 4. Thí Nghiệm Đầy Đủ: Tất Cả Datasets + Tất Cả Strategies

#### Script tự động
Tạo file `scripts/run_all_experiments.sh`:

```bash
#!/bin/bash

# CIFAR-10 - All strategies
python src/train.py -m data=cifar10 \
  experiment=al_random,al_entropy,al_least_confidence,al_margin,al_cp_size,al_cp_v_shaped,al_combined,al_combined_v_shaped

# CIFAR-100 - All strategies
python src/train.py -m data=cifar100 model=resnet18_cifar100 \
  experiment=al_random,al_entropy,al_least_confidence,al_margin,al_cp_size,al_cp_v_shaped,al_combined,al_combined_v_shaped

# STL-10 - All strategies
python src/train.py -m data=stl10 model=resnet18_stl10 \
  experiment=al_random,al_entropy,al_least_confidence,al_margin,al_cp_size,al_cp_v_shaped,al_combined,al_combined_v_shaped

# SVHN - All strategies
python src/train.py -m data=svhn \
  experiment=al_random,al_entropy,al_least_confidence,al_margin,al_cp_size,al_cp_v_shaped,al_combined,al_combined_v_shaped
```

Chạy:
```bash
bash scripts/run_all_experiments.sh
```

---

## ⚙️ Tùy Chỉnh Tham Số

### AL Settings

```bash
# Thay đổi số rounds và budget
python src/train.py \
  data=cifar10 \
  experiment=al_entropy \
  al.num_rounds=10 \
  al.budget_per_round=1000

# Thay đổi initial labeled size
python src/train.py \
  data=cifar10 \
  experiment=al_cp_size \
  data.al_splits.initial_labeled=3000 \
  data.al_splits.calibration_size=3000
```

### Training Settings

```bash
# Thay đổi epochs per round
python src/train.py \
  data=cifar10 \
  experiment=al_random \
  trainer.max_epochs=15

# Thay đổi batch size và learning rate
python src/train.py \
  data=cifar100 \
  model=resnet18_cifar100 \
  experiment=al_entropy \
  data.batch_size=256 \
  model.optimizer.lr=0.02
```

### Conformal Prediction Settings

```bash
# Thay đổi CP alpha (coverage level)
python src/train.py \
  data=cifar10 \
  experiment=al_cp_size \
  conformal.alpha=0.05  # 95% coverage instead of 90%

# Thay đổi weights cho combined strategies
python src/train.py \
  data=svhn \
  experiment=al_combined \
  combined_weights.entropy_weight=0.7 \
  combined_weights.cp_weight=0.3
```

### Device và Performance

```bash
# Chạy trên CPU
python src/train.py \
  data=cifar10 \
  experiment=al_random \
  trainer.accelerator=cpu

# Sử dụng GPU cụ thể
python src/train.py \
  data=cifar10 \
  experiment=al_entropy \
  trainer.devices=[0]

# Fast dev run (test nhanh)
python src/train.py \
  data=cifar10 \
  experiment=al_random \
  trainer.fast_dev_run=true
```

### Random Seed

```bash
# Chạy với seed khác nhau
python src/train.py data=cifar10 experiment=al_entropy seed=42
python src/train.py data=cifar10 experiment=al_entropy seed=123
python src/train.py data=cifar10 experiment=al_entropy seed=456
```

---

## 📁 Kết Quả

### Vị trí lưu kết quả
```
logs/runs/YYYY-MM-DD/HH-MM-SS/
├── results_{strategy}.json      # Metrics per round
├── checkpoints/                 # Model checkpoints
└── tensorboard/                 # TensorBoard logs
```

### Xem kết quả TensorBoard
```bash
tensorboard --logdir logs/runs
```

### Load và phân tích kết quả
```python
import json

# Load results
with open('logs/runs/.../results_al_entropy.json', 'r') as f:
    results = json.load(f)

# Print metrics
print(f"Final test accuracy: {results['test_acc'][-1]:.2f}%")
print(f"Final CP coverage: {results['cp_coverage'][-1]:.4f}")
```

---

## 🧪 Quick Test (Kiểm Tra Nhanh)

### Test 1 round với budget nhỏ
```bash
# CIFAR-10
python src/train.py \
  data=cifar10 \
  experiment=al_random \
  al.num_rounds=1 \
  al.budget_per_round=100 \
  trainer.max_epochs=2

# CIFAR-100
python src/train.py \
  data=cifar100 \
  model=resnet18_cifar100 \
  experiment=al_entropy \
  al.num_rounds=1 \
  al.budget_per_round=100 \
  trainer.max_epochs=2

# STL-10
python src/train.py \
  data=stl10 \
  model=resnet18_stl10 \
  experiment=al_cp_size \
  al.num_rounds=1 \
  al.budget_per_round=50 \
  trainer.max_epochs=2

# SVHN
python src/train.py \
  data=svhn \
  experiment=al_combined \
  al.num_rounds=1 \
  al.budget_per_round=200 \
  trainer.max_epochs=2
```

---

## 📝 Template Commands (Copy & Paste)

### CIFAR-10 Experiments
```bash
# Single strategy
python src/train.py data=cifar10 experiment=al_<STRATEGY>

# Multiple strategies
python src/train.py -m data=cifar10 experiment=al_random,al_entropy,al_cp_size

# Custom settings
python src/train.py data=cifar10 experiment=al_entropy \
  al.num_rounds=20 al.budget_per_round=2000 trainer.max_epochs=10
```

### CIFAR-100 Experiments
```bash
# Single strategy
python src/train.py data=cifar100 model=resnet18_cifar100 experiment=al_<STRATEGY>

# Multiple strategies
python src/train.py -m data=cifar100 model=resnet18_cifar100 \
  experiment=al_random,al_entropy,al_cp_v_shaped

# Custom settings
python src/train.py data=cifar100 model=resnet18_cifar100 experiment=al_combined \
  al.num_rounds=15 al.budget_per_round=3000
```

### STL-10 Experiments
```bash
# Single strategy
python src/train.py data=stl10 model=resnet18_stl10 experiment=al_<STRATEGY>

# Multiple strategies  
python src/train.py -m data=stl10 model=resnet18_stl10 \
  experiment=al_random,al_entropy,al_margin

# Custom settings (smaller dataset, adjust accordingly)
python src/train.py data=stl10 model=resnet18_stl10 experiment=al_cp_size \
  al.num_rounds=3 al.budget_per_round=300 data.batch_size=32
```

### SVHN Experiments
```bash
# Single strategy
python src/train.py data=svhn experiment=al_<STRATEGY>

# Multiple strategies
python src/train.py -m data=svhn \
  experiment=al_random,al_entropy,al_combined_v_shaped

# Custom settings (larger dataset)
python src/train.py data=svhn experiment=al_entropy \
  al.num_rounds=10 al.budget_per_round=5000
```

---

## 🔍 Xem Config Trước Khi Chạy

```bash
# Xem full configuration
python src/train.py data=cifar10 experiment=al_entropy --cfg job

# Chỉ xem những config bị override
python src/train.py data=stl10 model=resnet18_stl10 experiment=al_cp_size \
  --cfg job --resolve
```

---

## 💾 Evaluation (Đánh Giá Model Đã Train)

```bash
# Evaluate checkpoint
python src/eval.py \
  ckpt_path=logs/runs/2024-01-30/12-00-00/checkpoints/epoch_009.ckpt \
  data=cifar10
```

---

## 🎯 Recommended Experiments (Thí Nghiệm Đề Xuất)

### Baseline Comparison
So sánh Random vs Entropy vs CP strategies:
```bash
python src/train.py -m \
  data=cifar10 \
  experiment=al_random,al_entropy,al_cp_size,al_cp_v_shaped
```

### CP Strategy Comparison
So sánh CP variants:
```bash
python src/train.py -m \
  data=cifar100 \
  model=resnet18_cifar100 \
  experiment=al_cp_size,al_cp_v_shaped,al_combined,al_combined_v_shaped
```

### Cross-Dataset Study
Chạy cùng 1 strategy trên nhiều datasets để so sánh:
```bash
# Entropy across all datasets
python src/train.py data=cifar10 experiment=al_entropy
python src/train.py data=cifar100 model=resnet18_cifar100 experiment=al_entropy
python src/train.py data=stl10 model=resnet18_stl10 experiment=al_entropy
python src/train.py data=svhn experiment=al_entropy
```

---

## ⚠️ Lưu Ý Quan Trọng

1. **Model Config**: 
   - CIFAR-100 cần `model=resnet18_cifar100` (100 classes)
   - STL-10 cần `model=resnet18_stl10` (96x96 images)
   - CIFAR-10, SVHN dùng `model=resnet18` mặc định

2. **Dataset Size**:
   - STL-10 chỉ có 5000 training images → giảm budget và rounds
   - SVHN có 73257 training images → có thể tăng budget

3. **Memory**:
   - STL-10 (96x96) tốn nhiều memory hơn → giảm batch_size nếu cần
   - Recommended batch_size: CIFAR 128, STL-10 64

4. **CP Strategies**:
   - Cần calibration set → đảm bảo `data.al_splits.calibration_size` đủ lớn
   - Default alpha=0.1 (90% coverage)

---

## 📞 Support

Nếu gặp lỗi, check:
1. Config file: `python src/train.py --cfg job`
2. Hydra help: `python src/train.py --help`
3. Dataset downloaded: Check `data/` directory

Happy experimenting! 🚀
