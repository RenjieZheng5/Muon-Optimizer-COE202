# Muon Optimizer Small-Scale Study

This repository contains a reproducible PyTorch experiment framework for the
COE202 final project on AdamW, Muon, and Newton-Schulz iterations.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The current scripts also support `--dataset fake` for a fast smoke test without
downloading data. MNIST and Fashion-MNIST are downloaded as IDX files by the
project's built-in loader, so `torchvision` is optional.

## Smoke Test

```powershell
python train.py --dataset fake --model mlp --optimizer adamw --epochs 1 --seeds 0 --subset-train 256 --subset-val 128 --output-dir outputs/smoke
python train.py --dataset fake --model mlp --optimizer muon --epochs 1 --seeds 0 --subset-train 256 --subset-val 128 --output-dir outputs/smoke
python analyze.py --results outputs/smoke/metrics.csv --output-dir outputs/smoke
```

## Minimal Experiment

```powershell
python run_experiments.py --mode minimal --output-dir outputs/minimal --epochs 10
python analyze.py --results outputs/minimal/metrics.csv --output-dir outputs/minimal
```

This runs MNIST + MLP with AdamW and Muon over seeds `0,1,2`.

## Main Experiment

```powershell
python run_experiments.py --mode main --output-dir outputs/main --epochs 15
python analyze.py --results outputs/main/metrics.csv --output-dir outputs/main
```

This runs Fashion-MNIST + MLP with AdamW and Muon over seeds `0,1,2,3,4`.

## Newton-Schulz Ablation

```powershell
python run_experiments.py --mode ablation --output-dir outputs/ablation --epochs 15
python analyze.py --results outputs/ablation/metrics.csv --output-dir outputs/ablation
```

This runs Fashion-MNIST + MLP with Muon using `q = 1, 2, 3, 5` and both
coefficient choices from the project proposal.

## Hyperparameter Search

```powershell
python run_experiments.py --mode search --dataset fashion_mnist --model mlp --output-dir outputs/search --epochs 8
python analyze.py --results outputs/search/metrics.csv --output-dir outputs/search
```

The search uses the same number of learning-rate and weight-decay candidates
for AdamW and Muon, reducing tuning-budget bias.

## Outputs

Every run appends rows to `metrics.csv` with:

```text
run_id,dataset,model,optimizer,lr,weight_decay,seed,epoch,train_loss,val_loss,val_acc,epoch_time,peak_memory,ns_steps,ns_coeffs
```

`analyze.py` creates:

- `summary.csv`
- `loss_curve.png`
- `accuracy_curve.png`
- `time_to_threshold.csv`
