# A Small-Scale Experimental Study of the Muon Optimizer and Newton-Schulz Iterations

## 1. Problem Setup

This project asks whether Muon can achieve faster or more stable convergence
than AdamW on small neural networks, and how sensitive Muon's behavior is to
the Newton-Schulz iteration design.

We study supervised image classification with cross-entropy loss. The primary
datasets are MNIST and Fashion-MNIST, using an MLP and optionally a small CNN.
The evaluation metrics are training loss, validation accuracy, wall-clock time,
time-to-threshold loss, stability across random seeds, and peak GPU memory when
available.

## 2. Technical Approach

AdamW is used as the adaptive baseline with decoupled weight decay. Muon is
used for matrix-valued hidden parameters only. Biases, normalization parameters,
and classifier-head parameters are handled separately with AdamW.

For a matrix parameter with momentum matrix M, Muon ideally uses the polar
direction Polar(M). Instead of computing an SVD at every step, the implementation
approximates this direction with q Newton-Schulz iterations:

```text
X_{j+1} = a X_j + (b X_j X_j^T + c (X_j X_j^T)^2) X_j.
```

The main Muon coefficients are `(3.4445, -4.7750, 2.0315)`. The ablation also
tests the standard polynomial coefficients `(15/8, -5/4, 3/8)`.

## 3. Experimental Setup

All comparisons use the same model architecture, data split, training budget,
batch size, and random seeds. AdamW and Muon receive the same hyperparameter
search budget before final multi-seed evaluation.

Recommended final configuration:

```text
Minimal: MNIST + MLP + AdamW vs Muon + seeds 0,1,2
Main: Fashion-MNIST + MLP + AdamW vs Muon + seeds 0,1,2,3,4
Ablation: Fashion-MNIST + MLP + Muon q in {1,2,3,5}
```

## 4. Main Results

Insert generated artifacts here:

- `outputs/main/loss_curve.png`
- `outputs/main/accuracy_curve.png`
- `outputs/main/summary.csv`
- `outputs/ablation/loss_curve.png`
- `outputs/ablation/accuracy_curve.png`

Report the mean and standard deviation over seeds. Discuss early loss
reduction, final validation accuracy, epoch time, and whether larger
Newton-Schulz iteration counts produce a clear benefit.

## 5. Conclusion

The final conclusion should follow the empirical results. A useful conclusion
does not need to claim that Muon always beats AdamW. A balanced result can state
that Muon improves early optimization in some settings, that AdamW remains a
strong and robust baseline, and that the Newton-Schulz iteration count controls
a practical trade-off between approximation quality and runtime.

## Limitations and Future Work

This project focuses on small networks and image classification. Future work
could extend the comparison to larger CNNs, transformer blocks, CIFAR-10, or
language-model training settings where matrix-valued hidden layers dominate the
parameter count.
