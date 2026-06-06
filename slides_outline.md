# Slide Outline: Muon Optimizer Small-Scale Study

## Slide 1: Title

A Small-Scale Experimental Study of the Muon Optimizer and Newton-Schulz
Iterations for Neural Network Training

## Slide 2: Motivation

AdamW is stable and widely used. Muon is interesting because it replaces
coordinate-wise adaptive scaling with matrix-geometric updates for hidden
layers.

## Slide 3: Research Question

Can Muon converge faster or more stably than AdamW on small neural networks, and
how sensitive is it to Newton-Schulz iteration design?

## Slide 4: Muon Mathematical Idea

Show the pipeline:

```text
Gradient -> momentum matrix -> Newton-Schulz polar approximation -> update
```

## Slide 5: Experimental Setup

Datasets: MNIST and Fashion-MNIST.
Models: MLP and optional small CNN.
Metrics: loss, accuracy, time, stability, peak memory.

## Slide 6: AdamW vs Muon Results

Insert `outputs/main/loss_curve.png` and `outputs/main/accuracy_curve.png`.

## Slide 7: Newton-Schulz Ablation

Compare `q = 1, 2, 3, 5` and coefficient choices `muon` vs `standard`.

## Slide 8: Discussion

Explain whether Muon improves early loss, whether final accuracy changes, and
whether extra Newton-Schulz iterations are worth their cost.

## Slide 9: Conclusion

Summarize when Muon is useful, when AdamW remains preferable, and what the
small-scale results suggest about matrix-orthogonalized optimizer design.

## Slide 10: Future Work

Extend to larger CNNs, CIFAR-10, transformer blocks, or language-model settings.
