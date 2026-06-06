from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

import torch
from torch import Tensor


MUON_COEFFS = (3.4445, -4.7750, 2.0315)
STANDARD_COEFFS = (15.0 / 8.0, -5.0 / 4.0, 3.0 / 8.0)


def parse_coeffs(value: str | Sequence[float]) -> tuple[float, float, float]:
    if isinstance(value, str):
        named = {
            "muon": MUON_COEFFS,
            "standard": STANDARD_COEFFS,
        }
        lowered = value.lower()
        if lowered in named:
            return named[lowered]
        parts = [float(x.strip()) for x in value.split(",")]
    else:
        parts = [float(x) for x in value]
    if len(parts) != 3:
        raise ValueError("Newton-Schulz coefficients must contain exactly 3 values.")
    return (parts[0], parts[1], parts[2])


@torch.no_grad()
def zeropower_newton_schulz(
    grad: Tensor,
    steps: int = 3,
    coeffs: tuple[float, float, float] = MUON_COEFFS,
    eps: float = 1e-7,
) -> Tensor:
    """Approximate the polar factor of a matrix with Newton-Schulz iterations.

    The implementation follows the practical Muon pattern: reshape convolutional
    kernels to a matrix before calling this function, normalize by Frobenius
    norm, iterate on the smaller side when helpful, then restore orientation.
    """

    if grad.ndim != 2:
        raise ValueError("zeropower_newton_schulz expects a 2D tensor.")
    if steps < 1:
        return grad

    original_dtype = grad.dtype
    x = grad.float()
    norm = x.norm()
    if not torch.isfinite(norm) or norm < eps:
        return torch.zeros_like(grad)
    x = x / (norm + eps)

    transposed = False
    if x.size(0) > x.size(1):
        x = x.T
        transposed = True

    a, b, c = coeffs
    for _ in range(steps):
        xx_t = x @ x.T
        x = a * x + (b * xx_t + c * (xx_t @ xx_t)) @ x

    if transposed:
        x = x.T
    return x.to(dtype=original_dtype)


class Muon(torch.optim.Optimizer):
    """Muon optimizer for matrix-valued hidden parameters.

    This optimizer intentionally handles only tensors with at least two
    dimensions. Biases, normalization parameters, and classifier heads should be
    passed to a separate optimizer such as AdamW.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-2,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        ns_steps: int = 3,
        ns_coeffs: tuple[float, float, float] = MUON_COEFFS,
    ) -> None:
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            ns_coeffs=ns_coeffs,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            ns_steps = group["ns_steps"]
            ns_coeffs = group["ns_coeffs"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim < 2:
                    raise ValueError("Muon received a non-matrix parameter.")

                grad = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad, alpha=1.0 - momentum)

                matrix = buf.reshape(buf.shape[0], -1)
                update = zeropower_newton_schulz(matrix, ns_steps, ns_coeffs)
                update = update.reshape_as(p)
                scale = math.sqrt(max(1.0, matrix.shape[0] / matrix.shape[1]))

                if weight_decay != 0:
                    p.mul_(1.0 - lr * weight_decay)
                p.add_(update, alpha=-lr * scale)

        return loss


class OptimizerBundle:
    """Small wrapper that lets Muon and AdamW act as one optimizer."""

    def __init__(self, *optimizers: Optional[torch.optim.Optimizer]) -> None:
        self.optimizers = [opt for opt in optimizers if opt is not None]

    def zero_grad(self, set_to_none: bool = True) -> None:
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for opt in self.optimizers:
            opt.step()


def is_muon_parameter(name: str, param: Tensor) -> bool:
    if not param.requires_grad or param.ndim < 2:
        return False
    lowered = name.lower()
    excluded = ("classifier", "head", "output", "logits")
    return not any(token in lowered for token in excluded)


def split_muon_parameters(model: torch.nn.Module) -> tuple[list[Tensor], list[Tensor], list[str]]:
    muon_params: list[Tensor] = []
    aux_params: list[Tensor] = []
    muon_names: list[str] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if is_muon_parameter(name, param):
            muon_params.append(param)
            muon_names.append(name)
        else:
            aux_params.append(param)
    return muon_params, aux_params, muon_names
