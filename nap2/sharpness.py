"""Sharpness features (F1-F4) from Kalra et al. 2025 (ICLR).

Computes global-scalar-per-snapshot features that capture loss landscape
curvature, complementing NAP2's per-layer gradient statistics.

Features:
    F1 (weight_norm): L2 norm of all model parameters
    F2 (top_hessian_eigenvalue): Largest eigenvalue of loss Hessian via power iteration
    F3 (stability_ratio): eta*lambda_H/2 -- determines edge-of-stability regime
    F4 (sharpness_change_rate): delta_lambda_H/delta_t between consecutive snapshots
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


def compute_weight_norm(model: nn.Module) -> float:
    """Compute F1: L2 norm of all model parameters.

    Args:
        model: PyTorch model.

    Returns:
        Scalar L2 norm of concatenated parameter vector.
    """
    params = [p.data.flatten() for p in model.parameters()]
    return torch.norm(torch.cat(params)).item()


def compute_top_hessian_eigenvalue(
    model: nn.Module,
    loss_fn: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    max_iterations: int = 20,
    tol: float = 1e-3,
) -> Tuple[float, bool]:
    """Compute F2: top eigenvalue of the loss Hessian via power iteration.

    Uses torch.autograd.functional.hvp for Hessian-vector products.
    Model is temporarily set to eval mode to avoid BatchNorm issues.

    Args:
        model: PyTorch model (parameters will NOT be modified).
        loss_fn: Loss function (e.g., nn.CrossEntropyLoss()).
        inputs: Input batch tensor.
        targets: Target batch tensor.
        max_iterations: Maximum power iteration steps.
        tol: Convergence tolerance (relative eigenvalue change).

    Returns:
        (eigenvalue, converged): Top eigenvalue and whether iteration converged.
        Returns (0.0, True) if Hessian is zero (e.g., at a flat region).
        Returns (nan, False) if computation fails.
    """
    was_training = model.training
    model.eval()

    try:
        # Collect parameter names and values for functional_call
        param_names = [n for n, p in model.named_parameters() if p.requires_grad]
        param_values = tuple(
            p.data.clone().requires_grad_(True)
            for p in model.parameters() if p.requires_grad
        )

        # Define loss as a pure function of parameters (no in-place mutation)
        def loss_closure(*pvals):
            param_dict = dict(zip(param_names, pvals))
            outputs = torch.func.functional_call(model, param_dict, inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[-1]
            return loss_fn(outputs, targets)

        # Initialize random vector (same shape as params tuple)
        v = tuple(torch.randn_like(p) for p in param_values)
        v_norm = torch.sqrt(sum(torch.sum(vi ** 2) for vi in v))
        v = tuple(vi / v_norm for vi in v)

        eigenvalue = 0.0
        converged = False

        for iteration in range(max_iterations):
            # Compute Hessian-vector product
            _, hvp = torch.autograd.functional.hvp(
                loss_closure, param_values, v,
                create_graph=False, strict=False,
            )

            # Compute eigenvalue estimate (Rayleigh quotient: v^T H v)
            eigenvalue_new = sum(
                torch.sum(vi * hvi) for vi, hvi in zip(v, hvp)
            ).item()

            # Compute norm of Hv
            hvp_norm = torch.sqrt(sum(torch.sum(hvi ** 2) for hvi in hvp)).item()

            if hvp_norm < 1e-10:
                eigenvalue = 0.0
                converged = True
                break

            # Update eigenvector
            v = tuple(hvi / hvp_norm for hvi in hvp)

            # Check convergence
            rel_change = abs(eigenvalue_new - eigenvalue) / (abs(eigenvalue) + 1e-10)
            eigenvalue = eigenvalue_new

            if iteration > 0 and rel_change < tol:
                converged = True
                break

        return eigenvalue, converged

    except Exception as e:
        if was_training:
            model.train()
        import warnings
        warnings.warn(f"Hessian eigenvalue computation failed: {e}")
        return float('nan'), False

    finally:
        if was_training:
            model.train()


def compute_sharpness_features(
    model: nn.Module,
    loss_fn: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    learning_rate: float,
    prev_lambda_h: Optional[float],
    snapshot_interval: int,
    max_iterations: int = 20,
    tol: float = 1e-3,
) -> Dict[str, float]:
    """Compute all four sharpness features (F1-F4).

    Args:
        model: PyTorch model.
        loss_fn: Loss function.
        inputs: Input batch.
        targets: Target batch.
        learning_rate: Current learning rate eta.
        prev_lambda_h: lambda_H from previous snapshot (None if first snapshot).
        snapshot_interval: Mini-batches between snapshots.
        max_iterations: Max power iteration steps for F2.
        tol: Convergence tolerance for F2.

    Returns:
        Dict with keys "F1", "F2", "F3", "F4", "F2_converged".
    """
    f1 = compute_weight_norm(model)
    f2, converged = compute_top_hessian_eigenvalue(
        model, loss_fn, inputs, targets,
        max_iterations=max_iterations, tol=tol,
    )
    f3 = learning_rate * f2 / 2.0
    f4 = 0.0 if prev_lambda_h is None else (f2 - prev_lambda_h) / snapshot_interval

    return {
        "F1": f1,
        "F2": f2,
        "F3": f3,
        "F4": f4,
        "F2_converged": converged,
    }
