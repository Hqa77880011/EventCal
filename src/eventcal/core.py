import numpy as np
import torch
from scipy.optimize import linprog
from torch import nn
from torch.nn import functional as F


METHODS = ["eventcal", "no_inflation", "uniform_point", "temperature_point"]


def sharp_bounds(A, r, event, radius=0.0):
    A, r, event = np.asarray(A, float), np.asarray(r, float), np.asarray(event, float)
    if A.shape[1] != len(event) or A.shape[0] != len(r):
        raise ValueError("Observation operator dimensions disagree")
    if radius == 0:
        kwargs = {"A_eq": np.vstack([A, np.ones(len(event))]), "b_eq": np.r_[r, 1.0]}
    else:
        kwargs = {
            "A_ub": np.vstack([A, -A]),
            "b_ub": np.r_[r + radius, -r + radius],
            "A_eq": np.ones((1, len(event))),
            "b_eq": [1.0],
        }
    lower = linprog(event, bounds=[(0, 1)] * len(event), method="highs", **kwargs)
    upper = linprog(-event, bounds=[(0, 1)] * len(event), method="highs", **kwargs)
    if not lower.success or not upper.success:
        raise ValueError("Empty compatible set; check A, r and model-error radius")
    augmented = np.vstack([A, np.ones(len(event))])
    projector = np.linalg.pinv(augmented) @ augmented
    residual = np.linalg.norm(event - projector @ event)
    return float(lower.fun), float(-upper.fun), float(residual)


class Model(nn.Module):
    def __init__(self, cfg, method):
        super().__init__()
        self.method = method
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cfg["input_dim"], cfg["hidden"]),
            nn.GELU(),
            nn.Linear(cfg["hidden"], len(cfg["operator"])),
        )

    def forward(self, x):
        return self.network(x.float())


def loss(model, batch, cfg):
    return F.cross_entropy(model(batch["x"]), batch["y"].long())


def calibrate(model, validation, cfg):
    device = next(model.parameters()).device
    x = torch.as_tensor(validation.data["x"], device=device)
    with torch.no_grad():
        r = model(x).softmax(-1).cpu().numpy()
    if "observable_truth" in validation.data:
        residual = np.max(np.abs(r - validation.data["observable_truth"]), axis=1)
        interpretation = "observable-law error"
    else:
        target = np.eye(r.shape[1])[validation.data["y"].astype(int)]
        residual = np.max(np.abs(r - target), axis=1)
        interpretation = "marginal observable-outcome prediction"
    rank = int(np.ceil((len(residual) + 1) * (1 - cfg["delta"]))) - 1
    radius = float(np.sort(residual)[rank]) if rank < len(residual) else 1.0
    from scipy.optimize import minimize_scalar

    logits = model(x).detach().cpu().numpy()
    labels = validation.data["y"].astype(int)

    def objective(log_temperature):
        values = logits / np.exp(log_temperature)
        values -= values.max(-1, keepdims=True)
        normalizer = np.log(np.exp(values).sum(-1))
        return float(np.mean(normalizer - values[np.arange(len(labels)), labels]))

    temperature = float(np.exp(minimize_scalar(objective, bounds=(-3, 3), method="bounded").x))
    return {"radius": radius, "interpretation": interpretation, "temperature": temperature}


def predict(model, batch, cfg, calibration=None):
    logits = model(batch["x"])
    temperature = (
        (calibration or {}).get("temperature", 1.0) if model.method == "temperature_point" else 1.0
    )
    logits = logits / temperature
    r = logits.softmax(-1)
    bounds, structural, residuals = [], [], []
    A = np.asarray(cfg["operator"])
    radius = (
        0
        if model.method == "no_inflation"
        else (calibration or {}).get("radius", cfg.get("radius", 0))
    )
    for probability, event in zip(r.detach().cpu().numpy(), batch["event"].cpu().numpy()):
        low, high, residual = sharp_bounds(A, probability, event, 0)
        expanded_low, expanded_high, _ = sharp_bounds(A, probability, event, radius)
        if model.method in ("uniform_point", "temperature_point"):
            fine = np.linalg.lstsq(
                np.vstack([A, np.ones(A.shape[1])]), np.r_[probability, 1], rcond=None
            )[0]
            fine = np.maximum(fine, 0)
            fine /= fine.sum()
            expanded_low = expanded_high = float(event @ fine)
        bounds.append([expanded_low, expanded_high])
        structural.append([low, high])
        residuals.append([residual])
    return {
        "logits": logits,
        "bounds": torch.tensor(bounds, device=logits.device),
        "structural": torch.tensor(structural, device=logits.device),
        "row_space_residual": torch.tensor(residuals, device=logits.device),
    }


def metrics(prediction, data, cfg):
    structural = prediction["structural"]
    report = prediction["bounds"]
    width = structural[:, 1] - structural[:, 0]
    reported = report[:, 1] - report[:, 0]
    logits = np.asarray(prediction["logits"], dtype=float)
    probability = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probability /= probability.sum(axis=-1, keepdims=True)
    labels = np.asarray(data["y"], dtype=int)
    confidence = probability.max(axis=-1)
    correct = probability.argmax(axis=-1) == labels
    ece = 0.0
    edges = np.linspace(0, 1, cfg.get("calibration_bins", 15) + 1)
    for low, high in zip(edges[:-1], edges[1:]):
        selected = (confidence > low) & (confidence <= high)
        if selected.any():
            ece += selected.mean() * abs(correct[selected].mean() - confidence[selected].mean())
    result = {
        "observable_ece": float(ece),
        "observable_accuracy": float(correct.mean()),
        "observable_nll": float(
            -np.log(probability[np.arange(len(labels)), labels].clip(1e-12)).mean()
        ),
        "mean_structural_width": float(width.mean()),
        "mean_reported_width": float(reported.mean()),
        "false_certainty_rate": float(
            np.mean(
                (width > cfg["width_threshold"]) & (reported <= cfg["certainty_fraction"] * width)
            )
        ),
        "point_fraction": float(np.mean(reported <= cfg["identification_tolerance"])),
        "excess_width": float(np.mean(reported - width)),
    }
    if "event_probability" in data:
        true = data["event_probability"]
        result["event_coverage"] = float(np.mean((report[:, 0] <= true) & (true <= report[:, 1])))
    if "observable_truth" in data:
        true_bounds = [
            sharp_bounds(cfg["operator"], r, e)[:2]
            for r, e in zip(data["observable_truth"], data["event"])
        ]
        result["identified_set_coverage"] = float(
            np.mean(
                [
                    report[i, 0] <= low + 1e-7 and report[i, 1] >= high - 1e-7
                    for i, (low, high) in enumerate(true_bounds)
                ]
            )
        )
    return result


def inference(model, batch, cfg):
    return model(batch["x"])
