# EventCal

Implementation of **Event identification and calibration under coarsened supervision**.

## Installation

Use Python `>=3.11`. Run commands from the repository root. Create the environment, activate it with `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\Activate.ps1` (PowerShell), then install the package.

```bash
python -m venv .venv
```

```bash
python -m pip install -e .
```

## Data preparation

The adapter downloads [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html), reserves validation images, preserves the test set and exports coarse observations plus leaf-event queries. Merge its generated 20×100 observation operator into the training config before training.

```bash
python -m eventcal.cifar --output data/raw.npz --labels coarse --operator-config data/operator.json --seed 17
python -c "import json; from pathlib import Path; c=json.loads(Path('configs/default.json').read_text()); c.update(json.loads(Path('data/operator.json').read_text())); Path('data/train_config.json').write_text(json.dumps(c, indent=2))"
python -m eventcal prepare --source data/raw.npz --output data/prepared --seed 17
```

## Training

Use the same `data/train_config.json` throughout. Its seed is `17`; preserve the prepared splits and use separate output directories for new seeds.

```bash
python -m eventcal train --config data/train_config.json --data data/prepared --output runs/main
```

## Evaluation

```bash
python -m eventcal evaluate --checkpoint runs/main/eventcal/model.pt --data data/prepared/test.npz --output results/eventcal
```

## Comparisons and ablations

`suite` trains the implemented methods on the same prepared split. Use `--methods` to select a subset, then evaluate each method checkpoint with the evaluation command above.

```bash
python -m eventcal suite --config data/train_config.json --data data/prepared --output runs/comparison
```

## Outputs

Each training method directory contains `model.pt` and `history.jsonl`. Evaluation writes `predictions.npz` and `metrics.json`; plots read the supplied run directories.

```bash
python -m eventcal plot --inputs runs/main/eventcal results/eventcal --output figures
```

## Verification

Source checks used Python 3.13.5. Installed dependency metadata: numpy 2.1.3, torch 2.7.1+cu118, torchvision 0.22.1, scipy 1.15.3, scikit-learn 1.6.1, matplotlib 3.10.0. Python AST, configuration files, CLI parser definitions and source input/output paths were checked without invoking command handlers. Training, full evaluation and CI were not run for this review.
