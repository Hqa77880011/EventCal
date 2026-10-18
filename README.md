# EventCal

Implementation of **Event identification and calibration under coarsened supervision**.

Trains an observable-label classifier and computes sharp event bounds by compatible-simplex LPs. Structural width is separated from calibration inflation and row-space diagnostics.

## Install

Use Python 3.11 or newer. Install a matching PyTorch/TorchVision build for the target CPU or CUDA device, then install the package.

```bash
git clone https://github.com/Hqa77880011/EventCal.git
cd EventCal
python -m venv .venv
```

Activate with `source .venv/bin/activate` or `.venv\Scripts\Activate.ps1` in PowerShell.

```bash
python -m pip install -e .
```

## Prepare data

Use CIFAR-100 coarse labels or a known linear coarsening task. Register the observation operator before fitting the classifier.

x is image/features; y is the observed coarse label; event is a [N,K] fine-event vector. Optional observable_truth provides the controlled true observable law, and event_probability supplies evaluator target probabilities. Configure a column-stochastic operator [C,K].

```bash
python -m eventcal prepare --source data/raw.npz --output data/prepared --seed 17
```

Use `group` to keep every derived view of one source object in the same partition. An explicit integer `split` array (0=train, 1=validation, 2=test) preserves a registered split. Otherwise groups are assigned 60/20/20. Every stored field must share the sample axis. Files are loaded without pickle.

## Train

Edit `configs/default.json` for the input dimensions, device and training budget.

```bash
python -m eventcal train --config configs/default.json --data data/prepared --output runs/main
python -m eventcal suite --config configs/default.json --data data/prepared --output runs/comparison
```

`--methods` selects a subset. eventcal, no_inflation, uniform_point and temperature_point.

The common driver uses AdamW with cosine learning-rate decay, validation-objective early stopping and a fixed seed. Custom federated/continual drivers use their registered round/task budgets. It saves only a validation-selected model and measured epoch history. `learning_rate`, `weight_decay`, `batch_size`, `epochs`, `patience` and `device` control optimization. Use separate output roots for independent runs.

## Evaluate and visualize

```bash
python -m eventcal evaluate --checkpoint runs/main/eventcal/model.pt --data data/prepared/test.npz --output results/eventcal
python -m eventcal plot --inputs runs/main/eventcal results/eventcal --output figures
```

`model.pt` contains model parameters, configuration and method identity. `history.jsonl` contains measured training/validation loss and epoch time. Evaluation writes `predictions.npz` and `metrics.json`. Plotting creates loss curves and metric comparisons from supplied run directories.

Observable-label ECE, accuracy and NLL; structural/reported width, point fraction, false certainty, excess width, optional event/identified-set coverage and per-query timing. Observable ECE uses equal-width confidence bins (15 by default, controlled by `calibration_bins`); it scores coarse labels without assuming identified fine-event probabilities.

## Implementation conventions

Without observable-law annotations, calibration is explicitly a marginal observable-outcome prediction radius. It must not be interpreted as a distribution-free conditional-law confidence set. With controlled observable_truth, held-out model-error residuals calibrate the radius. Infeasible LPs report an input/model incompatibility rather than a point estimate.


## Pack image records

A JSONL manifest can point to images, masks and NPY feature arrays. Paths are relative to the manifest. All rows use identical keys; group stores the source object ID. For example, an image-classification row is `{"x":"images/1.png","y":0,"group":"object-1"}`.

```bash
python -m eventcal.pack --manifest data/manifest.jsonl --output data/raw.npz --channels 3 --target class
```

## CIFAR-100 preparation

```bash
python -m eventcal.cifar --output data/raw.npz --labels coarse --operator-config configs/operator.json
```

This downloads the author dataset through TorchVision, reserves 10% of its training images for validation, and preserves the original test set. It writes fine/coarse labels, source IDs and task metadata. Paired/audited transformation fields and ontology constraints must be added from the registered pair/ontology protocol before using those objectives.

Copy the generated 20x100 operator into the operator field of your training configuration; the default configuration is a four-leaf controlled example.

Inference timing runs the deployment forward pass separately from calibration, commutation and other evaluator diagnostics, after one warmup. Batch P50/P95 and examples/s retain the configured batch size. MAC/FLOP counts cover Conv/Linear/GRU multiply-adds; process RSS includes the dataset and Python runtime.
