# How to use the trained model

This guide explains how to reuse the trained conditional diffusion model **without re-running the pipeline**, how to generate and inspect designs, and how to adapt the code later.

## 1. What the model does

Input: a target absorption spectrum alpha(f) on 200-1000 Hz (81 values, 10 Hz steps, each in [0, 1]).
Output: candidate geometries of an ultraslim absorber cell (41 mm x 3.5 mm x 50 mm), ranked by how well their *re-simulated* spectrum matches the target.

A design is 13 numbers in millimetres:

| Index | Name | Meaning |
|---|---|---|
| 0-5 | `w1 ... w6` | channel widths (sum is fixed at 37.5 mm) |
| 6-10 | `g1 ... g5` | gap at each turn of the serpentine (1.5-12 mm) |
| 11 | `d2` | embedded-tube inner diameter (1-3 mm) |
| 12 | `l0` | embedded-tube length (3-35 mm) |

The cell is a fixed six-channel serpentine; the model varies its dimensions (a parametric topology family, not free-form shapes).

## 2. Files

| File | Purpose | Produced by |
|---|---|---|
| `notebooks/MNCC_diffusion_pipeline.ipynb` | full research pipeline: physics, dataset, training, evaluation, saving | -- |
| `src/mnc_inference.py` | all definitions needed to load and use the model (physics, network, sampling, metrics, plots) | -- |
| `notebooks/mnc_demo.ipynb` | demo notebook (loads the model, generates, plots, interactive widgets) | -- |
| `mnc_artifacts/mnc_ddpm.pt` | network weights, standardisation, architecture, physics constants, training metadata | pipeline, Part 5/9 |
| `mnc_artifacts/demo_targets.npz` | 300 realistic reference spectra (`A`) and their original designs (`X`) | pipeline, Part 9 |
| `mnc_artifacts/results.json` | all evaluation numbers of the run (baselines, guidance sweep, robustness, novelty, design metrics) | pipeline, Part 9 |
| `mnc_dataset.npz` | synthetic training / test data (`Xtr, Atr, Xte, Ate, FREQ`) | pipeline, Part 3 |

Keep `src/mnc_inference.py` and the `mnc_artifacts/` folder together (a copy on Google Drive is created by the last pipeline cell).

## 3. Quick start (Colab or local)

Requirements: Python 3.9+, `numpy`, `scipy`, `matplotlib`, `torch` (CPU is sufficient; sampling 64 candidates takes a few seconds). For the widgets in the demo notebook also `ipywidgets` (preinstalled in Colab).

```python
import sys
sys.path.insert(0, "src")  # run from the project root
import mnc_inference as mnc

meta = mnc.load("mnc_artifacts")                      # validates the checkpoint, then restores the model
target = mnc.reference_spectrum(index=0)              # realistic target (0 = best absorber of the held-out set)

result = mnc.generate_designs(target, n=4, scale=1.0, n_candidates=64, seed=0)
mnc.gallery(result, k=4)                              # four geometries + spectra vs. target
fig, metrics = mnc.design_card(result.x[0], target=target)   # one design in detail, plus numbers
```

In Colab, upload the `src/` folder and `mnc_artifacts/` (or mount Drive and set `MNC_MODULE_DIR` to the uploaded `src/` folder), then run `notebooks/mnc_demo.ipynb`.

## 4. Using your own target spectrum

```python
import numpy as np
target = np.interp(mnc.FREQ, f_hz, alpha_values).astype(np.float32)   # resample to the 81-point grid
result = mnc.generate_designs(target, n=4, scale=1.0, n_candidates=128)
print(result.rmse)                                    # spectrum RMSE of each returned design
```

Guidelines for reliable results:

- Use targets that **look like spectra of this cell family**: one to three absorption peaks that arise from a folded quarter-wave-type resonance. Smooth hand-drawn shapes the family cannot produce (e.g. a flat plateau of 0.9 over 400-800 Hz) are matched only approximately; check `result.rmse`.
- For a measured or simulated spectrum from elsewhere, shifting it slightly in frequency is fine (`mnc.shift_spectrum(a, 0.05)` for +5 %); accuracy degrades beyond about +-6 %.
- Values must be absorption coefficients in [0, 1] (not dB).

## 5. Choosing the sampling settings

| Argument | Meaning | Recommended |
|---|---|---|
| `scale` | classifier-free guidance weight; 0 ignores the target, 1 = plain conditional sampling, larger values follow the target more strictly but reduce validity | 1.0 (best accuracy in the sweep); try 2.0 for sharper peak matching |
| `n_candidates` | number of designs sampled before ranking with the physics model | 64 for quick demos, 256 for hard targets (generate-and-filter) |
| `n` | number of best candidates returned | 4 |
| `seed` | makes sampling reproducible | any integer |

## 6. Reading the outputs

`generate_designs` returns a `Generated` object (best first):

- `x` (n, 13) designs in mm, `alpha` (n, 81) re-simulated spectra, `rmse` (n,) distance to the target, `target`.

`design_metrics(x)` / `design_card(x)` report:

- peak absorption and frequency, mean absorption over 400-800 Hz;
- width of the longest band with alpha >= 0.7 and alpha >= 0.9;
- lambda / thickness at the lower band edge (deep-subwavelength measure);
- paper-style figure of merit (Eq. 8 of Zeng et al. 2026), the acoustic path length and the folding factor (path length / cell height). The figure of merit is computed from model-generated spectra of a single cell, so it is not directly comparable with the paper's measured value (a five-cell assembly).

Helper functions: `mnc.render_mask(x)` returns the cross-section as a pixel image (0 air, 1 wall, 2 tube; 0.5 mm pixels), `mnc.alpha_batch(w, g, d2, l0)` simulates any designs directly, `mnc.repair(x)` projects a vector onto the feasible set.

## 7. Verifying a design

The returned spectra come from the physics model (transmission line with thermoviscous losses). Before relying on a design:

1. Re-simulate it independently: `mnc.alpha_one(w, g, d2, l0, mnc.FREQ)` (loop implementation).
2. Compare with the paper's behaviour (Part 4 of the pipeline notebook).
3. For the final shortlist, run an FEM simulation (e.g. COMSOL thermoviscous acoustics) or print and measure; the labels are model-generated, not measured.

## 8. Re-training or changing things

- Re-run `notebooks/MNCC_diffusion_pipeline.ipynb` from the top (GPU runtime recommended; training 60 epochs takes a few minutes on a GPU).
- Training settings: `train_ddpm(Xtr, Atr, epochs=60, bs=512, lr=1e-3)` after `init_diffusion(Xtr)`.
- Physics constants (`LSCALE`, `EPS`, `DEL`, cell dimensions) live at the top of `src/mnc_inference.py`. They are stored in the checkpoint and checked by `load()` together with the architecture (D=13, C=81, h=256, nb=5), the diffusion steps (200) and `mu`/`sd`. A mismatch raises `ValueError`, because the stored model would then be inconsistent with the simulator; pass `strict=False` to load with a warning instead.
- The checkpoint format is a plain PyTorch dictionary (`torch.load(..., weights_only=False)`), so the weights can be inspected or exported.

## 9. Model card (summary)

| Item | Value |
|---|---|
| Model | conditional DDPM, 200 steps, cosine schedule, classifier-free guidance (10 % condition dropout) |
| Network | residual MLP, 5 blocks, width 256, about 1.5 M parameters |
| Data | about 2.1 x 10^5 designs labelled by the analytical model (random plus evolutionary enrichment); 2 000 held-out random designs for testing |
| Conditioning | 81-point absorption spectrum, 200-1000 Hz |
| Evaluation | spectrum RMSE after re-simulation, diversity, novelty, comparison with regression / retrieval / random (see `results.json`) |
| Intended use | research and teaching demonstrations of generative design for acoustic absorbers |
| Not suitable for | direct manufacturing decisions without FEM / experimental verification; designs outside the 6-channel family; targets outside 200-1000 Hz |

Reference: Zeng K. et al., *3D printed broadband ultraslim sound-absorbing metamaterial via high dissipation*, Virtual and Physical Prototyping 21(1), e2669702 (2026).

## 10. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ModuleNotFoundError: mnc_inference` | the `src/` folder is not on the path: `sys.path.insert(0, "<folder>")` |
| `FileNotFoundError ... mnc_ddpm.pt` | wrong artifact path; pass the folder that contains the file to `mnc.load(...)` |
| `ValueError: physics constants ... differ` | `src/mnc_inference.py` was edited after training; restore the constants, retrain, or pass `strict=False` to load anyway |
| RMSE of all candidates is large (> 0.2) | target outside the family's reach; use a realistic target or raise `n_candidates` |
| `weights_only` error with `torch.load` | use `weights_only=False` (the file contains NumPy arrays) |
