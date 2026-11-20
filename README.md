# Acoustic Metamaterial Design with Conditional Diffusion

A conditional diffusion model for proposing geometries of ultraslim sound absorbing cells from target absorption spectra. Candidate designs are re-simulated with the included physics model and ranked against the target.

## Project layout

- `notebooks/` — research and inference notebooks. Start with `mnc_demo.ipynb` to use a trained model, or `MNCC_diffusion_pipeline.ipynb` to inspect the end-to-end pipeline.
- `src/mnc_inference.py` — reusable physics, model loading, design generation, and plotting code.
- `docs/how_to_use_this_model.md` — detailed model usage and verification guide.
- `assets/figures/` — exported figures from exploratory runs.
- `mnc_artifacts/` — generated checkpoint and evaluation artifacts (created by the pipeline; not included here).

## Quick start

Install Python 3.9+ with `numpy`, `scipy`, `matplotlib`, and `torch`. For the interactive demo, install `ipywidgets` too. Run `notebooks/mnc_demo.ipynb` after placing the trained `mnc_artifacts/` directory in the project root. The demo notebook finds `src/mnc_inference.py` automatically when run from this project.

See [the model guide](docs/how_to_use_this_model.md) for setup details, inputs, outputs, and verification steps.
