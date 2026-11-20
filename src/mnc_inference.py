"""
mnc_inference.py
================
Conditional diffusion model (DDPM) for designing ultraslim sound-absorbing cells.

Given a target absorption spectrum alpha(f) on 200-1000 Hz, the model proposes several
candidate cell geometries; a physics model (transmission line + thermoviscous losses,
after Zeng et al. 2026, Virtual Phys. Prototyp. 21:1, e2669702) re-simulates every
candidate so that designs can be ranked and verified.

Typical use
-----------
>>> import mnc_inference as mnc
>>> meta = mnc.load("mnc_artifacts")                 # restores the trained model
>>> target = mnc.reference_spectrum(index=0)         # a realistic target spectrum
>>> result = mnc.generate_designs(target, n=4, scale=1.0, n_candidates=64)
>>> mnc.design_card(result.x[0], target=target)

The same definitions are used by the research notebook (MNCC_diffusion_pipeline.ipynb);
the notebook's code cells are generated from the marked sections of this file.
"""

# %% [section:imports]
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
from scipy.special import jv

dev = "cuda" if torch.cuda.is_available() else "cpu"

# %% [section:constants]
# ---- Physical constants (air, 20 degC) and cell geometry (all lengths in mm unless noted) ----
rho0, c0 = 1.204, 343.0            # density [kg/m^3], speed of sound [m/s]
eta, kappa = 1.81e-5, 0.0258       # dynamic viscosity [Pa s], thermal conductivity [W/m/K]
Cp, gam, P0 = 1005.0, 1.4, 101325.0  # specific heat [J/kg/K], heat-capacity ratio, static pressure [Pa]
mm = 1e-3                          # mm -> m

W, T = 41.0, 3.5                   # cell width and thickness (the "ultraslim" direction)
TW, HIN = 0.5, 49.0                # wall thickness, interior height (cell height = HIN + 2*TW = 50 mm)
EPS, DEL = 3.0, 0.3                # tube end-correction constants (paper Table S3, MNCC-1)
LSCALE = 0.9                       # effective-path calibration, fitted on paper UCC-N2/N4/N6 (Table 1)
MTERMS = 15                        # odd modes kept in the rectangular-duct series (converged, 200-1000 Hz)

N_CH = 6                           # channels per cell
AVAIL = W - (N_CH + 1) * TW        # width shared by the channels [mm]
D = 13                             # design-vector length: 6 widths + 5 gaps + tube diameter + tube length

# %% [section:physics]
def psi_rect(w, b, om):
    """Viscous and thermal loss functions (psi_v, psi_h) of a rectangular duct.

    Parameters
    ----------
    w, b : float   duct cross-section [m]
    om   : ndarray angular frequencies [rad/s]

    psi -> 1 for wide ducts (lossless) and -> 0 for very narrow ones. Double series over odd
    modes (m, n) from the duct momentum / energy equations with no-slip, isothermal walls.
    """
    m = np.arange(1, 2 * MTERMS, 2)
    al = (m * np.pi / w)[:, None, None] ** 2
    be = (m * np.pi / b)[None, :, None] ** 2
    wgt = (64 / np.pi**4) / ((m[:, None, None] ** 2) * (m[None, :, None] ** 2))
    out = []
    for nu in (eta, kappa / Cp):
        den = nu * (al + be) + 1j * om[None, None, :] * rho0
        out.append(((1j * om * rho0)[None, None, :] * wgt / den).sum((0, 1)))
    return out


def psi_circ(r, om):
    """Loss functions (psi_v, psi_h) of a circular tube of radius r [m] (Stinson)."""
    out = []
    for nu in (eta, kappa / Cp):
        x = np.sqrt(-1j * om * rho0 / nu) * r
        out.append(1 - 2 * jv(1, x) / (x * jv(0, x)))
    return out


def duct(psv, psh, om, S):
    """Wavenumber k and characteristic acoustic impedance Zc of a lossy duct of area S [m^2]."""
    rho = rho0 / psv
    C = (gam - (gam - 1) * psh) / (gam * P0)
    return om * np.sqrt(rho * C), np.sqrt(rho / C) / S


def tl(Zc, k, L, ZL):
    """Input impedance of a duct (length L) terminated by load ZL; time convention exp(+j w t)."""
    t = np.tan(k * L)
    return Zc * (ZL + 1j * Zc * t) / (Zc + 1j * ZL * t)


def tl_rigid(Zc, k, L):
    """Input impedance of a duct closed by a rigid wall."""
    return -1j * Zc / np.tan(k * L)


def path_lengths(w, g, l0):
    """Effective acoustic length [mm] of every channel, including turn hops and LSCALE.

    w : N channel widths, g : N-1 turn gaps, l0 : tube length (the tube occupies channel 1).
    """
    N = len(w)
    hop = [0] + [(w[i] + w[i + 1]) / 2 + TW for i in range(N - 1)] + [0]
    L = [(HIN - (g[i] if i < N - 1 else g[-1])) + 0.5 * hop[i] + 0.5 * hop[i + 1] for i in range(N)]
    L[0] -= l0
    return [x * LSCALE for x in L]


def alpha_one(w, g, d2, l0, f):
    """Reference (loop) implementation for a single design.

    Returns the absorption coefficient alpha(f) and the relative surface impedance Zs(f).
    Steps: rigid end -> channels N..1 -> embedded tube -> end correction -> alpha = 1 - |R|^2.
    """
    om = 2 * np.pi * np.asarray(f, float)
    N = len(w)
    L = path_lengths(w, g, l0)
    Z = None
    for i in range(N - 1, -1, -1):
        pv, ph = psi_rect(w[i] * mm, T * mm, om)
        k, Zc = duct(pv, ph, om, w[i] * mm * T * mm)
        Z = tl_rigid(Zc, k, L[i] * mm) if Z is None else tl(Zc, k, L[i] * mm, Z)
    r = d2 / 2 * mm
    Sn = np.pi * r * r
    pv, ph = psi_circ(r, om)
    k, Zc = duct(pv, ph, om, Sn)
    Zn = tl(Zc, k, (l0 + TW) * mm, Z)
    Zend = (EPS * np.sqrt(2 * om * rho0 * eta) + 1j * om * rho0 * DEL * d2 * mm) / Sn
    Zs = (Zn + Zend) * (W * mm * T * mm) / (rho0 * c0)
    return 1 - np.abs((Zs - 1) / (Zs + 1)) ** 2, Zs

# %% [section:simulator]
# ---- Batched simulator: loss functions are tabulated once and linearly interpolated ----
FREQ = np.arange(200, 1001, 10.0)      # evaluation grid [Hz]
OM = 2 * np.pi * FREQ
NF = len(FREQ)
WG = np.arange(0.5, 40.001, 0.05)      # channel-width grid [mm]
DG = np.arange(0.8, 3.6, 0.02)         # tube-diameter grid [mm]


def build_tables():
    """Tabulate psi_v, psi_h for all grid widths / diameters (about 0.3 s)."""
    PVw = np.zeros((len(WG), NF), complex)
    PHw = PVw.copy()
    for i, w in enumerate(WG):
        PVw[i], PHw[i] = psi_rect(w * mm, T * mm, OM)
    PVd = np.zeros((len(DG), NF), complex)
    PHd = PVd.copy()
    for i, d in enumerate(DG):
        PVd[i], PHd[i] = psi_circ(d / 2 * mm, OM)
    return PVw, PHw, PVd, PHd


TAB = build_tables()


def lut(tab, grid, x):
    """Linear interpolation of a table of per-frequency rows at positions x."""
    p = (np.clip(x, grid[0], grid[-1] - 1e-9) - grid[0]) / (grid[1] - grid[0])
    i = np.floor(p).astype(int)
    t = (p - i)[:, None]
    return tab[i] * (1 - t) + tab[i + 1] * t


def alpha_batch(w, g, d2, l0):
    """Vectorised absorption spectra.

    w (B,N) widths, g (B,N-1) gaps, d2 (B,) tube diameter, l0 (B,) tube length  [mm]
    Returns alpha (B, NF) on the FREQ grid.
    """
    PVw, PHw, PVd, PHd = TAB
    B, N = w.shape
    om = OM[None, :]
    hop = np.concatenate([np.zeros((B, 1)), (w[:, :-1] + w[:, 1:]) / 2 + TW, np.zeros((B, 1))], 1)
    gg = np.concatenate([g, g[:, -1:]], 1)
    L = (HIN - gg) + 0.5 * hop[:, :-1] + 0.5 * hop[:, 1:]
    L[:, 0] -= l0
    L = L * LSCALE
    Z = None
    for i in range(N - 1, -1, -1):
        S = (w[:, i] * mm * T * mm)[:, None]
        k, Zc = duct(lut(PVw, WG, w[:, i]), lut(PHw, WG, w[:, i]), om, S)
        Lm = (L[:, i] * mm)[:, None]
        Z = tl_rigid(Zc, k, Lm) if Z is None else tl(Zc, k, Lm, Z)
    Sn = (np.pi * (d2 / 2 * mm) ** 2)[:, None]
    k, Zc = duct(lut(PVd, DG, d2), lut(PHd, DG, d2), om, Sn)
    Zn = tl(Zc, k, ((l0 + TW) * mm)[:, None], Z)
    Zend = (EPS * np.sqrt(2 * om * rho0 * eta) + 1j * om * rho0 * DEL * (d2 * mm)[:, None]) / Sn
    Zs = (Zn + Zend) * (W * mm * T * mm) / (rho0 * c0)
    return 1 - np.abs((Zs - 1) / (Zs + 1)) ** 2


def batched_alpha(X, bs=5000):
    """alpha_batch over a large (n, 13) design matrix in chunks."""
    return np.concatenate([alpha_batch(*unpack(X[i:i + bs])) for i in range(0, len(X), bs)])


def band_mean(a, lo=400, hi=800):
    """Mean absorption in [lo, hi] Hz for each row of a (n, NF)."""
    return a[:, (FREQ >= lo) & (FREQ <= hi)].mean(1)

# %% [section:design_space]
# ---- Design vector x = [w1..w6 | g1..g5 | d2 | l0]  (mm) ----
def pack(w, g, d2, l0):
    """Assemble design vectors (n, 13) from widths, gaps, tube diameter and length."""
    return np.concatenate([w, g, d2[:, None], l0[:, None]], 1)


def unpack(x):
    """Inverse of pack: returns (w, g, d2, l0)."""
    return x[:, :6], x[:, 6:11], x[:, 11], x[:, 12]


def valid(w, g, d2, l0):
    """Boolean mask of designs that satisfy the manufacturing / geometry constraints."""
    return ((w.min(1) >= 1.0) & (w[:, 0] >= d2 + 1.0) & (l0 <= HIN - g[:, 0] - 3) & (l0 >= 3)
            & (d2 >= 1.0) & (d2 <= 3.0) & (g.min(1) >= 1.5) & (g.max(1) <= 12))


def sample_designs(n, rng):
    """Draw n random valid designs (log-normal widths, uniform gaps / tube)."""
    out, have = [], 0
    while have < n:
        raw = np.exp(rng.normal(0, 0.9, (n, N_CH)))
        w = raw / raw.sum(1, keepdims=True) * AVAIL
        g = rng.uniform(1.5, 12, (n, N_CH - 1))
        d2 = rng.uniform(1.0, 3.0, n)
        l0 = rng.uniform(3, 35, n)
        ok = valid(w, g, d2, l0)
        out.append((w[ok], g[ok], d2[ok], l0[ok]))
        have += ok.sum()
    return [np.concatenate(z)[:n] for z in zip(*out)]


def repair(x):
    """Project arbitrary vectors onto the feasible design set (exact for widths, clipping otherwise)."""
    x = x.copy()
    w, g, d2, l0 = x[:, :6], x[:, 6:11], x[:, 11], x[:, 12]
    wmin = np.array([2.0, 1, 1, 1, 1, 1])              # w1 >= d2 + 1 >= 2 mm; others >= 1 mm
    ex = np.maximum(w - wmin, 1e-6)
    w = wmin + ex / ex.sum(1, keepdims=True) * (AVAIL - wmin.sum())   # sum fixed, minima kept
    g = np.clip(g, 1.5, 12)
    d2 = np.clip(d2, 1.0, 3.0)
    d2 = np.clip(np.minimum(d2, w[:, 0] - 1.0), 1.0, 3.0)
    l0 = np.clip(l0, 3, None)
    l0 = np.clip(np.minimum(l0, HIN - g[:, 0] - 3), 3, None)
    return pack(w, g, d2, l0)

# %% [section:diffusion]
# ---- Conditional DDPM: epsilon-prediction MLP, cosine schedule, classifier-free guidance ----
TSTEPS = 200
model = None                       # trained denoiser (set by train_ddpm / load)
mu = sd = None                     # design standardisation (set by init_diffusion / load)
betas = alphas = abar = None       # noise schedule tensors (set by init_diffusion / load)


def make_schedule(tsteps=TSTEPS, device=dev):
    """Cosine noise schedule (Nichol & Dhariwal): returns betas, alphas, cumulative alpha-bar."""
    s = 0.008
    tt = torch.arange(tsteps + 1) / tsteps
    ab = torch.cos((tt + s) / (1 + s) * math.pi / 2) ** 2
    ab = ab / ab[0]
    b = torch.clip(1 - ab[1:] / ab[:-1], 0, 0.999)
    a = 1 - b
    return b.to(device), a.to(device), torch.cumprod(a, 0).to(device)


def init_diffusion(Xtrain):
    """Set standardisation statistics from the training designs and build the noise schedule."""
    global mu, sd, betas, alphas, abar
    mu, sd = Xtrain.mean(0), Xtrain.std(0) + 1e-6
    betas, alphas, abar = make_schedule(TSTEPS, dev)


def tx(x):
    """Standardise designs -> tensor."""
    return torch.tensor((x - mu) / sd, dtype=torch.float32)


def tc(a):
    """Scale spectra (alpha in [0,1]) for conditioning -> tensor."""
    return torch.tensor((a - 0.5) / 0.3, dtype=torch.float32)


class SinEmb(nn.Module):
    """Sinusoidal embedding of the diffusion step."""

    def __init__(self, d):
        super().__init__()
        self.d = d

    def forward(self, t):
        h = self.d // 2
        fr = torch.exp(-math.log(10000) * torch.arange(h, device=t.device) / h)
        a = t[:, None].float() * fr[None]
        return torch.cat([a.sin(), a.cos()], 1)


class Block(nn.Module):
    """Pre-norm residual MLP block; the (step + spectrum) embedding is added to its input."""

    def __init__(self, h):
        super().__init__()
        self.n = nn.LayerNorm(h)
        self.f = nn.Sequential(nn.Linear(h, 2 * h), nn.SiLU(), nn.Linear(2 * h, h))

    def forward(self, x, e):
        return x + self.f(self.n(x + e))


class Denoiser(nn.Module):
    """epsilon_theta(x_t, t, spectrum): predicts the noise added to a design vector.

    `drop` marks samples whose spectrum is replaced by a learned null embedding
    (classifier-free guidance).
    """

    def __init__(self, D=D, C=NF, h=256, nb=5):
        super().__init__()
        self.inp = nn.Linear(D, h)
        self.temb = nn.Sequential(SinEmb(h), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h))
        self.cemb = nn.Sequential(nn.Linear(C, h), nn.SiLU(), nn.Linear(h, h))
        self.null = nn.Parameter(torch.zeros(h))
        self.blocks = nn.ModuleList([Block(h) for _ in range(nb)])
        self.out = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, D))

    def forward(self, x, t, c, drop=None):
        ce = self.cemb(c)
        if drop is not None:
            ce = torch.where(drop[:, None], self.null[None].expand_as(ce), ce)
        e = self.temb(t) + ce
        h = self.inp(x)
        for b in self.blocks:
            h = b(h, e)
        return self.out(h)


def train_ddpm(Xtrain, Atrain, epochs=60, bs=512, lr=1e-3, p_uncond=0.1, log_every=10, seed=0):
    """Train the denoiser (requires init_diffusion). Returns the list of epoch losses; sets `model`."""
    global model
    torch.manual_seed(seed)
    X_t, C_t = tx(Xtrain).to(dev), tc(Atrain).to(dev)
    model = Denoiser().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * (len(X_t) // bs))
    print("parameters: %.2fM | training designs: %d | device: %s"
          % (sum(p.numel() for p in model.parameters()) / 1e6, len(X_t), dev))
    losses, t0 = [], time.time()
    for ep in range(epochs):
        idx = torch.randperm(len(X_t), device=dev)
        run, nb = 0.0, 0
        for i in range(0, len(idx) - bs + 1, bs):
            b = idx[i:i + bs]
            x0, c = X_t[b], C_t[b]
            t = torch.randint(0, TSTEPS, (bs,), device=dev)
            eps = torch.randn_like(x0)
            xt = abar[t].sqrt()[:, None] * x0 + (1 - abar[t]).sqrt()[:, None] * eps
            drop = torch.rand(bs, device=dev) < p_uncond
            loss = F.mse_loss(model(xt, t, c, drop), eps)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run += loss.item()
            nb += 1
        losses.append(run / nb)
        if ep % log_every == 0 or ep == epochs - 1:
            print("epoch %3d  loss %.4f  (%.0fs)" % (ep, losses[-1], time.time() - t0))
    return losses

# %% [section:sampling]
@torch.no_grad()
def sample_x(cond, n_per=8, scale=1.0):
    """Generate designs for target spectra.

    cond  : (K, NF) target spectra, alpha in [0, 1]
    n_per : candidates per target
    scale : classifier-free guidance weight (0 = ignore target, 1 = conditional, >1 = stronger)
    Returns (K, n_per, 13) feasible designs [mm] (after `repair`).
    """
    model.eval()
    K = len(cond)
    c = tc(cond).to(dev).repeat_interleave(n_per, 0)
    B = len(c)
    x = torch.randn(B, D, device=dev)
    nodrop = torch.zeros(B, dtype=torch.bool, device=dev)
    alld = ~nodrop
    for t in reversed(range(TSTEPS)):
        tb = torch.full((B,), t, device=dev)
        eps_u = model(x, tb, c, alld)
        eps = eps_u + scale * (model(x, tb, c, nodrop) - eps_u)
        x = (x - betas[t] / (1 - abar[t]).sqrt() * eps) / alphas[t].sqrt()
        if t > 0:
            x = x + betas[t].sqrt() * torch.randn_like(x)
    x = x.cpu().numpy() * sd + mu
    return repair(x).reshape(K, n_per, D)


def evaluate(designs):
    """Re-simulate designs (K, n, 13) with the physics model -> alpha (K, n, NF)."""
    K, n, _ = designs.shape
    return alpha_batch(*unpack(designs.reshape(-1, D))).reshape(K, n, NF)


@dataclass
class Generated:
    """Result of `generate_designs`, sorted by spectrum RMSE (best first)."""
    x: np.ndarray        # (n, 13) designs [mm]
    alpha: np.ndarray    # (n, NF) re-simulated spectra
    rmse: np.ndarray     # (n,) RMSE to the target
    target: np.ndarray   # (NF,) target spectrum


def generate_designs(target, n=4, scale=1.0, n_candidates=None, seed=None):
    """Sample `n_candidates` designs for one target, re-simulate, and return the best `n`.

    Generate-and-filter: the physics model ranks the diffusion candidates by spectrum RMSE.
    """
    if seed is not None:
        torch.manual_seed(seed)
    target = np.asarray(target, np.float32)
    m = n_candidates or n
    d = sample_x(target[None], m, scale)[0]
    a = evaluate(d[None])[0]
    r = np.sqrt(((a - target[None]) ** 2).mean(1))
    o = np.argsort(r)[:n]
    return Generated(d[o], a[o], r[o], target)

# %% [section:metrics]
def rmse(a_gen, a_tgt):
    """RMSE between generated spectra (K, n, NF) and targets (K, NF) -> (K, n)."""
    return np.sqrt(((a_gen - a_tgt[:, None, :]) ** 2).mean(-1))


def diversity(d):
    """Mean pairwise distance between candidates in standardised design space; d: (K, n, 13)."""
    z = (d - mu) / sd
    K, n, _ = z.shape
    Dm = np.sqrt(((z[:, :, None] - z[:, None]) ** 2).sum(-1))
    return Dm.sum((1, 2)) / (n * (n - 1))


def longest_run(mask):
    """(start, end) indices of the longest run of True values, or None."""
    best, start = None, None
    for i, v in enumerate(list(mask) + [False]):
        if v and start is None:
            start = i
        if not v and start is not None:
            if best is None or i - start > best[1] - best[0] + 1:
                best = (start, i - 1)
            start = None
    return best


def design_metrics(x, alpha=None, target=None, level=0.7):
    """Engineering summary of one design: spectrum, bandwidth, paper-style FOM and geometry.

    The absorption band is the longest contiguous frequency range with alpha >= `level`.
    FOM follows Zeng et al. Eq. 8: alpha_avg * (df / f_centre) * lambda_max / V^(1/3),
    with lambda_max = c0 / f_low and V the cell volume (W * T * height).
    """
    x = np.asarray(x, float)
    w, g, d2, l0 = x[:6], x[6:11], x[11], x[12]
    if alpha is None:
        alpha = alpha_batch(w[None], g[None], np.array([d2]), np.array([l0]))[0]
    m = {"peak_alpha": float(alpha.max()), "peak_freq_Hz": float(FREQ[alpha.argmax()]),
         "mean_alpha_400_800": float(alpha[(FREQ >= 400) & (FREQ <= 800)].mean())}
    cell_height = HIN + 2 * TW
    for lvl, key in ((level, "band"), (0.9, "band90")):
        run = longest_run(alpha >= lvl)
        if run is None:
            m[key + "_Hz"] = (0.0, 0.0)
            m[key + "_width_Hz"] = 0.0
        else:
            lo, hi = FREQ[run[0]], FREQ[run[1]]
            m[key + "_Hz"] = (float(lo), float(hi))
            m[key + "_width_Hz"] = float(hi - lo)
            if key == "band":
                fc = (lo + hi) / 2
                rel = (hi - lo) / fc
                avg = float(alpha[run[0]:run[1] + 1].mean())
                V = W * T * cell_height
                m["alpha_avg_in_band"] = avg
                m["relative_bandwidth"] = float(rel)
                m["lambda_over_thickness"] = float(c0 / lo * 1e3 / cell_height)
                m["FOM_paper_eq8"] = float(avg * rel * (c0 / lo * 1e3) / V ** (1 / 3))
    m.setdefault("FOM_paper_eq8", 0.0)
    L = path_lengths(list(w), list(g), l0)
    m["acoustic_path_mm"] = float(sum(L) + l0 + TW)
    m["folding_factor"] = m["acoustic_path_mm"] / cell_height
    m["min_channel_mm"] = float(w.min())
    m["tube_d2_mm"], m["tube_l0_mm"] = float(d2), float(l0)
    if target is not None:
        m["rmse_to_target"] = float(np.sqrt(((alpha - target) ** 2).mean()))
    return m

# %% [section:visualisation]
def render_mask(x, h=0.5):
    """Pixel image of the cell cross-section (height 50 mm x width 41 mm, y up).

    Values: 0 = air, 1 = wall / panel, 2 = embedded-tube wall. Pixel size h [mm].
    """
    w, g, d2, l0 = x[:6], x[6:11], x[11], x[12]
    P = lambda v: int(round(v / h))
    nx, ny = P(W), P(HIN + 2 * TW)
    img = np.zeros((ny, nx), np.uint8)
    img[:P(TW), :] = 1
    img[-P(TW):, :] = 1
    img[:, :P(TW)] = 1
    img[:, -P(TW):] = 1
    xs = TW
    for i in range(N_CH - 1):                      # panel i separates channel i and i+1
        hh = HIN - g[i]
        y0 = (HIN + TW - hh) if i % 2 == 0 else TW   # even panels hang from the top plate
        img[P(y0):P(y0 + hh), P(xs + w[i]):P(xs + w[i]) + P(TW)] = 1
        xs += w[i] + TW
    cx = TW + w[0] / 2
    ytop, ybot = P(HIN + TW), P(HIN + TW - l0)
    for xx in (cx - d2 / 2 - TW, cx + d2 / 2):
        img[ybot:ytop, P(xx):P(xx) + P(TW)] = 2
    return img


def acoustic_path(x):
    """Centre-line (x, y) [mm] of the sound path from the tube mouth to the rigid end."""
    w, g, d2, l0 = x[:6], x[6:11], x[11], x[12]
    cx, xs = [], TW
    for i in range(N_CH):
        cx.append(xs + w[i] / 2)
        xs += w[i] + TW
    pts = [(cx[0], HIN + 2 * TW)]
    for i in range(N_CH - 1):
        yturn = TW + g[i] / 2 if i % 2 == 0 else HIN + TW - g[i] / 2
        pts += [(cx[i], yturn), (cx[i + 1], yturn)]
    pts.append((cx[-1], HIN + TW - 1.0 if (N_CH - 1) % 2 == 1 else TW + 1.0))
    return np.array(pts)


_CMAP = ListedColormap(["#f4f6f8", "#2b2b2b", "#c0392b"])


def draw(ax, x, title="", show_path=True, label_dims=True):
    """Draw one design: walls (dark), embedded tube (red), air (light), acoustic path (blue)."""
    ax.imshow(render_mask(x), cmap=_CMAP, vmin=0, vmax=2, origin="lower", extent=[0, W, 0, HIN + 2 * TW])
    if show_path:
        p = acoustic_path(x)
        ax.plot(p[:, 0], p[:, 1], color="#2a7fff", lw=1.4, ls="--")
        for a, b in zip(p[:-1], p[1:]):
            ax.annotate("", xy=((a + b) / 2 + (b - a) * 0.05), xytext=((a + b) / 2 - (b - a) * 0.05),
                        arrowprops=dict(arrowstyle="-|>", color="#2a7fff", lw=1.2))
    if label_dims:
        xs = TW
        for i in range(N_CH):
            ax.text(xs + x[i] / 2, -2.2, "%.1f" % x[i], ha="center", va="top", fontsize=6.5)
            xs += x[i] + TW
        ax.text(W / 2, -5.5, "channel widths [mm]", ha="center", va="top", fontsize=6.5, color="gray")
    ax.set_xlim(0, W)
    ax.set_ylim(-8 if label_dims else 0, HIN + 2 * TW)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=8)


def design_card(x, target=None, title=None, level=0.7):
    """One-figure summary of a design: geometry, spectrum (vs target) and key numbers."""
    x = np.asarray(x, float)
    a = alpha_batch(x[None, :6], x[None, 6:11], x[None, 11], x[None, 12])[0]
    m = design_metrics(x, a, target, level)
    fig = plt.figure(figsize=(11.5, 4.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[0.8, 1.6], height_ratios=[1.5, 1])
    draw(fig.add_subplot(gs[:, 0]), x, title or "Generated cell geometry")
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(FREQ, a, color="#1f77b4", lw=2, label="simulated")
    if target is not None:
        ax.plot(FREQ, target, "k--", lw=1.8, label="target")
    run = longest_run(a >= level)
    if run:
        ax.axvspan(FREQ[run[0]], FREQ[run[1]], color="#2ecc71", alpha=0.15, label="alpha >= %.1f band" % level)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Absorption coefficient")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.25)
    tx_ = fig.add_subplot(gs[1, 1])
    tx_.axis("off")
    band = m["band_Hz"]
    lines = [
        "peak alpha %.2f at %.0f Hz        mean alpha (400-800 Hz) %.2f" % (m["peak_alpha"], m["peak_freq_Hz"], m["mean_alpha_400_800"]),
        "alpha>=%.1f band: %.0f-%.0f Hz (%.0f Hz wide)   alpha>=0.9 width: %.0f Hz"
        % (level, band[0], band[1], m["band_width_Hz"], m["band90_width_Hz"]),
        "lambda / thickness at band start: %.1f        paper-style FOM (Eq. 8): %.1f" % (m.get("lambda_over_thickness", 0), m["FOM_paper_eq8"]),
        "acoustic path %.0f mm in a %.0f mm cell (folding x%.1f)   tube d=%.1f mm, l=%.1f mm"
        % (m["acoustic_path_mm"], HIN + 2 * TW, m["folding_factor"], m["tube_d2_mm"], m["tube_l0_mm"]),
    ]
    if "rmse_to_target" in m:
        lines.append("spectrum RMSE to target: %.3f" % m["rmse_to_target"])
    tx_.text(0, 1, "\n".join(lines), va="top", fontsize=8.5, family="monospace")
    plt.tight_layout()
    return fig, m


def gallery(result, k=4):
    """Top-k generated geometries side by side, with their spectra overlaid on the target."""
    k = min(k, len(result.x))
    fig = plt.figure(figsize=(2.9 * k, 6.2))
    gs = fig.add_gridspec(2, k, height_ratios=[1.7, 1])
    for j in range(k):
        draw(fig.add_subplot(gs[0, j]), result.x[j], "#%d  RMSE %.3f" % (j + 1, result.rmse[j]))
    ax = fig.add_subplot(gs[1, :])
    for j in range(k):
        ax.plot(FREQ, result.alpha[j], lw=1.6, label="#%d" % (j + 1))
    ax.plot(FREQ, result.target, "k--", lw=2, label="target")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Absorption coefficient")
    ax.legend(ncol=k + 1, fontsize=8)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    return fig

# %% [section:io]
ARTIFACT_FILES = {"checkpoint": "mnc_ddpm.pt", "demo": "demo_targets.npz", "results": "results.json"}


def save_artifacts(path, train_meta=None, results=None, demo=None):
    """Write everything needed to reuse the trained model without retraining.

    path       : output directory (created)
    train_meta : dict of training facts (epochs, final loss, dataset size, seed, ...)
    results    : dict of evaluation numbers (saved as results.json)
    demo       : dict with arrays 'A' (n, NF) and optionally 'X' (n, 13): reference spectra / designs
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "format_version": 1,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "mu": np.asarray(mu), "sd": np.asarray(sd),
        "arch": {"D": D, "C": NF, "h": 256, "nb": 5},
        "tsteps": TSTEPS,
        "physics": {"FREQ": FREQ, "LSCALE": LSCALE, "EPS": EPS, "DEL": DEL, "W": W, "T": T, "TW": TW, "HIN": HIN},
        "train_meta": train_meta or {},
    }
    torch.save(ckpt, path / ARTIFACT_FILES["checkpoint"])
    if results is not None:
        (path / ARTIFACT_FILES["results"]).write_text(json.dumps(results, indent=2, default=float))
    if demo is not None:
        np.savez_compressed(path / ARTIFACT_FILES["demo"], **demo)
    print("saved to", path.resolve(), "->", sorted(p.name for p in path.iterdir()))


def load(path, device=None, strict=True):
    """Restore the trained model and statistics from `save_artifacts`.

    The checkpoint metadata is validated before the weights are used: architecture
    (D=13, C=81, h=256, nb=5), diffusion steps, design standardisation (mu, sd) and the
    physics constants. A mismatch raises `ValueError` when `strict` is True (default),
    because the stored model would then be inconsistent with this module; with
    `strict=False` a physics mismatch is reported as a warning instead.

    Only load checkpoints from a trusted artifact directory: the file is a pickled
    PyTorch dictionary.

    Returns a metadata dict with keys 'format_version', 'arch', 'tsteps', 'physics'
    and 'train_meta'.
    """
    global model, mu, sd, betas, alphas, abar, dev
    if device is not None:
        dev = device
    path = Path(path)
    ckpt_path = path / ARTIFACT_FILES["checkpoint"]
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)

    expected_arch = {"D": D, "C": NF, "h": 256, "nb": 5}
    arch = ckpt.get("arch")
    if arch != expected_arch:
        raise ValueError(f"unexpected architecture {arch}; this module expects {expected_arch}")
    tsteps = ckpt.get("tsteps")
    if tsteps != TSTEPS:
        raise ValueError(f"unexpected number of diffusion steps {tsteps}; this module expects {TSTEPS}")

    mu_ = np.asarray(ckpt.get("mu"), dtype=float)
    sd_ = np.asarray(ckpt.get("sd"), dtype=float)
    if mu_.shape != (D,) or sd_.shape != (D,):
        raise ValueError(f"mu and sd must both have {D} entries; got {mu_.shape} and {sd_.shape}")
    if not np.isfinite(mu_).all() or not np.isfinite(sd_).all() or not (sd_ > 0).all():
        raise ValueError("mu and sd must be finite and sd must be strictly positive")

    ph = ckpt.get("physics") or {}
    expected_physics = {"FREQ": FREQ, "LSCALE": LSCALE, "EPS": EPS, "DEL": DEL,
                        "W": W, "T": T, "TW": TW, "HIN": HIN}
    mismatched = []
    for key, expected in expected_physics.items():
        if key not in ph:
            mismatched.append(f"{key} (missing)")
            continue
        got = np.asarray(ph[key], dtype=float)
        ref = np.asarray(expected, dtype=float)
        if got.shape != ref.shape or not np.allclose(got, ref):
            mismatched.append(key)
    if mismatched:
        msg = ("physics constants in this module differ from the ones used for training: "
               + ", ".join(mismatched))
        if strict:
            raise ValueError(msg + " (pass strict=False to load anyway)")
        print("WARNING:", msg)

    model = Denoiser(**arch).to(dev)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    mu, sd = mu_, sd_
    betas, alphas, abar = make_schedule(tsteps, dev)
    print("model restored | architecture: %s | %d diffusion steps | parameters: %.2fM | device: %s"
          % (arch, tsteps, sum(p.numel() for p in model.parameters()) / 1e6, dev))
    global _ARTIFACT_DIR
    _ARTIFACT_DIR = path
    return {"format_version": ckpt.get("format_version"), "arch": arch, "tsteps": tsteps,
            "physics": {k: np.asarray(ph[k]) for k in expected_physics if k in ph},
            "train_meta": ckpt.get("train_meta") or {}}


_ARTIFACT_DIR = None


def reference_spectrum(index=0):
    """A realistic target: the index-th best-absorbing held-out spectrum stored with the model."""
    d = np.load(_ARTIFACT_DIR / ARTIFACT_FILES["demo"])
    return d["A"][index].astype(np.float32)


def reference_design(index=0):
    """The original design that produced `reference_spectrum(index)` (if stored)."""
    d = np.load(_ARTIFACT_DIR / ARTIFACT_FILES["demo"])
    return d["X"][index].astype(float)


def shift_spectrum(a, frac):
    """Slide a spectrum along frequency by `frac` (0.05 = +5 %). Stays reliable up to about +-6 %."""
    return np.interp(FREQ, FREQ * (1 + frac), a).astype(np.float32)
