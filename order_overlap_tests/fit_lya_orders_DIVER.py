#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Minimal example: fit the Lyα line of the z = 5.36 LAE DIVER-GDN-1020514 (DJA v4.6, G140M/F070LP) in all three
diffraction orders at once with unite's higher-order overlap mode.

In this spectrum Lyα appears three times: in first order at λ_1 = 0.773 µm, and as second- and third-order images at
2 λ_1 = 1.546 µm and 3 λ_1 = 2.319 µm. With `order_overlap` set on the disperser, unite adds to the first-order model
F_1(λ) the terms r_m(λ) F_1(λ/m), where r_m(λ) = S_m(λ/m) / (m² S_1(λ)) comes from the msaexp sensitivity curves and
order-m light has the resolving power R_m(λ) = m R_1(λ) at its own wavelength. One Lyα line (one flux, redshift and
profile) therefore describes all three images: their positions, widths and fluxes follow from the model.

What you need to know to set this up:
    1. The disperser carries the ratios r_m, computed below from the msaexp sensitivity table: nirspec.G140M(order_overlap=...).
    2. unite only fits pixels inside continuum regions, and the model F_1(λ/m) is only non-zero where λ/m falls inside a
       continuum region. So there is one region around Lyα itself (whose continuum is also used for the overlap), and
       one region at the observed wavelengths of each higher-order image (given in rest-frame wavelength of first-order
       light, i.e. m × 1215.67 Å). The power-law continuum of those regions absorbs the first-order light at those
       wavelengths. These regions contain no line, so prepare() must be told to keep them (drop_empty_regions=False).
    3. Nothing else is special: the line is defined once, at its first-order (rest-frame) wavelength.

The script runs with either implementation of the overlap:
    - the order-overlap branch of https://github.com/joriswitstok/unite (order_overlap={m: callable}), or
    - https://github.com/TheSkyentist/unite/pull/29 (order_overlap=[OrderOverlap(m, <ratio file>)]), for which the
      ratio files are written to --data-dir; --order-scale then fits a normalisation of each of r_2 and r_3.
It also needs matplotlib. The spectrum and the msaexp sensitivity table are downloaded to --data-dir on first use.

NB: without --order-scale, the positions and normalisations of the higher-order images are fixed (no free per-order
scaling or shift); a toy-model fit of this spectrum with those free (DJA/Spectral_overlap/lya_orders_DIVER.py) gives
normalisations consistent with one, but a small shift of the second-order image (-37 ± 7 km/s, a third of a pixel),
which here slightly compromises the fit of the first-order line.

    python fit_lya_orders_DIVER.py --data-dir ./data

Joris Witstok, 5 November 2026 (both implementations: 11 November 2026)
"""

import os
import argparse
import urllib.request

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.table import Table

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from unite import instrument, model, prior
from unite.compute import evaluate_model
from unite.continuum import ContinuumConfiguration, ContinuumRegion, PowerLaw
from unite.instrument import RScale, nirspec
from unite.line import FWHM, Flux, LineConfiguration, Redshift
from unite.spectrum import Spectra, from_arrays

parser = argparse.ArgumentParser(description="Fit Lyα of DIVER-GDN-1020514 in three diffraction orders with unite")
parser.add_argument("--data-dir", default="data", help="Folder for the downloaded spectrum and sensitivity table")
parser.add_argument("--out", default="fit_lya_orders_DIVER.png", help="Output figure")
parser.add_argument("--order-scale", action=argparse.BooleanOptionalAction, default=False, help="Fit a normalisation of each of r_2 and r_3 (PR 29 only)")
parser.add_argument("--window", type=float, default=3000.0, help="Half-width of each order's window around Lyα (km/s)")
parser.add_argument("--num-warmup", type=int, default=500, help="NUTS warmup steps")
parser.add_argument("--num-samples", type=int, default=500, help="NUTS samples")
args = parser.parse_args()

C_KMS = 299792.458
LYA_REST = 1215.67  # Å (vacuum)
Z_SPEC = 5.35904  # z_best in DJA v4.6
FLUX_UNIT = u.Unit("1e-20 erg s-1 cm-2 AA-1")

SPEC_URL = "https://s3.amazonaws.com/msaexp-nirspec/extractions/diver-gdn-01a-v4/diver-gdn-01a-v4_g140m-f070lp_8018_1020514.spec.fits"
SENS_URL = "https://raw.githubusercontent.com/gbrammer/msaexp/main/msaexp/data/extended_sensitivity/msaexp_sensitivity_g140m_f070lp_001.fits"

def download(url):
    os.makedirs(args.data_dir, exist_ok=True)
    path = os.path.join(args.data_dir, os.path.basename(url))
    if not os.path.isfile(path):
        print("Downloading {}".format(url))
        urllib.request.urlretrieve(url, path + ".part")
        os.replace(path + ".part", path)
    return path

# --- Data: the DJA 1D spectrum, converted from µJy to f_λ ---
with fits.open(download(SPEC_URL)) as hdulist:
    spec_1D = hdulist["SPEC1D"].data
    wave = np.asarray(spec_1D["wave"], dtype=float)  # µm
    to_flam = 1e-29 * 2.99792458e18 / (wave * 1e4)**2 / FLUX_UNIT.scale
    flux = np.asarray(spec_1D["flux"], dtype=float) * to_flam
    err = np.asarray(spec_1D["err"], dtype=float) * to_flam

# Pixel edges halfway between the pixel centres; invalid pixels are dropped
mids = 0.5 * (wave[1:] + wave[:-1])
low = np.concatenate([[2 * wave[0] - mids[0]], mids])
high = np.concatenate([mids, [2 * wave[-1] - mids[-1]]])
valid = np.isfinite(flux) & np.isfinite(err) & (err > 0)

# --- 1. The disperser, with the higher-order throughput ratios r_2 and r_3 from msaexp ---
# msaexp tabulates the sensitivity S_m of order m (count rate per µJy) against the wavelength of the photons; order-m
# light of wavelength λ/m lands on the pixel of first-order wavelength λ, and the pipeline calibrates all counts with
# S_1(λ). In f_ν the calibrated spectrum is F(λ) + Σ_m S_m(λ/m) / S_1(λ) F(λ/m); in f_λ this gains a factor 1/m², so
#     r_m(λ) = S_m(λ/m) / (m² S_1(λ)).
pr29_api = hasattr(instrument, "OrderOverlap")
sens = Table.read(download(SENS_URL))
wl_sens = np.asarray(sens["wavelength"], dtype=float)
S_1 = np.nan_to_num(np.asarray(sens["sensitivity"], dtype=float))
overlap = {}
for m in (2, 3):
    S_m = np.nan_to_num(np.asarray(sens["sensitivity_{:d}".format(m)], dtype=float))
    wl_obs = m * wl_sens
    S_1_obs = np.interp(wl_obs, wl_sens, S_1, left=0.0, right=0.0)
    ratio = np.zeros_like(wl_obs)
    positive = (S_1_obs > 0) & (S_m > 0)
    ratio[positive] = S_m[positive] / (m**2 * S_1_obs[positive])
    print("Order {:d}: r_m = {:.3f} at the image of Lyα".format(m, np.interp(m * LYA_REST * 1e-4 * (1 + Z_SPEC), wl_obs, ratio)))
    if pr29_api:
        # PR 29: a file with an observed-wavelength column (length unit) and a dimensionless throughput_ratio
        path = os.path.join(args.data_dir, "throughput_ratio_g140m-f070lp_order{:d}.ecsv".format(m))
        Table({"wavelength": wl_obs * u.um, "throughput_ratio": ratio}).write(path, overwrite=True)
        scale = instrument.OrderScale(name="order{:d}".format(m), prior=prior.TruncatedNormal(1.0, 0.2, 0.5, 1.5)) if args.order_scale else None
        overlap[m] = instrument.OrderOverlap(m, path, scale=scale)
    else:
        # Fork: a callable of the observed wavelength, zero outside the table
        overlap[m] = nirspec.SensitivityRatio(m, wl_obs, ratio)
if pr29_api:
    overlap = list(overlap.values())
disperser = nirspec.G140M(
    r_source="point",
    # Resolution scale relative to the point-source R curve (as in fit_unite.py)
    r_scale=RScale(prior=prior.TruncatedNormal(loc=0.85, scale=0.15, low=0.5, high=1.1)),
    order_overlap=overlap,
)

spectrum = from_arrays(low=low[valid] * u.um, high=high[valid] * u.um, flux=flux[valid] * FLUX_UNIT,
                       error=err[valid] * FLUX_UNIT, disperser=disperser, name="g140m")
spectra = Spectra([spectrum], redshift=Z_SPEC)

# --- 3. One Lyα line, at its first-order wavelength ---
# An asymmetric exponentially modified Gaussian: Gaussian core with separate blue and red exponential tails
lines = LineConfiguration()
lines.add_line("Lya", LYA_REST * u.AA, profile="aemg",
               flux=Flux("Lya", prior=prior.Uniform(0, 10)),
               redshift=Redshift("Lya", prior=prior.Uniform(-0.02, 0.02)),
               fwhm_gauss=FWHM("Lya_gauss", prior=prior.Uniform(0, 800)),
               fwhm_l_blue=FWHM("Lya_blue", prior=prior.Uniform(0, 800)),
               fwhm_l_red=FWHM("Lya_red", prior=prior.Uniform(0, 1500)))

# --- 2. Continuum regions: Lyα itself, and the observed windows of the second- and third-order images ---
dz_window = args.window / C_KMS
regions = [ContinuumRegion(m * LYA_REST * (1 - dz_window) * u.AA, m * LYA_REST * (1 + dz_window) * u.AA,
                           form=PowerLaw(), name="order{}".format(m)) for m in (1, 2, 3)]
continuum = ContinuumConfiguration(regions)

# NB: the higher-order regions contain no line of their own (only the image of Lyα), so they must be kept explicitly:
# by default prepare() drops continuum regions without lines, and the fit would then only see the first order
spectra.prepare(lines, continuum, drop_empty_regions=False)
spectra.compute_scales(spectra.prepared_line_config, spectra.prepared_cont_config)
builder = model.ModelBuilder(spectra.prepared_line_config, spectra.prepared_cont_config, spectra)
samples, model_args = builder.fit(num_warmup=args.num_warmup, num_samples=args.num_samples, seed=0)

# --- Results ---
def site(*needles):
    """Name of the single posterior sample site whose name contains all needles"""
    keys = [k for k in samples if all(n in k for n in needles)]
    assert len(keys) == 1, (needles, sorted(samples))
    return keys[0]

def summary(x, unit=""):
    q16, q50, q84 = np.percentile(np.asarray(x), [16, 50, 84])
    return "{:.3g} (+{:.2g}/-{:.2g}){}".format(q50, q84 - q50, q50 - q16, unit)

line_scale = spectra.line_scale.to_value(u.erg / u.s / u.cm**2)
print("Lyα flux:", summary(np.asarray(samples[site("flux", "Lya")]) * line_scale / 1e-18, " × 1e-18 erg/s/cm^2"))
print("Lyα velocity offset:", summary(np.asarray(samples[site("z_", "Lya")]) / (1 + Z_SPEC) * C_KMS, " km/s"))
for name in ("Lya_gauss", "Lya_blue", "Lya_red"):
    print("{}:".format(name), summary(samples[site(name)], " km/s"))
print("r_scale:", summary(samples[next(k for k in samples if k.startswith("r_scale"))]))
for key in sorted(k for k in samples if k.startswith("order_scale")):
    print("{}:".format(key), summary(samples[key]))

# --- Figure: data and posterior-median model in each order's window ---
pred = evaluate_model(samples, model_args)[0]
wl = np.asarray(pred.wavelength)
data, data_err = np.asarray(model_args.spectra[0].flux), np.asarray(model_args.spectra[0].error)
total = np.median(pred.total, axis=0)
# (the higher-order term: an array on the fork, a dictionary of arrays per order in PR 29)
higher = np.median(sum(pred.overlap.values()) if isinstance(pred.overlap, dict) else pred.overlap, axis=0)

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 3.6), constrained_layout=True)
for ax, m in zip(axes, (1, 2, 3)):
    v = (wl / (m * LYA_REST * 1e-4 * (1 + Z_SPEC)) - 1) * C_KMS
    sel = np.abs(v) < args.window
    ax.fill_between(v[sel], (data - data_err)[sel], (data + data_err)[sel], step="mid", color="0.8", lw=0)
    ax.step(v[sel], data[sel], where="mid", color="k", lw=0.8, label="Data")
    ax.step(v[sel], total[sel], where="mid", color="#0072B2", lw=1.5, label="unite model")
    if m > 1:
        ax.step(v[sel], higher[sel], where="mid", color="#E69F00", lw=1.2, ls="--", label="Higher-order term")
    ax.axhline(0, color="0.5", lw=0.5)
    ax.set_title("Order {}: {:.3f} µm".format(m, m * LYA_REST * 1e-4 * (1 + Z_SPEC)))
    ax.set_xlabel(r"Velocity relative to systemic Ly$\alpha$ (km s$^{-1}$)")
    ax.legend(loc="upper right", fontsize="small")
axes[0].set_ylabel(r"$F_\lambda$ ($10^{-20}$ erg s$^{-1}$ cm$^{-2}$ $\mathrm{\AA}^{-1}$)")
fig.savefig(args.out, dpi=200)
print("Saved {}".format(args.out))
