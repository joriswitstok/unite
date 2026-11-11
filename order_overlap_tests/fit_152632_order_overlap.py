#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Minimal working example: fit the five NIRSpec spectra of the broad-line source JADES-GS 187956 at z = 3.704 (DJA v4.6,
PID 1286; ID 152632 of the DJAxUNITE catalogue) with unite, including the higher-order light in its G140M/F070LP and
G235M/F170LP spectra.

This is a reduced version of the main model of the catalogue fit (fit_unite.py, v4.6.0), keeping its structure:

    1. Five spectra (PRISM, G140M, G235M, G395M, G395H) sharing one resolution scale; the PRISM spectrum is the
       reference of the flux scales and has a free pixel offset.
    2. The dispersers of G140M/F070LP and G235M/F170LP carry the second-order throughput ratio
       r_2(λ_obs) = S_2(λ_obs / 2) / (4 S_1(λ_obs)), from the msaexp extended-sensitivity tables (third-order light
       stays below 1% over the extracted spectra and is left out, as in the catalogue).
    3. Narrow lines with a common redshift and width, and broad components of the hydrogen lines: free for Hα and Hβ,
       tied to the broad-to-narrow ratio of the nearest free line for the others (dependent priors).
    4. A Legendre polynomial continuum of order 5 in each of three regions, split at the Balmer and Paschen limits.
       These regions span all spectra, which the overlap term requires: the model F(λ/m) is only non-zero where λ/m
       falls inside a continuum region.
    5. Variational inference (a multivariate-normal guide, started from the starting values of the priors) gives the
       posterior; --nuts continues with NUTS, using the covariance of the guide as dense mass matrix.

Reduced with respect to the catalogue: 39 lines rather than 131 (those detected redwards of the Balmer limit, with
fixed multiplet ratios; no UV lines), a uniform rather than log-uniform prior on the narrow width, flux scales centred
on 1 rather than on the measured ratios to the PRISM spectrum, and no starting values from the DJA line measurements.
The variational inference takes a few minutes on a laptop and reproduces the kinematics and calibration parameters of
the catalogue fit.

Where the second-order light matters in this source (a blue rest-frame UV continuum):
    - G140M beyond 1.3 µm: r_2 reaches 0.17 at 1.4 µm, and the second-order image of the UV continuum raises the
      spectrum by up to a quarter between 1.3 and 2.2 µm.
    - G235M between 3.2 and 4.2 µm (r_2 up to 0.05) and again beyond 4.9 µm: a second-order continuum, and the image
      of [O II] 3727, 3730 at 3.507 µm. In between, r_2 is of order 0.001: the images of Hβ and [O III] (4.57 to
      4.71 µm) are negligible.

The script runs with either implementation of the overlap:
    - the order-overlap branch of https://github.com/joriswitstok/unite (order_overlap={m: callable}), or
    - https://github.com/TheSkyentist/unite/pull/29 (order_overlap=[OrderOverlap(m, <ratio file>)]), for which the
      ratio files are written to --data-dir; --order-scale then fits one normalisation of r_2 shared by both gratings.
Use --no-order-overlap for the same fit without the term. The spectra and sensitivity tables are downloaded to
--data-dir on first use.

    python fit_152632_order_overlap.py --data-dir ./data
    python fit_152632_order_overlap.py --data-dir ./data --no-order-overlap --out fit_152632_no_overlap.png

Joris Witstok, 11 November 2026
"""

import os
import time
import argparse
import urllib.request

import numpy as np
from astropy import units as u
from astropy.table import Table

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

import jax
jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp
import numpyro
from numpyro import infer

import unite
from unite import instrument, model, prior
from unite.compute import evaluate_model
from unite.continuum import ContinuumConfiguration, ContinuumRegion, Legendre
from unite.continuum.config import ContShape, Scale
from unite.instrument import nirspec
from unite.line import FWHM, Flux, LineConfiguration, Redshift
from unite.spectrum import Spectra, from_DJA

parser = argparse.ArgumentParser(description="Fit the spectra of DJA source 152632 (z = 3.704) with unite, with higher-order overlap")
bool_action = argparse.BooleanOptionalAction
parser.add_argument("--data-dir", default="data", help="Folder for the downloaded spectra and sensitivity tables")
parser.add_argument("--out", default="fit_152632_order_overlap.png", help="Output figure")
parser.add_argument("--order-overlap", action=bool_action, default=True, help="Model the second-order light in the G140M and G235M spectra")
parser.add_argument("--order-scale", action=bool_action, default=False, help="Fit a normalisation of r_2, shared by both gratings (PR 29 only)")
parser.add_argument("--svi-steps", type=int, default=6000, help="Steps of the variational inference")
parser.add_argument("--nuts", action=bool_action, default=False, help="Sample with NUTS after the variational inference (slow: about an hour)")
parser.add_argument("--num-chains", type=int, default=4, help="NUTS chains")
parser.add_argument("--num-warmup", type=int, default=300, help="NUTS warmup iterations")
parser.add_argument("--num-samples", type=int, default=500, help="Posterior samples (per chain for NUTS)")
args = parser.parse_args()
numpyro.set_host_device_count(args.num_chains)

C_KMS = 299792.458
Z_SPEC = 3.70407  # z_best in DJA v4.6
ZP1 = 1.0 + Z_SPEC

DJA_URL = "https://s3.amazonaws.com/msaexp-nirspec/extractions/jades-gds03-v4/jades-gds03-v4_{}_1286_187956.spec.fits"
MSAEXP_URL = "https://raw.githubusercontent.com/gbrammer/msaexp/main/msaexp/data/extended_sensitivity/msaexp_sensitivity_{}_001.fits"
GRATING_FILTERS = ["prism-clear", "g140m-f070lp", "g235m-f170lp", "g395m-f290lp", "g395h-f290lp"]
# Grating/filter combinations with second-order light, and the observed range (µm) of their DJA extractions
OVERLAP_RANGES = {"g140m-f070lp": (0.6, 2.3), "g235m-f170lp": (1.6, 3.8)}

def download(url):
    os.makedirs(args.data_dir, exist_ok=True)
    path = os.path.join(args.data_dir, os.path.basename(url))
    if not os.path.isfile(path):
        print("Downloading {}".format(url))
        urllib.request.urlretrieve(url, path + ".part")
        os.replace(path + ".part", path)
    return path

# --- 1. Higher-order throughput ratios from the msaexp sensitivity curves ---
# msaexp tabulates the sensitivity S_m of order m (count rate per µJy) against the wavelength of the photons; order-m
# light of wavelength λ/m lands on the pixel of first-order wavelength λ, and the pipeline calibrates all counts with
# S_1(λ). In f_ν the calibrated spectrum is F(λ) + Σ_m S_m(λ/m) / S_1(λ) F(λ/m); in f_λ this gains a factor 1/m², so
#     r_m(λ) = S_m(λ/m) / (m² S_1(λ)).
pr29_api = hasattr(instrument, "OrderOverlap")
order_scale = instrument.OrderScale(prior=prior.TruncatedNormal(1.0, 0.2, 0.5, 1.5)) if args.order_scale and pr29_api else None

def throughput_ratio(grating_filter, m):
    """Observed wavelength (µm) and ratio r_m of the order-m to the first-order throughput (f_λ convention)"""
    sens = Table.read(download(MSAEXP_URL.format(grating_filter.replace("-", "_"))))
    wl = np.asarray(sens["wavelength"], dtype=float)
    S_1 = np.nan_to_num(np.asarray(sens["sensitivity"], dtype=float))
    S_m = np.nan_to_num(np.asarray(sens["sensitivity_{:d}".format(m)], dtype=float))
    wl_obs = m * wl
    S_1_obs = np.interp(wl_obs, wl, S_1, left=0.0, right=0.0)
    ratio = np.zeros_like(wl_obs)
    valid = (S_1_obs > 0) & (S_m > 0)
    ratio[valid] = S_m[valid] / (m**2 * S_1_obs[valid])
    return (wl_obs, ratio)

def order_overlap(grating_filter, orders=(2, 3), min_ratio=0.01):
    """The order_overlap argument of the disperser of a grating/filter combination: the orders whose ratio exceeds
    min_ratio somewhere over the extracted spectra (each order adds a model evaluation)"""
    if not args.order_overlap or grating_filter not in OVERLAP_RANGES:
        return None
    overlap = {}
    for m in orders:
        wl_obs, ratio = throughput_ratio(grating_filter, m)
        in_range = (wl_obs >= OVERLAP_RANGES[grating_filter][0]) & (wl_obs <= OVERLAP_RANGES[grating_filter][1])
        if not np.any(ratio[in_range] > min_ratio):
            continue
        if pr29_api:
            # PR 29: a file with an observed-wavelength column (length unit) and a dimensionless throughput_ratio
            path = os.path.join(args.data_dir, "throughput_ratio_{}_order{:d}.ecsv".format(grating_filter, m))
            Table({"wavelength": wl_obs * u.um, "throughput_ratio": ratio}).write(path, overwrite=True)
            overlap[m] = instrument.OrderOverlap(m, path, scale=order_scale)
        else:
            # Fork: a callable of the observed wavelength, zero outside the table
            overlap[m] = nirspec.SensitivityRatio(m, wl_obs, ratio)
        print("{}: order {:d}, r_m up to {:.3f} over the extracted spectrum".format(grating_filter, m, ratio[in_range].max()))
    return (list(overlap.values()) if pr29_api else overlap) or None

# --- 2. Spectra and dispersers ---
resolution_scale = instrument.RScale(prior=prior.TruncatedNormal(loc=0.85, scale=0.15, low=0.5, high=1.1, init=0.85))
spectra_list = []
for gf in GRATING_FILTERS:
    name = gf.split("-")[0]
    is_prism = name == "prism"
    overlap = order_overlap(gf)
    disperser = getattr(nirspec, name.upper())(
        name=name, r_source="point", r_scale=resolution_scale,
        # The PRISM spectrum is the reference of the flux scales
        flux_scale=None if is_prism else instrument.FluxScale(name=name, prior=prior.TruncatedNormal(loc=1.0, scale=0.2, low=0.25, high=4.0, init=1.0)),
        pix_offset=instrument.PixOffset(name=name, prior=prior.TruncatedNormal(loc=0.1, scale=0.15, low=-0.4, high=0.8, init=0.1)) if is_prism else None,
        **({"order_overlap": overlap} if overlap else {}))
    spectra_list.append(from_DJA(path=download(DJA_URL.format(gf)), disperser=disperser, name=name))
spectra = Spectra(spectra_list, redshift=Z_SPEC)
print(spectra)

# --- 3. Lines ---
# Narrow lines: name, rest-frame wavelength (Å, vacuum), width of the flux prior (in units of the line scale), and for
# multiplets with fixed ratios the name of their common flux parameter and the strength relative to it
NARROW_LINES = [
    ("OII3727", 3727.09, 0.15, "OII3727", 1.0), ("OII3730", 3729.84, 0.15, "OII3727", 1.0),
    ("NeIII3870", 3869.85, 0.15, "NeIII3969", 3.32), ("NeIII3969", 3968.59, 0.15, "NeIII3969", 1.0),
    ("HI3971", 3971.20, 0.05), ("HI4103", 4102.89, 0.15), ("HI4342", 4341.69, 0.5), ("OIII4364", 4364.45, 0.15),
    ("HI4863", 4862.71, 0.5), ("OIII4960", 4960.29, 0.5, "OIII4960", 1.0), ("OIII5008", 5008.24, 0.5, "OIII4960", 2.98),
    ("HeI5877", 5877.29, 0.15), ("OI6302", 6302.05, 0.15, "OI6366", 3.13), ("OI6366", 6365.54, 0.15, "OI6366", 1.0),
    ("NII6550", 6549.86, 0.15, "NII6550", 1.0), ("HI6565", 6564.60, 0.5), ("NII6585", 6585.27, 0.15, "NII6550", 2.95),
    ("SII6718", 6718.30, 0.15), ("SII6733", 6732.67, 0.15), ("HeI7067", 7067.20, 0.05),
    ("OII7321", 7321.13, 0.05, "OII7332", 0.60), ("OII7322", 7322.20, 0.05, "OII7332", 2.0),
    ("OII7332", 7331.76, 0.05, "OII7332", 1.0), ("OII7333", 7332.83, 0.05, "OII7332", 1.05), ("OI8449", 8448.78, 0.05),
    ("SIII9070", 9070.05, 0.15, "SIII9070", 1.0), ("SIII9532", 9532.25, 0.15, "SIII9070", 2.51), ("HI9549", 9548.65, 0.05),
    ("CI9827", 9826.80, 0.05, "CI9827", 1.0), ("CI9853", 9852.94, 0.05, "CI9827", 2.98),
    ("HI10052", 10052.12, 0.05), ("HeI10833", 10833.14, 0.15), ("HI10941", 10941.08, 0.15),
]
# Broad components: free for Hα and Hβ, the others tied to the broad-to-narrow flux ratio of the nearest free line
BROAD_FREE = ["HI6565", "HI4863"]
BROAD_TIED = {"HI4342": "HI4863", "HI4103": "HI4863", "HI10941": "HI6565", "HI10052": "HI6565"}

dz = 0.001 * ZP1
z_common = Redshift(name="common", prior=prior.Uniform(-dz, dz, init=0.0))
fwhm_common = FWHM(name="common", prior=prior.Uniform(10.0, 1000.0, init=200.0))
z_broad = Redshift(name="broad", prior=prior.Uniform(-dz, dz, init=0.0))
fwhm_broad = FWHM(name="broad", prior=prior.Uniform(800.0, 10000.0, init=2500.0))

def flux_prior(width):
    # Half-normal around zero, bounded at three times the line scale and started at its median
    return prior.TruncatedNormal(loc=0.0, scale=width, low=0.0, high=3.0, init=0.674 * width)

lines = LineConfiguration()
wavelengths, narrow_flux = {}, {}
for name, wl, width, *doublet in NARROW_LINES:
    flux_name, strength = doublet if doublet else (name, 1.0)
    if flux_name not in narrow_flux:
        narrow_flux[flux_name] = Flux(flux_name, prior=flux_prior(width))
    wavelengths[name] = wl
    lines.add_line(name=name, center=wl * u.AA, profile="Gaussian", redshift=z_common, flux=narrow_flux[flux_name],
                   strength=strength, fwhm_gauss=fwhm_common)

broad_flux = {name: Flux(name + "_broad", prior=flux_prior(0.5)) for name in BROAD_FREE}
for name, ref in BROAD_TIED.items():
    broad_flux[name] = Flux(name + "_broad", prior=prior.Fixed(narrow_flux[name] * broad_flux[ref] / narrow_flux[ref]))
for name, flux in broad_flux.items():
    lines.add_line(name=name + "_broad", center=wavelengths[name] * u.AA, profile="Gaussian", redshift=z_broad, flux=flux,
                   fwhm_gauss=fwhm_broad)

# --- 4. Continuum: a Legendre polynomial of order 5 in three regions, split at the Balmer and Paschen limits ---
# The scale of a region is its mean level and the shape parameters are in units of it
LEGENDRE_ORDER = 5
max_wl_emit = max(float(np.max(np.asarray(s.high)) * s.unit.to(u.AA)) for s in spectra_list) / ZP1
REGIONS = {"UV": (1300.0, 3645.07), "opt": (3645.07, 8201.40), "NIR": (8201.40, max_wl_emit)}

def continuum_configuration(levels=None):
    """Continuum configuration; with the mean level of each region (in units of the continuum scale), the prior of its
    scale is a truncated normal of relative width 30% around it, rather than flat"""
    regions = []
    for name, (low, high) in REGIONS.items():
        if levels is None:
            scale_prior = prior.Uniform(0.0, 2.0, init=0.7)
        else:
            scale_prior = prior.TruncatedNormal(loc=levels[name], scale=max(0.3 * levels[name], 0.02), low=0.0, high=2.0, init=levels[name])
        params = {"scale": Scale(name=name, prior=scale_prior),
                  **{"p{:d}".format(i): ContShape(name=name, prior=prior.Uniform(-3.0, 3.0, init=0.0)) for i in range(1, LEGENDRE_ORDER + 1)}}
        regions.append(ContinuumRegion(low * u.AA, high * u.AA, form=Legendre(order=LEGENDRE_ORDER), name=name, params=params))
    return ContinuumConfiguration(regions)

# --- 5. Flux scales and model ---
# (drop_empty_regions=False keeps all regions, whether or not they contain a line)
def prepare(continuum):
    filtered = spectra.prepare(lines, continuum, drop_empty_regions=False)
    spectra.compute_scales(*filtered, line_mask_width=5000 * u.km / u.s, box_width=3000 * u.km / u.s)
    return filtered

prepare(continuum_configuration())
# The level of the continuum in each region, from the median of the PRISM spectrum
prism = spectra_list[0]
wl_emit = 0.5 * np.asarray(prism.low + prism.high) * prism.unit.to(u.AA) / ZP1
F_prism = np.asarray(prism.flux) * prism.flux_unit.to(spectra.continuum_scale.unit) / spectra.continuum_scale.value
levels = {name: float(np.clip(np.nanmedian(F_prism[(wl_emit > low) & (wl_emit < high)]), 0.02, 1.98)) for name, (low, high) in REGIONS.items()}
filtered_lines, filtered_cont = prepare(continuum_configuration(levels))

# The line scale is the peak of the strongest line times a box of 3000 km/s, but the flux of an unresolved line is its
# peak times the width of the line-spread function (up to 10000 km/s for the PRISM): the scale is widened accordingly,
# so that the flux priors (bounded at 3 times the line scale) do not cap the strongest line
R_min = min(float(np.min(np.asarray(s.disperser.R(0.5 * (s.low + s.high))))) for s in spectra_list)
spectra.line_scale = spectra.line_scale * min(max(1.06 * (C_KMS / R_min) / 3000.0, 1.0), 4.0)
print("Line scale: {:.4g}\nContinuum scale: {:.4g}\nContinuum levels: {}".format(spectra.line_scale, spectra.continuum_scale, levels))

model_fn, model_args = model.ModelBuilder(line_config=filtered_lines, continuum_config=filtered_cont, spectra=spectra).build()

# --- 6. Variational inference, optionally followed by NUTS ---
t0 = time.time()
guide = infer.autoguide.AutoMultivariateNormal(model_fn, init_loc_fn=infer.init_to_value(values=model_args.init_values), init_scale=0.01)
# (Adam with clipped gradients and a learning rate decaying exponentially from 0.01 to 1e-4)
optimiser = numpyro.optim.ClippedAdam(step_size=lambda i: jnp.maximum(0.01 * 0.7**(8.0 * i / args.svi_steps), 1e-4), clip_norm=10.0)
svi = infer.SVI(model_fn, guide, optimiser, loss=infer.Trace_ELBO(num_particles=4))
svi_result = svi.run(jax.random.PRNGKey(0), args.svi_steps, model_args, progress_bar=True)
n_params = int(svi_result.params["auto_loc"].size)
print("SVI of the {:d}-parameter model: ELBO loss {:.1f} (mean of the last 100 steps), {:.0f} s".format(
      n_params, float(np.mean(np.asarray(svi_result.losses[-100:]))), time.time() - t0))

if args.nuts:
    # NUTS, started at the median of the guide with its covariance (in unconstrained space) as dense mass matrix
    t0 = time.time()
    scale_tril = svi_result.params["auto_scale_tril"]
    kernel = infer.NUTS(model_fn, target_accept_prob=0.8, max_tree_depth=8, dense_mass=True, inverse_mass_matrix=scale_tril @ scale_tril.T,
                        adapt_mass_matrix=False, init_strategy=infer.init_to_value(values=guide.median(svi_result.params)))
    mcmc = infer.MCMC(kernel, num_warmup=args.num_warmup, num_samples=args.num_samples, num_chains=args.num_chains)
    mcmc.run(jax.random.PRNGKey(1), model_args)
    samples = mcmc.get_samples()
    print("NUTS: {:.0f} s, {:d} divergences".format(time.time() - t0, int(np.sum(np.asarray(mcmc.get_extra_fields()["diverging"])))))
else:
    samples = guide.sample_posterior(jax.random.PRNGKey(1), svi_result.params, sample_shape=(args.num_samples,))
samples = {key: np.asarray(value) for key, value in samples.items()}

# --- Results ---
def summary(key, factor=1.0, offset=0.0):
    q16, q50, q84 = np.percentile(samples[key] * factor + offset, [16, 50, 84])
    return "{:.5g} (+{:.2g}/-{:.2g})".format(q50, q84 - q50, q50 - q16)

line_scale = spectra.line_scale.to_value(1e-18 * u.erg / u.s / u.cm**2)
print("\nPosterior medians (16th-84th percentiles)")
for key in sorted(key for key in samples if key.split("_")[0] in ["z", "fwhm", "r", "pix", "order"] or key.startswith("flux_scale")):
    print("  {}: {}".format(key, summary(key, offset=Z_SPEC if key.startswith("z_") else 0.0)))
for key in ["flux_HI4863", "flux_HI4863_broad", "flux_OIII4960", "flux_HI6565", "flux_HI6565_broad", "flux_HI10052", "flux_HI10941"]:
    print("  {} (1e-18 erg/s/cm2): {}".format(key, summary(key, factor=line_scale)))

# Model spectra of a subset of the posterior draws (the decomposition per line takes memory)
rng = np.random.default_rng(0)
subset = rng.choice(len(next(iter(samples.values()))), size=min(100, args.num_samples), replace=False)
predictions = evaluate_model({key: value[subset] for key, value in samples.items()}, model_args)

def overlap_term(pred):
    """Median higher-order contribution: an array on the fork, a dictionary of arrays per order in PR 29"""
    if pred.overlap is None or (isinstance(pred.overlap, dict) and not pred.overlap):
        return np.zeros(pred.total.shape[1])
    return np.median(sum(pred.overlap.values()) if isinstance(pred.overlap, dict) else pred.overlap, axis=0)

# Goodness of fit: per spectrum, and over the pixels reached by second-order light (twice the cut-on of the filter)
SECOND_ORDER_FROM = {"g140m": 1.3, "g235m": 3.3}
print("\nChi-squared of the posterior-median model (unscaled uncertainties)")
results = {}
for s, pred in zip(model_args.spectra, predictions):
    wl = np.asarray(pred.wavelength)
    data, err = np.asarray(s.flux), np.asarray(s.error)
    total = np.median(pred.total, axis=0)
    results[s.name] = (wl, data, err, total, overlap_term(pred))
    chi2 = ((data - total) / err)**2
    line = "  {}: {:.1f} over {:d} pixels (reduced {:.3f})".format(s.name, chi2.sum(), chi2.size, chi2.mean())
    if s.name in SECOND_ORDER_FROM:
        sel = wl > SECOND_ORDER_FROM[s.name]
        line += "; beyond {:.1f} µm: {:.1f} over {:d} pixels (reduced {:.3f})".format(SECOND_ORDER_FROM[s.name], chi2[sel].sum(), sel.sum(), chi2[sel].mean())
    print(line)

# --- Figure: the G140M and G235M spectra, the model and its second-order term ---
def rebin(x, n):
    return np.nanmean(x[:x.size // n * n].reshape(-1, n), axis=1)

# (spectrum, wavelength range, rebinning factor, title, lines marked in second and in first order)
panels = [("g140m", (0.9, 3.2), 16, "G140M/F070LP (rebinned by 16 pixels)", [], []),
          ("g235m", (1.7, 5.3), 8, "G235M/F170LP (rebinned by 8 pixels)", [], []),
          ("g235m", (3.15, 3.75), 2, "G235M/F170LP (rebinned by 2 pixels): onset of the second order, and the image of [O II] 3727, 3730",
           ["OII3727", "NeIII3870"], ["HeI7067", "OII7322"])]
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8.27, 11.69 * 0.75), constrained_layout=True)
flux_unit = model_args.spectra[0].flux_unit
for ax, (name, (wl_min, wl_max), n, title, second_order_lines, first_order_lines) in zip(axes, panels):
    wl, data, err, total, higher = results[name]
    sel = (wl > wl_min) & (wl < wl_max)
    ax.step(rebin(wl[sel], n), rebin(data[sel], n), where="mid", color="0.6", lw=0.8, label="Data")
    ax.step(rebin(wl[sel], n), rebin(total[sel], n), where="mid", color="k", lw=1.0, label="Model")
    ax.step(rebin(wl[sel], n), rebin((total - higher)[sel], n), where="mid", color="C0", lw=1.0, alpha=0.8, label="First order only")
    ax.step(rebin(wl[sel], n), rebin(higher[sel], n), where="mid", color="C3", lw=1.0, alpha=0.8, label="Second-order term")
    # Second-order images of lines (dotted) and first-order lines (dashed)
    for line_name in second_order_lines:
        ax.axvline(2 * wavelengths[line_name] * ZP1 * 1e-4, color="C3", ls=":", lw=0.8)
    for line_name in first_order_lines:
        ax.axvline(wavelengths[line_name] * ZP1 * 1e-4, color="C0", ls="--", lw=0.8)
    ax.axhline(0, color="0.5", lw=0.5)
    ax.set_xlim(wl_min, wl_max)
    ymax = 1.5 * np.nanpercentile(rebin(total[sel], n), 98)
    ax.set_ylim(-0.1 * ymax, ymax)
    ax.set_title(title, fontsize="medium")
fig.supylabel(r"Flux density $F_\lambda$ ({})".format(flux_unit.to_string("latex_inline")))
axes[0].legend(loc="upper right", ncols=2, frameon=False, fontsize="small")
axes[-1].set_xlabel(r"Observed wavelength $\lambda_\mathrm{obs} \, (\mathrm{\mu m})$")
fig.savefig(args.out, dpi=200, bbox_inches="tight")
print("\nSaved {} (unite from {}, {})".format(args.out, os.path.dirname(unite.__file__), "PR 29 interface" if pr29_api else "fork interface"))
