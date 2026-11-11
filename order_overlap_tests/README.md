# unite

Standalone examples of fitting NIRSpec spectra with [unite](https://github.com/TheSkyentist/unite), written to be
shared (with Raphael Hviding, for testing the higher-order grating overlap). Each script is self-contained: it needs
only unite, numpyro and matplotlib, and downloads its own spectra and msaexp sensitivity tables to `--data-dir`.

Both scripts compute the throughput ratios r_m(λ_obs) = S_m(λ_obs/m) / (m² S_1(λ_obs)) from the msaexp tables
themselves and run with either implementation of the overlap, detected at run time:

- the `order-overlap` branch of the fork in `Collabcode/unite` (`order_overlap={m: callable}`), the version used for
  the v4.6.0 catalogue;
- [TheSkyentist/unite#29](https://github.com/TheSkyentist/unite/pull/29) (`order_overlap=[OrderOverlap(m, file)]`),
  for which the ratio files are written to `--data-dir`. `--order-scale` then fits the normalisation of the ratios
  (`OrderScale`).

## Scripts

- `fit_152632_order_overlap.py`: the five spectra (PRISM, G140M, G235M, G395M, G395H) of the broad-line source at
  z = 3.704 (ID 152632, JADES-GS 187956), with second-order light in G140M/F070LP and G235M/F170LP. A reduced
  version of the main model of `DJA/fit_unite.py` (39 lines rather than 131, 53 parameters rather than 105): shared
  resolution scale, flux scales relative to the PRISM spectrum, PRISM pixel offset, narrow lines and broad hydrogen
  components (tied through dependent priors), Legendre continuum in three regions, variational inference (about two
  minutes on the MacBook) and optionally NUTS (`--nuts`, not timed). `--no-order-overlap` gives the same fit without
  the term. The figure shows the G140M and G235M spectra, the model and its second-order term.
- `fit_lya_orders_DIVER.py`: the Lyα line of the z = 5.36 LAE DIVER-GDN-1020514 in first, second and third order of
  its G140M/F070LP spectrum, fitted as one line (NUTS, about 15 s). Adapted from
  `DJA/Spectral_overlap/unite_fit_lya_orders.py`, which only runs on the fork.

## Running

In the Pixi environment of the fork:

```
cd unite
pixi run --frozen -e test-py312 python /path/to/scripts/fit_152632_order_overlap.py --data-dir <folder>
```

To run against another unite (e.g. a checkout of PR 29), put it first on the `PYTHONPATH`. For PR 29, from the fork
(with `upstream` pointing to TheSkyentist/unite):

```
git fetch upstream pull/29/head:pr29
git worktree add ../unite-pr29 pr29
PYTHONPATH=../unite-pr29 pixi run --frozen -e test-py312 python /path/to/scripts/fit_lya_orders_DIVER.py --data-dir <folder>
```

The script prints which unite it imports and which interface it uses.

## Results (11 November 2026)

Fork at `order-overlap` (commit `ecb063d`, based on v3.5.2, with the uncommitted `CONT_LSF_REFERENCE` change in
`unite/model.py`, which does not affect these models) and PR 29 at commit `1ff3d77`, variational inference with the default settings.

`fit_152632_order_overlap.py`, chi-squared of the posterior-median model:

| Run | G140M (4031 pixels) | G235M (3510 pixels) | Flux scale G140M |
| --- | --- | --- | --- |
| Fork, overlap | 4272.8 | 4438.8 | 1.023 |
| PR 29, overlap | 4272.9 | 4438.1 | 1.023 |
| PR 29, overlap with free `OrderScale` | 4249.9 | 4396.0 | 1.005 |
| Fork, no overlap | 5657.1 | 4818.3 | 1.105 |

- The two implementations agree: the ELBO loss (-4026.8 and -4026.9) and all parameters match within the
  sampling noise.
- The kinematics and calibration match the catalogue fit: broad FWHM 2956 km/s (catalogue 2956), common FWHM 838 km/s
  (846), resolution scale 0.822 (0.828), broad redshift 3.6994 (3.699).
- Without the term, the G140M flux scale absorbs part of the second-order continuum (1.105 instead of 1.023).
- With a free normalisation shared by both gratings, the second-order ratio is scaled by 1.23 ± 0.03 (from the
  variational posterior, not NUTS).

`fit_lya_orders_DIVER.py`: Lyα flux 45.6 ± 0.6 × 10⁻¹⁸ erg/s/cm² on both implementations; with free normalisations in
PR 29, 0.99 ± 0.03 (second order) and 0.96 ± 0.12 (third order).

PR 29 warns at build time (`UserWarning`): "Spectrum 'g140m': a continuum region edge falls inside the order 2 source
window [0.7655, 1.1708] um; the overlap continuum switches off partway."
The second-order ratio is non-zero on the pixels of both image regions (1.531–1.562 µm and 2.296–2.342 µm). The source
wavelengths λ/2 of the first lie inside the Lyα region (0.7653–0.7808 µm), but those of the second (1.148–1.171 µm)
lie in no continuum region, so the model has no first-order flux there and the second-order continuum from those
wavelengths is left out (as in the fork, which drops it silently). The free power law of the third-order region takes
it up, so the fit is unaffected; a continuum region at 1.148–1.171 µm would model it explicitly. The reported window
is the minimum and maximum of two disjoint pieces, so the message suggests a region edge inside one range rather than
an uncovered sub-range: reporting the uncovered part (here 1.148–1.171 µm) would make it clearer.
