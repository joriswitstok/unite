# Instruments & Spectrum Loading

`unite` splits instrumental configuration and spectral data into two modules:

- **`unite.instrument`** — disperser classes and calibration tokens (hardware configuration)
- **`unite.spectrum`** — spectrum containers and loader functions (data I/O)

This separation means you can configure and serialize dispersers independently of any particular
dataset, then load spectra against those configurations.

---

## Dispersers

A disperser describes an instrument's spectral characteristics:

- **R(λ)** — the resolving power as a function of wavelength, used to compute the LSF FWHM
  at each pixel.
- **(dλ/dpix)(λ)** — the wavelength dispersion per pixel as a function of wavelength, used to
  convert between pixel offsets and wavelength offsets.

### Calibration Tokens

For a given instrument the user can optionally attach calibration tokens with associated priors:

- {class}`~unite.instrument.RScale`: Resolution scaling factor (e.g. R_eff = R_nominal x RScale)
- {class}`~unite.instrument.FluxScale`: Flux scaling factor (e.g. F_eff = F_model x FluxScale)
- {class}`~unite.instrument.PixOffset`: Detector pixel displacement (e.g. λ_eff = λ_model − (dλ/dpix)(λ_model) x PixOffset)

These absorb calibration offsets and uncertainties between instruments.

```python
from unite.instrument import RScale, FluxScale, PixOffset
```

In practice `FluxScale` and `PixOffset` are most useful for **relative** calibration between
spectra — one spectrum is treated as the reference with fixed calibration, and the other spectra
have free parameters to account for differences in flux calibration and wavelength solution.

:::{warning}
At least one spectrum should normally keep a **fixed** `FluxScale` (the default). If every
spectrum in a fit is given a free `FluxScale`, the parameter is completely degenerate with the
overall flux normalization of the model (line fluxes, continuum amplitude) and the fit becomes
non-identifiable. Only give every spectrum a free `FluxScale` if you have an informative prior on
it (e.g. from an independent photometric calibration) — never leave every spectrum's `FluxScale`
free with an uninformative prior.
:::

#### RScale — Resolution Calibration

Multiplies the nominal $R(\lambda)$ by a free factor:
$R_\mathrm{eff}(\lambda) = R_\mathrm{nominal}(\lambda) \times \text{r\_scale}$

```python
r = RScale(prior=prior.TruncatedNormal(1.0, 0.05, 0.8, 1.2))
```

Useful when the actual LSF is broader or narrower than the calibration predicts — e.g., for
extended sources in NIRSpec, or when the slit filling fraction differs from the calibration
assumption.

#### FluxScale — Relative Flux Calibration

Multiplies the model flux for all pixels in a spectrum by a free factor. A value of 1.0 means no
correction.

```python
f = FluxScale(prior=prior.Uniform(0.5, 2.0))
```

#### PixOffset — Wavelength Offset

Encodes the displacement of the spectrum on the detector relative to the wavelength calibration,
in pixels. A positive value means the spectrum is displaced toward longer wavelengths: the model
subtracts `pix_offset x (dλ/dpix)` from the calibrated pixel-edge wavelengths, shifting them
blueward to compensate.

For example, if a disperser consistently returns redshifts that are too high compared to a
reference (the calibrated wavelengths are too long), set a positive `PixOffset` to correct it:

```python
p = PixOffset(prior=prior.Uniform(-2, 2))
```

#### Parameter Naming

Each calibration token accepts an optional semantic label as its first argument. A
**type-specific prefix is automatically prepended** to form the final site name:

- `RScale('shared')` → site name `'r_scale_shared'` (prefix: `r_scale_`)
- `FluxScale('shared')` → site name `'flux_scale_shared'` (prefix: `flux_scale_`)
- `PixOffset('shared')` → site name `'pix_offset_shared'` (prefix: `pix_offset_`)

Pass **only the semantic label**, not the full prefixed name. For unshared tokens on a single named
disperser, the disperser name is used automatically (e.g., `RScale()` on `'G235H'` →
`'r_scale_G235H'`).

#### Sharing Calibration Tokens

Pass the **same token instance** to multiple dispersers to share a calibration parameter:

```python
shared_r = RScale(prior=prior.TruncatedNormal(1.0, 0.05, 0.8, 1.2), name='shared')
g235h = nirspec.G235H(r_scale=shared_r)
g395h = nirspec.G395H(r_scale=shared_r)
# → Both dispersers use the same 'r_scale_shared' parameter in the model
```

#### Fixed Calibration

Use {class}`~unite.prior.Fixed` to apply a known correction without uncertainty. This is the
default for all calibration tokens — if you don't specify a token the disperser is treated as the
reference with no correction.

```python
g395h = nirspec.G395H(flux_scale=FluxScale(prior=prior.Fixed(1.15)))
```

---

### JWST/NIRSpec

Convenience classes are provided for each NIRSpec disperser:

```python
from unite.instrument import nirspec

# nirspec.G140H, nirspec.G140M, nirspec.G235H, nirspec.G235M,
# nirspec.G395H, nirspec.G395M, nirspec.PRISM
```

NIRSpec dispersers support two resolution calibration modes via `r_source`:

```python
g235h = nirspec.G235H(r_source='point')     # point-source R (de Graaff et al. 2024)
g235h = nirspec.G235H(r_source='uniform')   # uniform illumination R from JDox
```

- **`'point'`** (default): Uses polynomial fits to the slit-centered point-source resolving power
  derived from
  [de Graaff et al. (2024)](https://ui.adsabs.harvard.edu/abs/2024A%26A...684A..87D/abstract).
  Recommended for most sources. Even resolved sources can use this curve combined with an `RScale`
  token to account for the effective slit filling fraction or position.
- **`'uniform'`**: Uses tabulated resolving power for uniform illumination of the slit from
  [JDOX](https://jwst-docs.stsci.edu/jwst-near-infrared-spectrograph/nirspec-instrumentation/nirspec-dispersers-and-filters#gsc.tab=0).

Here is an example applying calibration tokens to NIRSpec dispersers, using realistic offsets
between G395M and PRISM based on the RUBIES survey
([de Graaff et al. 2025](https://ui.adsabs.harvard.edu/abs/2025A%26A...697A.189D/abstract)):

```python
from unite.instrument import nirspec, RScale, FluxScale, PixOffset
from unite import prior

# Free resolution scale shared across both dispersers
r_scale = RScale(prior=prior.TruncatedNormal(low=0.7, high=1.3, loc=1.0, scale=0.3))

# G395M is the reference — no FluxScale or PixOffset
g395m = nirspec.G395M(r_scale=r_scale)

# PRISM has free flux and wavelength offsets relative to G395M
prism = nirspec.PRISM(
    r_scale=r_scale,
    flux_scale=FluxScale(prior=prior.TruncatedNormal(low=0.8, high=1.4, loc=1.1, scale=0.1)),  # de Graaff+25 Fig 8
    pix_offset=PixOffset(prior=prior.TruncatedNormal(low=-0.2, high=0.6, loc=0.2, scale=0.1)),  # de Graaff+25 Fig 9
)
```

#### Higher-order grating overlap

NIRSpec gratings are blazed, and the order-sorting filters do not fully suppress light from
higher diffraction orders: with **F070LP**, G140M/H spectra contain second-order light redwards
of ~1.3 µm (twice the filter cut-on), and with **F170LP**, G235M/H contain second-order light
redwards of ~3.3 µm. Third-order light starts at three times the cut-on and only reaches the
detector in extended extractions. A pipeline that calibrates the whole trace as first-order light
therefore produces a spectrum

$$
F_\lambda^{\rm obs}(\lambda) = F_\lambda(\lambda) + \sum_{m \ge 2} r_m(\lambda)\,
F_\lambda(\lambda / m),
$$

where $F_\lambda$ is the intrinsic (first-order) spectrum and $r_m(\lambda)$ is a deterministic
ratio of the order-$m$ to first-order throughput at the observed wavelength.

Every disperser accepts an optional `order_overlap` mapping `{order: r_m}` of JAX-jittable
callables (observed wavelength in the disperser's unit → ratio). When set, `unite` adds
$r_m(\lambda)\,F_1(\lambda/m)$ to the model in **both** integration modes:

- the full first-order model (lines **and** continuum, including absorbers) is evaluated at
  $\lambda/m$, so it is only non-zero where $\lambda/m$ falls inside a configured continuum
  region — make sure your {class}`~unite.continuum.ContinuumConfiguration` covers both the
  contaminated range and the range it is contaminated by;
- light of wavelength $\lambda_m = \lambda/m$ forms the same image on the detector (the same
  number of pixels) in every order, while the dispersion of order $m$ is $m$ times larger, so
  the order-$m$ resolving power is $R_m(\lambda_m) = m\,R_1(\lambda_m)$, with $R_1$ evaluated at
  the **photon** wavelength $\lambda_m$. In observed wavelength the order-$m$ image of a line is
  therefore as wide as the first-order LSF at $\lambda_m$ (plus $m$ times the intrinsic width),
  not $m$ times as wide. For gratings, whose resolving power is close to $R_1 \propto \lambda$,
  this nearly equals the first-order LSF of the pixel where the light lands, $R_1(\lambda)$;
  convolution mode convolves the overlap term separately for this reason;
- `RScale`, `FluxScale` and `PixOffset` apply to the overlap term exactly as to the first-order
  term.

{func}`~unite.instrument.nirspec.order_overlap_from_msaexp` builds the ratio callables from the
empirical sensitivity curves shipped with
[msaexp](https://github.com/gbrammer/msaexp/tree/main/msaexp/data/extended_sensitivity)
(`msaexp_sensitivity_{grating}_{filter}_001.fits`, columns `wavelength`, `sensitivity`,
`sensitivity_2`, `sensitivity_3`). The files are not bundled with `unite`; pass a path or an
{class}`~astropy.table.Table`:

```python
from unite.instrument import nirspec

overlap = nirspec.order_overlap_from_msaexp('msaexp_sensitivity_g140m_f070lp_001.fits')
# {2: SensitivityRatio(order=2, support=(1.28, 3.2) um, ...),
#  3: SensitivityRatio(order=3, ...)}
g140m = nirspec.G140M(order_overlap=overlap)
```

**msaexp convention.** msaexp tabulates `sensitivity_m` (count rate per µJy, with the order-$m$
dispersion already folded in) against the *intrinsic* wavelength $\lambda_1$ of the photons;
order-$m$ light of wavelength $\lambda_1$ lands on the pixel whose first-order wavelength is
$m\lambda_1$, and the pipeline divides the total count rate by the first-order sensitivity at the
observed wavelength. In $f_\nu$ this gives $F_\nu^{\rm obs}(\lambda) = F_\nu(\lambda) +
\sum_m S_m(\lambda/m)/S_1(\lambda)\,F_\nu(\lambda/m)$; converting to the $f_\lambda$ units used by
`unite` introduces a factor $m^{-2}$, so the helper returns

$$
r_m(\lambda) = \frac{S_m(\lambda/m)}{m^2\,S_1(\lambda)},
$$

linearly interpolated and zero outside the tabulated range. Pass `orders=(2,)` to ignore the
third order.

:::{note}
`order_overlap` is a runtime-only attribute: {class}`~unite.instrument.InstrumentConfig`
serialisation warns and drops it, so re-attach it after loading a configuration from YAML.
The overlap contribution is reported separately as
{attr}`SpectrumPrediction.overlap <unite.compute.SpectrumPrediction.overlap>` and as an
`order_overlap` column in {func}`~unite.results.make_spectra_tables`.
:::

---

### SDSS

The SDSS disperser computes $R(\lambda)$ from the `wdisp` column in the native SDSS pipeline
output. The disperser's resolution grid is populated at load time by {func}`~unite.spectrum.from_sdss_fits`.

```python
from unite.instrument import sdss

disperser = sdss.SDSSDisperser()
# Calibration tokens can be attached here as with any disperser
```

---

### DESI

DESI spectra are split across three spectrograph arms — B (3600–5800 Å), R (5760–7620 Å), and
Z (7520–9824 Å) — each saved as a separate set of HDUs in a single FITS file. Arms overlap by
~40–100 Å at both junctions; `unite`'s coverage-filtering handles the overlaps naturally when each
arm is a separate {class}`~unite.spectrum.Spectrum`.

The resolving power $R(\lambda)$ is derived at load time from the banded LSF resolution matrix
stored in the FITS file. Each pixel carries 11 diagonal-band weights; under the Gaussian LSF
assumption their second moment gives $\sigma$ in pixels, which is converted to FWHM and then to
$R = \lambda / \mathrm{FWHM}$.

```python
from unite.instrument import desi

b = desi.DESIDisperser('B')
r = desi.DESIDisperser('R')
z = desi.DESIDisperser('Z')
# Calibration tokens can be attached as with any disperser
```

---

### Generic Dispersers

For instruments without built-in support, `unite` provides two generic disperser classes.
Import them explicitly from `unite.instrument.generic`.

#### SimpleDisperser

{class}`~unite.instrument.generic.SimpleDisperser` defines the pixel scale from the input
wavelength array (one wavelength element per pixel). The resolution can be a constant or an array.

```python
from unite.instrument import generic
import numpy as np
from astropy import units as u

wavelength = np.linspace(4000, 9000, 500) * u.AA
disperser = generic.SimpleDisperser(
    wavelength=wavelength,
    R=3000.0,
    name='my_spectrograph',
)
```

Three modes are supported for specifying the dispersion:

| Mode | Argument | Description |
|------|----------|-------------|
| Resolving power | `R=value` | $R = \lambda / \Delta\lambda$ (constant or array) |
| Wavelength dispersion | `dlam=value` | $\Delta\lambda$ per pixel (constant or array) |
| Velocity dispersion | `dvel=value` | $\Delta v$ per pixel in km/s (constant or array) |

#### GenericDisperser

For maximum flexibility, pass callables that return $R(\lambda)$ and $d\lambda/\mathrm{pix}(\lambda)$:

```python
from unite.instrument import generic

def my_R(wavelength):
    return 2000 + 500 * (wavelength - 4000) / 5000

def my_dlam(wavelength):
    return wavelength / my_R(wavelength)

disperser = generic.GenericDisperser(R_func=my_R, dlam_dpix_func=my_dlam, unit=u.AA, name='custom')
```

---

### InstrumentConfig & Serialization

{class}`~unite.instrument.config.InstrumentConfig` wraps a set of dispersers for serialization
and named lookup:

```python
from unite.instrument.config import InstrumentConfig

dc = InstrumentConfig([prism, g395m])
```

Dispersers can be retrieved by name:

```python
d = dc['G395M']      # returns the G395M disperser
d = dc['PRISM']      # returns the PRISM disperser
names = dc.names     # ['PRISM', 'G395M']
```

This is particularly useful after loading a configuration from YAML:

```python
dc = InstrumentConfig.load('dispersers.yaml')
g395m = dc['G395M']  # retrieve with calibration tokens intact
```

`validate()` is called automatically on construction and warns if any calibration axis has **no
fixed anchor** — meaning all dispersers have a free parameter on that axis, making it
unidentifiable.

Instrument configs can be combined with addition:

```python
dc1 = InstrumentConfig.load('dispersers1.yaml')
dc2 = InstrumentConfig.load('dispersers2.yaml')
dc_combined = dc1 + dc2
```

---

## Loading Spectra

All spectrum loaders live in `unite.spectrum` and return a {class}`~unite.spectrum.Spectrum`
object. Loaders always require a configured disperser.

### from_edges

{func}`~unite.spectrum.from_edges` is the standard generic loader — it works with any disperser
and accepts {class}`astropy.units.Quantity` arrays for the pixel bin edges, flux, and error.
Arguments are positional.

```python
from unite.spectrum import from_edges
from astropy import units as u

flux_unit = 1e-20 * u.erg / (u.s * u.cm**2 * u.AA)

spectrum = from_edges(wl_edges[:-1], wl_edges[1:], flux_array * flux_unit, err_array * flux_unit, disperser)
```

Pass a boolean `mask` array (True = bad pixel) to exclude pixels before the spectrum is
constructed — masked pixels are removed entirely, not zeroed:

```python
bad = np.zeros(len(flux_array), dtype=bool)
bad[42] = True   # flag a known bad pixel

spectrum = from_edges(low, high, flux, err, disperser, mask=bad)
```

An optional `name` keyword overrides the disperser name on the resulting spectrum.

---

### from_centers

{func}`~unite.spectrum.from_centers` is for the common case where only pixel-centre wavelengths
are available.  Pixel bin edges are derived via {func}`numpy.gradient`.  Arguments are positional.

```python
from unite.spectrum import from_centers

spectrum = from_centers(wavelengths, flux, err, disperser)
```

```{note}
The gradient-based edge derivation is most accurate for gapless, reasonably uniform grids.
For spectra with detector gaps or strongly non-uniform spacing, supply `low` and `high`
directly via `from_edges`.
```

`from_centers` also accepts `mask` and `name` keyword arguments with the same semantics as
`from_edges`.

---

### from_arrays

{func}`~unite.spectrum.from_arrays` is a keyword-only interface that subsumes both of the above.
All arguments must be passed by name.  Supply either *center* or *low* + *high* — not both.

```python
from unite.spectrum import from_arrays

# Edge-based (equivalent to from_edges)
spectrum = from_arrays(low=low, high=high, flux=flux, error=err, disperser=disperser)

# Centre-based (equivalent to from_centers)
spectrum = from_arrays(center=wavelengths, flux=flux, error=err, disperser=disperser)

# With bad-pixel masking
spectrum = from_arrays(low=low, high=high, flux=flux, error=err, disperser=disperser, mask=bad)
```

This is useful when the wavelength representation is determined at runtime or when you want to
make the choice of representation explicit at the call site.

These three loaders are the right choice for:
- Simulated spectra or pipeline-reduced data in custom formats
- Any instrument not covered by `from_DJA`, `from_sdss_fits`, or `from_desi_fits`
- Quick testing with synthetic data

---

### from_DJA

{func}`~unite.spectrum.from_DJA` loads a NIRSpec spectrum in the format of the
[DAWN JWST Archive](https://dawn-cph.github.io/dja/index.html). It accepts any configured
disperser and can load directly from a URL (with optional `astropy` caching).

```python
from unite.instrument import nirspec
from unite.spectrum import from_DJA

g235h = nirspec.G235H()

spectrum = from_DJA('spectrum_g235h.fits', disperser=g235h)
spectrum = from_DJA('https://url_to_dja_spectrum.fits', disperser=g235h, cache=True)
```

---

### from_sdss_fits

{func}`~unite.spectrum.from_sdss_fits` loads an SDSS spectrum from a native pipeline FITS file
(or URL). It **requires** an {class}`~unite.instrument.sdss.SDSSDisperser` — the disperser's
$R(\lambda)$ grid is populated from the `wdisp` column in the file.

```python
from unite.instrument import sdss
from unite.spectrum import from_sdss_fits

disperser = sdss.SDSSDisperser()
spectrum = from_sdss_fits('spec-PLATE-MJD-FIBER.fits', disperser=disperser)
```

---

### from_desi_fits

{func}`~unite.spectrum.from_desi_fits` loads a DESI spectrum from a native coadd FITS file. It
returns a **list** of {class}`~unite.spectrum.Spectrum` objects — one per arm — which can be
passed directly to {class}`~unite.spectrum.Spectra`.

```python
from unite.spectrum import from_desi_fits, Spectra

spectra_list = from_desi_fits('spectra.fits')  # returns [b, r, z]
spectra = Spectra(spectra_list, redshift=z_sys)
```

Each arm's {class}`~unite.instrument.desi.DESIDisperser` is created automatically and its
$R(\lambda)$ grid is populated from the file's LSF resolution matrix.

**Selecting arms and spectra**

```python
# Load only the R and Z arms
spectra_list = from_desi_fits('spectra.fits', arms=['R', 'Z'])

# Load the second spectrum in the file (index=0 by default)
spectra_list = from_desi_fits('spectra.fits', index=1)
```

Arms absent from the file emit a {class}`UserWarning` and are skipped gracefully rather than
raising an error, so the same call works on files with partial arm coverage.

**Pre-constructing dispersers with calibration tokens**

To attach calibration tokens before loading, pre-construct the dispersers and pass them via the
`dispersers` argument. Arms not present in `dispersers` are created automatically.

```python
from unite.instrument import desi, FluxScale
from unite import prior

# B arm is the flux reference; R arm has a free flux scale
r_disp = desi.DESIDisperser('R',
    flux_scale=FluxScale(prior=prior.Uniform(0.8, 1.2))
)
spectra_list = from_desi_fits('spectra.fits', dispersers={'R': r_disp})
```

**Labelling spectra**

The optional `name` argument sets a base label; each arm is suffixed with its lower-case letter.

```python
spectra_list = from_desi_fits('spectra.fits', name='target')
# → spectrum names: 'target_b', 'target_r', 'target_z'
```

---

## Mixing Instruments

`unite` can combine spectra from entirely different instruments in one fit. Each spectrum uses its
own disperser; shared line tokens ensure the same physical parameters are used across all spectra.

```python
from unite.instrument import nirspec, sdss, FluxScale
from unite import prior
from unite.spectrum import Spectra, from_DJA, from_sdss_fits

# NIRSpec grating — flux reference, no FluxScale
g395h = nirspec.G395H()

# SDSS disperser with a free flux scale relative to NIRSpec
sdss_disp = sdss.SDSSDisperser(
    flux_scale=FluxScale(prior=prior.Uniform(0.5, 2.0))
)

spec_nir  = from_DJA('g395h.fits', disperser=g395h)
spec_sdss = from_sdss_fits('spec-PLATE-MJD-FIBER.fits', disperser=sdss_disp)

spectra = Spectra([spec_nir, spec_sdss], redshift=z_sys)
```
