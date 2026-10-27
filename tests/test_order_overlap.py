"""Tests for higher-order grating overlap (``Disperser.order_overlap``)."""

from __future__ import annotations

import math
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from astropy import units as u
from astropy.table import Table
from numpyro import handlers
from numpyro.infer.util import log_density

from unite import model, prior
from unite.compute import evaluate_model
from unite.continuum import ContinuumConfiguration, ContinuumRegion, PowerLaw
from unite.continuum.config import ContShape, NormWavelength, Scale
from unite.instrument import InstrumentConfig, nirspec
from unite.instrument.generic import GenericDisperser
from unite.instrument.nirspec import SensitivityRatio, order_overlap_from_msaexp
from unite.line import FWHM, Flux, LineConfiguration, Redshift
from unite.results import make_spectra_tables
from unite.spectrum import Spectra, from_arrays

erf = np.vectorize(math.erf)
C_KMS = 299792.458
FWHM_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))

# ---------------------------------------------------------------------------
# Toy G140M/F070LP-like scenario
# ---------------------------------------------------------------------------

Z_SYS = 0.22
LINE_REST_AA = 6563.0
LINE_FLUX = 0.02  # in flux_unit * um
LINE_FWHM_KMS = 400.0
PIX = 0.00064  # um, ~G140M dispersion
FLUX_UNIT = u.Unit('1e-20 erg s-1 cm-2 AA-1')
# Rest-frame continuum regions (Angstrom); the continuum is one power law
# F = (lam_obs / 1 um) ** CONT_BETA across both, so the regions join smoothly.
REGIONS = {'opt': (3646.0, 8204.0), 'nir': (8204.0, 20000.0)}
CONT_BETA = -1.5
R2_AMP = 0.1  # 10% second-order copy


def _region_params(lo, hi):
    """Power-law ``(scale, beta, norm_wav)`` of region [lo, hi] (Angstrom)."""
    nw_rest = 0.5 * (lo + hi)
    nw_obs = nw_rest * 1e-4 * (1 + Z_SYS)
    return nw_obs**CONT_BETA, CONT_BETA, nw_rest


def _r2(w):
    """Smooth second-order throughput ratio: 0 below 1.3 um, 0.1 above 1.5 um."""
    return R2_AMP * jnp.clip((w - 1.3) / 0.2, 0.0, 1.0)


def _pixel_gaussian(low, high, center, sigma, area):
    """Pixel-averaged Gaussian of integrated flux *area* over [low, high]."""
    cdf = lambda x: 0.5 * (1.0 + erf((x - center) / (math.sqrt(2.0) * sigma)))  # noqa: E731
    return area * (cdf(high) - cdf(low)) / (high - low)


def _continuum(wl_obs):
    """Observed-frame f_lambda continuum of the toy source at wavelengths (um)."""
    out = np.zeros_like(wl_obs)
    for lo, hi in REGIONS.values():
        lo_obs, hi_obs = lo * 1e-4 * (1 + Z_SYS), hi * 1e-4 * (1 + Z_SYS)
        scale, beta, nw_rest = _region_params(lo, hi)
        nw = nw_rest * 1e-4 * (1 + Z_SYS)
        inside = (wl_obs >= lo_obs) & (wl_obs <= hi_obs)
        out = np.where(inside, scale * (wl_obs / nw) ** beta, out)
    return out


def _make_toy(disperser, *, with_second_order=True):
    """Toy spectrum: continuum + line, plus a 10% second-order image if requested.

    The second-order image of light of wavelength ``lam1`` lands at
    ``2 * lam1`` with ``R_2(lam1) = 2 R(lam1)``: in observed wavelength its
    LSF FWHM is the first-order LSF of the photon wavelength,
    ``lam1 / R(lam1)``.
    """
    low = np.arange(0.70, 1.80, PIX)
    high = low + PIX
    wl = 0.5 * (low + high)
    lam_c = LINE_REST_AA * 1e-4 * (1 + Z_SYS)

    def _line_image(order):
        c = order * lam_c
        lsf = lam_c / float(disperser.R(lam_c))  # first-order LSF of the photons
        sigma = math.hypot(c * LINE_FWHM_KMS / C_KMS, lsf) * FWHM_TO_SIGMA
        # F_1(lam / order) integrates to order * LINE_FLUX over observed lam.
        return _pixel_gaussian(low, high, c, sigma, order * LINE_FLUX)

    flux = _continuum(wl) + _line_image(1)
    if with_second_order:
        flux = flux + np.asarray(_r2(wl)) * (_continuum(wl / 2) + _line_image(2))
    err = np.full_like(flux, 0.01)
    return from_arrays(
        low=low * u.um,
        high=high * u.um,
        flux=flux * FLUX_UNIT,
        error=err * FLUX_UNIT,
        disperser=disperser,
        name='g140m',
    )


def _build(disperser, *, with_second_order=True, integration_mode='analytic'):
    """Build the toy model with every parameter fixed at the truth."""
    spec = _make_toy(disperser, with_second_order=with_second_order)
    spectra = Spectra([spec], redshift=Z_SYS)
    spectra.line_scale = 1.0 * FLUX_UNIT * u.um
    spectra.continuum_scale = 1.0 * FLUX_UNIT

    lc = LineConfiguration()
    lc.add_line(
        'Ha',
        LINE_REST_AA * u.AA,
        flux=Flux('Ha', prior=prior.Fixed(LINE_FLUX)),
        fwhm_gauss=FWHM('c', prior=prior.Fixed(LINE_FWHM_KMS)),
        redshift=Redshift('c', prior=prior.Fixed(0.0)),
    )
    regions = []
    for name, (lo, hi) in REGIONS.items():
        scale, beta, nw_rest = _region_params(lo, hi)
        regions.append(
            ContinuumRegion(
                lo * u.AA,
                hi * u.AA,
                form=PowerLaw(),
                name=name,
                params={
                    'scale': Scale(prior=prior.Fixed(scale)),
                    'beta': ContShape(prior=prior.Fixed(beta)),
                    'norm_wav': NormWavelength(prior=prior.Fixed(nw_rest)),
                },
            )
        )
    cc = ContinuumConfiguration(regions)
    spectra.prepare(lc, cc, drop_empty_regions=False)
    builder = model.ModelBuilder(
        spectra.prepared_line_config, spectra.prepared_cont_config, spectra
    )
    model_fn, args = builder.build(integration_mode=integration_mode)
    # build() trims spectra to the continuum regions; return the trimmed copy.
    return model_fn, args, args.spectra[0]


def _overlap_region(spec):
    wl = np.asarray(spec.wavelength)
    return wl > 1.3  # _r2 is zero at and below 1.3 um


# ---------------------------------------------------------------------------
# Disperser attribute
# ---------------------------------------------------------------------------


class TestDisperserAttribute:
    def test_default_empty(self):
        assert nirspec.G140M().order_overlap == {}
        assert nirspec.NIRSpec('g140m').order_overlap == {}

    @pytest.mark.parametrize(
        'cls',
        [
            nirspec.G140M,
            nirspec.G140H,
            nirspec.G235M,
            nirspec.G235H,
            nirspec.G395M,
            nirspec.G395H,
            nirspec.PRISM,
        ],
    )
    def test_subclasses_accept_kwarg(self, cls):
        d = cls(order_overlap={3: _r2, 2: _r2})
        assert list(d.order_overlap) == [2, 3]  # sorted copy
        assert d.order_overlap[2] is _r2

    def test_generic_disperser_sets_attribute(self):
        d = GenericDisperser(
            R_func=lambda w: jnp.full_like(w, 1000.0),
            dlam_dpix_func=lambda w: w / 1000.0,
            unit=u.um,
        )
        assert d.order_overlap == {}

    def test_validation(self):
        with pytest.raises(TypeError, match='keys must be integers'):
            nirspec.G140M(order_overlap={'2': _r2})
        with pytest.raises(ValueError, match='>= 2'):
            nirspec.G140M(order_overlap={1: _r2})
        with pytest.raises(TypeError, match='callable'):
            nirspec.G140M(order_overlap={2: 0.1})

    def test_str_lists_orders(self):
        d = nirspec.G140M(order_overlap={2: _r2})
        assert 'Order Overlap: 2' in str(d)
        assert 'Order Overlap' not in str(nirspec.G140M())

    def test_serialisation_warns_and_drops(self):
        d = nirspec.G140M(order_overlap={2: _r2})
        cfg = InstrumentConfig([d])
        with pytest.warns(UserWarning, match='runtime-only'):
            data = cfg.to_dict()
        cfg2 = InstrumentConfig.from_dict(data)
        assert cfg2['G140M'].order_overlap == {}
        # Without overlap no warning is emitted.
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            InstrumentConfig([nirspec.G140M()]).to_dict()


# ---------------------------------------------------------------------------
# msaexp sensitivity helper
# ---------------------------------------------------------------------------


def _synthetic_table(*, with_third=True, with_nan=False):
    """msaexp-like table: known analytic first/second/third-order curves."""
    wl = np.linspace(0.6, 3.3, 2701)
    s1 = np.where(wl >= 0.66, 1.0 + 0.5 * wl, 0.0)
    s2 = np.where((wl >= 0.64) & (wl <= 1.6), 0.4 * np.exp(-(wl - 0.7) / 0.3), 0.0)
    t = Table({'wavelength': wl, 'sensitivity': s1, 'sensitivity_2': s2})
    if with_third:
        s3 = np.where((wl >= 0.68) & (wl <= 1.07), 0.02, 0.0)
        t['sensitivity_3'] = s3
    if with_nan:
        t['sensitivity_2'][1000] = np.nan
    return t


class TestSensitivityHelper:
    def test_ratio_values_follow_msaexp_convention(self):
        t = _synthetic_table()
        overlap = order_overlap_from_msaexp(t)
        assert sorted(overlap) == [2, 3]
        r2 = overlap[2]
        assert isinstance(r2, SensitivityRatio)
        assert r2.order == 2
        wl = np.asarray(t['wavelength'])
        s1, s2 = np.asarray(t['sensitivity']), np.asarray(t['sensitivity_2'])
        # r_2(lam_obs) = S_2(lam_obs / 2) / (2**2 * S_1(lam_obs)).
        for lam_obs in (1.4, 1.6, 2.0, 2.8):
            expected = np.interp(lam_obs / 2, wl, s2) / (4 * np.interp(lam_obs, wl, s1))
            np.testing.assert_allclose(float(r2(lam_obs)), expected, rtol=1e-6)
        # Third order: S_3(lam_obs / 3) / (9 S_1(lam_obs)).
        r3 = overlap[3]
        s3 = np.asarray(t['sensitivity_3'])
        lam_obs = 2.4
        expected = np.interp(lam_obs / 3, wl, s3) / (9 * np.interp(lam_obs, wl, s1))
        np.testing.assert_allclose(float(r3(lam_obs)), expected, rtol=1e-6)
        assert expected > 0

    def test_zero_outside_support(self):
        overlap = order_overlap_from_msaexp(_synthetic_table())
        r2, r3 = overlap[2], overlap[3]
        # Second order lands at 2 * [0.64, 1.6] = [1.28, 3.2].
        assert float(r2(1.0)) == 0.0
        assert float(r2(1.27)) == 0.0
        assert float(r2(1.5)) > 0.0
        assert float(r2(3.25)) == 0.0
        assert float(r2(10.0)) == 0.0
        lo, hi = r2.support
        assert 1.28 <= lo <= 1.33 and 3.15 <= hi <= 3.2
        # Third order lands at 3 * [0.68, 1.07] = [2.04, 3.21].
        assert float(r3(2.0)) == 0.0
        assert float(r3(2.5)) > 0.0
        assert float(r3(3.3)) == 0.0
        # Vectorised and jittable.
        w = jnp.linspace(0.5, 4.0, 50)
        out = jax.jit(r2)(w)
        assert out.shape == w.shape
        assert np.all(np.asarray(out) >= 0)

    def test_orders_selection_and_missing_columns(self):
        t = _synthetic_table(with_third=False)
        assert sorted(order_overlap_from_msaexp(t)) == [2]
        assert sorted(order_overlap_from_msaexp(t, orders=(2,))) == [2]
        with pytest.raises(KeyError, match='sensitivity_3'):
            order_overlap_from_msaexp(t, orders=(2, 3), strict=True)
        with pytest.raises(ValueError, match='>= 2'):
            order_overlap_from_msaexp(t, orders=(1,))
        with pytest.raises(KeyError, match='sensitivity'):
            order_overlap_from_msaexp(Table({'wavelength': [1.0, 2.0]}))

    def test_nan_entries_are_zero(self):
        t = _synthetic_table(with_nan=True)
        r2 = order_overlap_from_msaexp(t)[2]
        lam_nan = 2 * float(t['wavelength'][1000])
        assert float(r2(lam_nan)) == 0.0
        assert np.all(np.isfinite(np.asarray(r2.ratio)))

    def test_reads_fits_file(self, tmp_path):
        path = tmp_path / 'msaexp_sensitivity_toy_001.fits'
        _synthetic_table().write(path)
        overlap = order_overlap_from_msaexp(path)
        assert sorted(overlap) == [2, 3]
        assert 'order=2' in repr(overlap[2])

    def test_attaches_to_nirspec(self):
        overlap = order_overlap_from_msaexp(_synthetic_table())
        d = nirspec.G140M(order_overlap=overlap)
        assert d.order_overlap[2] is overlap[2]


# ---------------------------------------------------------------------------
# Model behaviour
# ---------------------------------------------------------------------------


class TestOverlapModel:
    def test_regression_reproduces_injected_second_order(self):
        """With order_overlap the model matches the toy data; without it, it does not."""
        d_on = nirspec.G140M(order_overlap={2: _r2})
        _, args_on, spec = _build(d_on)
        pred_on = evaluate_model({}, args_on)[0]
        data = np.asarray(spec.flux)
        region = _overlap_region(spec)
        scale = np.max(data)
        resid_on = np.abs(pred_on.total[0] - data) / scale
        assert resid_on[region].max() < 1e-3
        assert resid_on.max() < 1e-3

        d_off = nirspec.G140M()
        _, args_off, _ = _build(d_off)  # same data, no overlap term
        pred_off = evaluate_model({}, args_off)[0]
        resid_off = np.abs(pred_off.total[0] - data) / scale
        assert resid_off[region].max() > 1e-2
        # Outside the overlap region both models agree with the data.
        assert resid_off[~region].max() < 1e-3

        # The overlap component is reported separately and sums correctly.
        assert pred_on.overlap is not None
        assert pred_off.overlap is None
        parts = sum(pred_on.lines.values()) + sum(pred_on.continuum_regions.values())
        np.testing.assert_allclose(parts + pred_on.overlap, pred_on.total, rtol=1e-10)
        assert np.all(pred_on.overlap[0][~region] == 0.0)
        assert np.all(pred_on.overlap[0][region] > 0.0)

    @staticmethod
    def _second_order_image_sigma(disperser, integration_mode='analytic'):
        """Gaussian sigma (observed um) of the 2nd-order line image."""
        _, args, spec = _build(disperser, integration_mode=integration_mode)
        pred = evaluate_model({}, args)[0]
        wl = np.asarray(spec.wavelength)
        lam_c2 = 2 * LINE_REST_AA * 1e-4 * (1 + Z_SYS)
        window = np.abs(wl - lam_c2) < 0.01
        image = pred.overlap[0][window] - np.median(pred.overlap[0][window])
        image = np.clip(image, 0, None)
        x = wl[window]
        mean = np.sum(x * image) / np.sum(image)
        return math.sqrt(np.sum((x - mean) ** 2 * image) / np.sum(image))

    def test_second_order_line_image_resolution(self):
        """A 2nd-order line image has R_2(lam1) = 2 R(lam1) at the photon wavelength.

        In observed um its LSF FWHM is the first-order LSF of the photon
        wavelength lam1; reusing the LSF-convolved 1st-order model at lam/2
        would make it twice too broad.
        """
        d_on = nirspec.G140M(order_overlap={2: _r2})
        sigma = self._second_order_image_sigma(d_on)
        lam_c = LINE_REST_AA * 1e-4 * (1 + Z_SYS)
        lsf = lam_c / float(d_on.R(lam_c))
        expected = math.hypot(2 * lam_c * LINE_FWHM_KMS / C_KMS, lsf) * FWHM_TO_SIGMA
        np.testing.assert_allclose(sigma, expected, rtol=0.05)

    @pytest.mark.parametrize('integration_mode', ['analytic', 'convolution'])
    def test_resolution_uses_photon_wavelength(self, integration_mode):
        """With constant R, the 2nd-order LSF is R(lam1), not R(2 lam1), in observed um.

        For gratings R(2 lam1) ~ 2 R(lam1), so evaluating the first-order
        LSF at the landing pixel would be nearly the same; with a constant R
        it is twice as broad, which this test distinguishes.
        """
        r_const = 1000.0
        d = GenericDisperser(
            R_func=lambda w: jnp.full_like(jnp.asarray(w, dtype=float), r_const),
            dlam_dpix_func=lambda w: w / 1000.0,
            unit=u.um,
            name='g140m',
        )
        d.order_overlap = {2: _r2}
        sigma = self._second_order_image_sigma(d, integration_mode)
        lam_c = LINE_REST_AA * 1e-4 * (1 + Z_SYS)
        vel = 2 * lam_c * LINE_FWHM_KMS / C_KMS
        expected = math.hypot(vel, lam_c / r_const) * FWHM_TO_SIGMA
        landing = math.hypot(vel, 2 * lam_c / r_const) * FWHM_TO_SIGMA
        assert landing / expected > 1.15
        np.testing.assert_allclose(sigma, expected, rtol=0.02)

    def test_analytic_and_convolution_agree(self):
        d = nirspec.G140M(order_overlap={2: _r2})
        _, args_a, spec = _build(d, integration_mode='analytic')
        _, args_c, _ = _build(d, integration_mode='convolution')
        pred_a = evaluate_model({}, args_a)[0]
        pred_c = evaluate_model({}, args_c)[0]
        interior = slice(10, -10)
        np.testing.assert_allclose(
            pred_a.total[0, interior], pred_c.total[0, interior], rtol=2e-3
        )
        np.testing.assert_allclose(
            pred_a.overlap[0, interior],
            pred_c.overlap[0, interior],
            rtol=2e-3,
            atol=2e-3 * np.max(pred_a.overlap),
        )
        # And the numpyro model functions (likelihood path) agree as well.
        model_fn_a, _, _ = _build(d, integration_mode='analytic')
        obs_a = handlers.trace(handlers.seed(model_fn_a, 0)).get_trace(args_a)
        obs_c = handlers.trace(handlers.seed(model_fn_a, 0)).get_trace(args_c)
        site = f'obs_{spec.name}'
        mu_a = np.asarray(obs_a[site]['fn'].loc)
        mu_c = np.asarray(obs_c[site]['fn'].loc)
        np.testing.assert_allclose(mu_a[interior], mu_c[interior], rtol=2e-3)

    @pytest.mark.parametrize('integration_mode', ['analytic', 'convolution'])
    def test_model_matches_evaluate_model(self, integration_mode):
        """The likelihood mean in unite_model equals evaluate_model's total."""
        d = nirspec.G140M(order_overlap={2: _r2})
        model_fn, args, spec = _build(d, integration_mode=integration_mode)
        tr = handlers.trace(handlers.seed(model_fn, 0)).get_trace(args)
        mu = np.asarray(tr[f'obs_{spec.name}']['fn'].loc) * args.norm_factors[0]
        pred = evaluate_model({}, args)[0]
        # model.py integrates the continuum over pixels while compute.py
        # evaluates it at pixel centres, hence the loose tolerance.
        np.testing.assert_allclose(mu, pred.total[0], rtol=1e-3)

    def test_spectra_table_has_overlap_column(self):
        d = nirspec.G140M(order_overlap={2: _r2})
        _, args, spec = _build(d)
        tables = make_spectra_tables({}, args)
        t = tables[spec.name]
        assert 'order_overlap' in t.colnames
        assert t['order_overlap'].unit == FLUX_UNIT
        pct = make_spectra_tables({}, args, percentiles=np.array([0.5]))[spec.name]
        assert pct['order_overlap'].shape == (len(pct), 1)
        _, args_off, _ = _build(nirspec.G140M())
        assert (
            'order_overlap' not in make_spectra_tables({}, args_off)[spec.name].colnames
        )

    def test_without_overlap_matches_zero_ratio_exactly(self):
        """No order_overlap and an all-zero ratio give identical outputs."""
        for mode in ('analytic', 'convolution'):
            _, args_off, _ = _build(nirspec.G140M(), integration_mode=mode)
            zero = nirspec.G140M(order_overlap={2: lambda w: jnp.zeros_like(w)})
            _, args_zero, _ = _build(zero, integration_mode=mode)
            pred_off = evaluate_model({}, args_off)[0]
            pred_zero = evaluate_model({}, args_zero)[0]
            assert np.array_equal(pred_off.total, pred_zero.total)
            assert pred_off.overlap is None
            assert np.all(pred_zero.overlap == 0.0)

    def test_sampled_model_is_differentiable(self):
        """With free parameters and calibration tokens the overlap term is jittable/differentiable."""
        from unite.instrument import FluxScale, PixOffset, RScale

        d = nirspec.G140M(
            order_overlap={2: _r2},
            r_scale=RScale(prior=prior.Uniform(0.8, 1.2)),
            flux_scale=FluxScale(prior=prior.Uniform(0.5, 1.5)),
            pix_offset=PixOffset(prior=prior.Uniform(-1.0, 1.0)),
        )
        spec = _make_toy(d)
        spectra = Spectra([spec], redshift=Z_SYS)
        lc = LineConfiguration()
        lc.add_line(
            'Ha',
            LINE_REST_AA * u.AA,
            flux=Flux('Ha', prior=prior.Uniform(0, 1)),
            fwhm_gauss=FWHM('c', prior=prior.Uniform(100, 1000)),
            redshift=Redshift('c', prior=prior.Uniform(-0.002, 0.002)),
        )
        cc = ContinuumConfiguration(
            [
                ContinuumRegion(lo * u.AA, hi * u.AA, form=PowerLaw(), name=name)
                for name, (lo, hi) in REGIONS.items()
            ]
        )
        spectra.prepare(lc, cc, drop_empty_regions=False)
        spectra.compute_scales(
            spectra.prepared_line_config, spectra.prepared_cont_config
        )
        model_fn, args = model.ModelBuilder(
            spectra.prepared_line_config, spectra.prepared_cont_config, spectra
        ).build()
        tr = handlers.trace(handlers.seed(model_fn, 0)).get_trace(args)
        params = {
            k: v['value']
            for k, v in tr.items()
            if v['type'] == 'sample' and not v['is_observed']
        }
        assert {'r_scale_G140M', 'flux_scale_G140M', 'pix_offset_G140M'} <= set(params)

        def _ld(p):
            return log_density(model_fn, (args,), {}, p)[0]

        ld, grad = jax.jit(jax.value_and_grad(_ld))(params)
        assert np.isfinite(float(ld))
        assert all(np.isfinite(float(g)) for g in grad.values())
