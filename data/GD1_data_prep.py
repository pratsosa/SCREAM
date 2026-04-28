"""
GD1_data_prep.py — Phase 2 of the GD-1 Gaia x DECaLS data pipeline.

Loads the HDF5 output from GD1_download.py, applies extinction corrections,
computes DECaLS magnitudes, crossmatches with StreamFinder labels, converts
to stream coordinates, applies quality cuts, defines the signal region,
and saves a cleaned FITS file.

Run from SCREAM/data/:
    python GD1_data_prep.py
"""

import logging
import numpy as np
from astropy.table import Table, unique as table_unique
from astropy.io import fits
import astropy.units as u
import galstreams

from adql_utils import ra_dec_to_phi1_phi2, pmra_pmdec_to_pmphi12
from transforms import mw_transmission_ls, flux_to_mag_ls

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

# Ignore WARNING: AstropyDeprecationWarning
import warnings
from astropy.utils.exceptions import AstropyDeprecationWarning
warnings.filterwarnings('ignore', category=AstropyDeprecationWarning)


HDF5_PATH = '/pscratch/sd/p/pratsosa/fetcher_output_new/GD-1-I21/GD-1-I21_matched.hdf5'
SF_PATH = '/pscratch/sd/p/pratsosa/general_stream_data/streamfinder_gaiadr3.fits'
OUTPUT_PATH = '/pscratch/sd/p/pratsosa/GD-1_gaia_x_decals_040726.fits'
STREAM_TRACK = 'GD-1-I21'

# ── 2.1 Load HDF5 ──────────────────────────────────────────────────────────────
log.info(f"Loading HDF5 from {HDF5_PATH}")
t = Table.read(HDF5_PATH, path='data')
log.info(f"Loaded {len(t)} rows")

# ── 2.2 Drop stars with no LS crossmatch ───────────────────────────────────────
n_before = len(t)
t = t[~np.isnan(t['match_dist_arcsec'])]
log.info(f"After LS crossmatch filter: {len(t)} rows ({n_before - len(t)} dropped, no LS counterpart)")

# ── 2.3 Compute LS flux errors and convert positional errors ──────────────────
# Drop rows with zero or negative inverse-variance (would produce inf flux errors).
n_before = len(t)
ivar_mask = (
    (np.array(t['ls_flux_ivar_g']) > 0)
    & (np.array(t['ls_flux_ivar_r']) > 0)
    & (np.array(t['ls_flux_ivar_z']) > 0)
)
t = t[ivar_mask]
log.info(f"After ivar validity filter: {len(t)} rows ({n_before - len(t)} dropped, zero/negative ivar)")

# Flux errors from inverse-variance weights.
t['flux_err_g'] = 1.0 / np.sqrt(t['ls_flux_ivar_g'])
t['flux_err_r'] = 1.0 / np.sqrt(t['ls_flux_ivar_r'])
t['flux_err_z'] = 1.0 / np.sqrt(t['ls_flux_ivar_z'])

# ra_error and dec_error arrive from Gaia DR3 in milliarcseconds; convert to degrees.
t['ra_error']  = (np.array(t['ra_error'])  * u.mas).to(u.deg).value
t['dec_error'] = (np.array(t['dec_error']) * u.mas).to(u.deg).value

# ── 2.4 Compute DECaLS magnitudes from fluxes ──────────────────────────────────
# DECaLS fluxes are in nanomaggies; flux ≤ 0 yields inf/nan (handled in Cut A).
t['gmag'] = 22.5 - 2.5 * np.log10(t['ls_flux_g'])
t['rmag'] = 22.5 - 2.5 * np.log10(t['ls_flux_r'])
t['zmag'] = 22.5 - 2.5 * np.log10(t['ls_flux_z'])

# ── 2.5 Apply DECaLS extinction correction ─────────────────────────────────────
# Uses mw_transmission_ls from transforms.py (coefficients g=3.214, r=2.165, z=1.211),
# consistent with photometry.py and shared_step.
t['EBV'] = t['ls_ebv']   # temporary alias expected by mw_transmission_ls
trans_g = mw_transmission_ls(t, "g")
trans_r = mw_transmission_ls(t, "r")
trans_z = mw_transmission_ls(t, "z")
t['gmag0'] = flux_to_mag_ls(np.array(t['ls_flux_g']) / trans_g)
t['rmag0'] = flux_to_mag_ls(np.array(t['ls_flux_r']) / trans_r)
t['zmag0'] = flux_to_mag_ls(np.array(t['ls_flux_z']) / trans_z)
t.remove_column('EBV')


# ── 2.6 StreamFinder crossmatch (Gaia source_id) ──────────────────────────────
log.info(f"Loading StreamFinder catalog from {SF_PATH}")
sf = Table(fits.getdata(SF_PATH))
stream_ids = np.array(sf['Gaia'][sf['Stream'] == 53])

common, idx_t, _ = np.intersect1d(np.array(t['source_id']), stream_ids, return_indices=True)
stream_col = np.zeros(len(t), dtype=bool)
stream_col[idx_t] = True
t['stream'] = stream_col
log.info(f"StreamFinder GD-1 members matched: {stream_col.sum()} / {len(t)}")

# ── 2.7 Coordinate transformation (stream frame) ──────────────────────────────
log.info("Computing stream-frame coordinates via galstreams")
mws = galstreams.MWStreams(verbose=False, implement_Off=True)
stream_frame = mws[STREAM_TRACK].stream_frame

phi1, phi2 = ra_dec_to_phi1_phi2(stream_frame, t['ra'] * u.deg, t['dec'] * u.deg)
pm_phi1, pm_phi2 = pmra_pmdec_to_pmphi12(
    stream_frame,
    t['ra'] * u.deg, t['dec'] * u.deg,
    t['pmra'] * u.mas / u.yr, t['pmdec'] * u.mas / u.yr
)
t['phi1'] = phi1
t['phi2'] = phi2
t['pm_phi1'] = pm_phi1
t['pm_phi2'] = pm_phi2

# ── 2.8 Quality cuts ───────────────────────────────────────────────────────────

# Cut A — Invalid DECaLS magnitudes (flux ≤ 0 or NaN)
grz_mask = ~(
    np.isinf(t['gmag0']) | np.isinf(t['rmag0']) | np.isinf(t['zmag0'])
    | np.isnan(t['gmag0']) | np.isnan(t['rmag0']) | np.isnan(t['zmag0'])
)
n_before = len(t)
t = t[grz_mask]
log.info(f"Cut A (invalid DECaLS mags): {len(t)} rows remaining ({n_before - len(t)} dropped)")
log.info(f"  Stream stars after Cut A: {t['stream'].sum()}")

# Cut B — Parallax cut (distance-based)
distance_mid = mws.summary.loc[STREAM_TRACK, 'distance_mid']
par_mask = (np.array(t['parallax']) - 3 * np.abs(np.array(t['parallax_error']))) < 1.0 / distance_mid
n_before = len(t)
t = t[par_mask]
log.info(f"Cut B (parallax, distance_mid={distance_mid:.2f} kpc): {len(t)} rows remaining ({n_before - len(t)} dropped)")
log.info(f"  Stream stars after Cut B: {t['stream'].sum()}")

# Cut C — Color and magnitude cuts (main sequence selection)
color = t['phot_bp_mean_mag'] - t['phot_rp_mean_mag']
color_mask = (color > 0.0) & (color < 1.0)
g_mask = t['phot_g_mean_mag'] < 20.2
n_before = len(t)
t = t[color_mask & g_mask]
log.info(f"Cut C (color/mag): {len(t)} rows remaining ({n_before - len(t)} dropped)")
log.info(f"  Stream stars after Cut C: {t['stream'].sum()}")

# Cut D — Remove NaNs in all feature columns
feature_cols = [
    'phi1', 'phi2', 'pm_phi1', 'pm_phi2',
    'pmra_error', 'pmdec_error',
    'phot_g_mean_mag', 'bp_rp',
    'parallax', 'parallax_error',
    'gmag0', 'rmag0', 'zmag0',
    'ra', 'dec',
    'phot_g_mean_flux_error', 'phot_bp_mean_flux_error', 'phot_rp_mean_flux_error',
    'flux_err_g', 'flux_err_r', 'flux_err_z',
    'ra_error', 'dec_error',
    'ls_ebv',
]
nan_mask = np.ones(len(t), dtype=bool)
for col in feature_cols:
    nan_mask &= ~np.isnan(np.array(t[col]).astype(float))
n_before = len(t)
t = t[nan_mask]
log.info(f"Cut D (NaN removal): {len(t)} rows remaining ({n_before - len(t)} dropped)")
log.info(f"  Stream stars after Cut D: {t['stream'].sum()}")

# ── 2.9 Deduplicate on source_id ──────────────────────────────────────────────
n_before = len(t)
t = table_unique(t, keys='source_id', keep='first')
log.info(f"After deduplication: {len(t)} rows ({n_before - len(t)} duplicates removed)")

# ── 2.10 Define signal region (pm_phi1) ───────────────────────────────────────
pm_phi1_arr = np.array(t['pm_phi1'])
stream_arr = np.array(t['stream'])
stream_pm_phi1 = pm_phi1_arr[stream_arr]

sig_low = np.percentile(stream_pm_phi1, 5)
sig_high = np.percentile(stream_pm_phi1, 95)
signal_region = (pm_phi1_arr > sig_low) & (pm_phi1_arr < sig_high)
t['signal_region'] = signal_region

log.info(f"Signal region pm_phi1: [{sig_low:.3f}, {sig_high:.3f}] mas/yr")
log.info(f"Stars in signal region: {signal_region.sum()} / {len(t)} ({signal_region.sum()/len(t)*100:.1f}%)")

# ── 2.11 Save output ───────────────────────────────────────────────────────────
t.write(OUTPUT_PATH, overwrite=True)
log.info(f"Saved {len(t)} rows to {OUTPUT_PATH}")
log.info(f"Stream stars: {t['stream'].sum()} | Signal region: {t['signal_region'].sum()}")
