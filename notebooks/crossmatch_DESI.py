# %% [markdown]
# # DESI Crossmatch
# Loads GD-1 Gaia×DECaLS catalog, joins model predictions, crossmatches with
# DESI VRAD measurements, and saves the result to a single FITS file.
# Run this once; validation_figure.py reads the output directly.

# %%
import numpy as np
from astropy.table import Table
from astropy import table
import pandas as pd

# %% [markdown]
# ## Constants

# %%
DATA_PATH      = '/pscratch/sd/p/XXXXa'
DESI_DATA_PATH = '/pscratch/sd/p/XXXXa/DESI_data'
RESULTS_PATH   = '/pscratch/sd/p/XXXXa/GD1_SCREAM/results'
OUTPUT_PATH    = '/pscratch/sd/p/XXXXa/GD-1_gaia_x_decals_VRAD2.fits'

# %% [markdown]
# ## Step 1 — Core catalog
# Load GD-1 Gaia×DECaLS fits and join model predictions on source_id.

# %%
cat = Table.read(f'{DATA_PATH}/GD-1_gaia_x_decals_040726.fits')

# preds = pd.read_csv(f'{RESULTS_PATH}/cv_test_cv_predictions.csv')
preds = pd.read_csv(f'{RESULTS_PATH}/cv2_hearty_sweep_cv_predictions.csv')
# CSV source_id columns default to float64; coerce before joining.
preds['source_id'] = preds['source_id'].astype(np.int64)
preds = Table.from_pandas(preds)

cat = table.join(cat, preds, keys='source_id')

# %% [markdown]
# ## Step 2 — DESI VRAD + stream/background labels
# Loads mwsall-pix-iron.fits (RVTAB + GAIA extensions), deduplicates by
# lowest VRAD_ERR per source, then merges with XXXX's GD1_DESI_memprob.fits.
# XXXX's entries take priority for both VRAD and stream membership labels.
# The combined catalog is left-joined onto cat on source_id.

# %%
# --- 2a: mwsall VRAD ---
DR1_DESI_rvtab = Table.read(
    f'{DESI_DATA_PATH}/mwsall-pix-iron.fits', hdu='RVTAB', mask_invalid=False)
DR1_DESI_gaia  = Table.read(
    f'{DESI_DATA_PATH}/mwsall-pix-iron.fits', hdu='GAIA',  mask_invalid=False)

# The two extensions are row-aligned; direct column assignment is safe.
DR1_DESI_rvtab['source_id'] = DR1_DESI_gaia['SOURCE_ID'].astype(np.int64)

DR1_DESI_cat = DR1_DESI_rvtab[['source_id', 'VRAD', 'VRAD_ERR']]

# Drop masked / invalid source_ids; keep best (lowest VRAD_ERR) per source.
desi_df = DR1_DESI_cat.to_pandas()
desi_df = desi_df.dropna(subset=['source_id'])
desi_df = desi_df[desi_df['source_id'] != 0]
desi_df['source_id'] = desi_df['source_id'].astype(np.int64)
desi_df = desi_df.sort_values('VRAD_ERR', ascending=True)
desi_df = desi_df.drop_duplicates(subset=['source_id'], keep='first')
print(f'{len(desi_df)} unique Gaia sources with valid DESI VRAD measurements')

# --- 2b: XXXX's catalog ---
XXXX = Table.read(f'{DESI_DATA_PATH}/GD1_DESI_memprob.fits')
XXXX = XXXX[['SOURCE_ID', 'VRAD', 'VRAD_ERR', 'p_stream', 'p_cocoon']]
XXXX.rename_column('SOURCE_ID', 'source_id')
XXXX_df = XXXX.to_pandas()
XXXX_df['source_id'] = XXXX_df['source_id'].astype(np.int64)
# Deduplicate XXXX's catalog by best VRAD_ERR, just in case.
XXXX_df = XXXX_df.sort_values('VRAD_ERR', ascending=True)
XXXX_df = XXXX_df.drop_duplicates(subset=['source_id'], keep='first')
XXXX_df['desi_label'] = (XXXX_df['p_stream'] + XXXX_df['p_cocoon'] >= 0.5).astype(int)
print(f'{len(XXXX_df)} sources in XXXX\'s GD1_DESI_memprob catalog')
print(f'{XXXX_df["desi_label"].sum()} stream members (p_stream + p_cocoon >= 0.5)')

# --- 2c: Combine — XXXX first, mwsall fills in the rest ---
# mwsall entries not in XXXX are labelled background (0); p_stream/p_cocoon are NaN.
mwsall_only_df = desi_df[~desi_df['source_id'].isin(XXXX_df['source_id'])].copy()
mwsall_only_df['p_stream']   = np.nan
mwsall_only_df['p_cocoon']   = np.nan
mwsall_only_df['desi_label'] = 0

combined_df = pd.concat([XXXX_df, mwsall_only_df], ignore_index=True)
desi_combined = Table.from_pandas(combined_df)
print(f'{len(desi_combined)} total sources in combined DESI catalog')

# --- 2d: Left-join onto cat ---
cat = table.join(cat, desi_combined, keys='source_id', join_type='left')
cat.rename_column('VRAD',     'DESI_VRAD')
cat.rename_column('VRAD_ERR', 'DESI_VRAD_ERR')

# --- 2e: Unmask columns (masked = star not observed by DESI at all) ---
def _unmask_float(col):
    return np.where(np.ma.getmaskarray(col), np.nan, np.ma.getdata(col).astype(float))

cat['DESI_VRAD']     = _unmask_float(cat['DESI_VRAD'])
cat['DESI_VRAD_ERR'] = _unmask_float(cat['DESI_VRAD_ERR'])
cat['p_stream']      = _unmask_float(cat['p_stream'])
cat['p_cocoon']      = _unmask_float(cat['p_cocoon'])

# desi_label: -1 for stars not observed by DESI, 0 background, 1 stream.
desi_label_arr = np.where(
    np.ma.getmaskarray(cat['desi_label']),
    -1,
    np.ma.getdata(cat['desi_label']).astype(int))
cat['desi_label'] = desi_label_arr

n_matched = np.isfinite(np.array(cat['DESI_VRAD'])).sum()
print(f'{n_matched} / {len(cat)} stars have a DESI VRAD measurement')
print(f'{(desi_label_arr == 1).sum()} stream, '
      f'{(desi_label_arr == 0).sum()} background, '
      f'{(desi_label_arr == -1).sum()} not observed by DESI')

# %% [markdown]
# ## Save

# %%
cat.write(OUTPUT_PATH, overwrite=True)
print(f'Saved crossmatched catalog to {OUTPUT_PATH}')
