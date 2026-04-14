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
DATA_PATH      = '/pscratch/sd/p/pratsosa'
DESI_DATA_PATH = '/pscratch/sd/p/pratsosa/DESI_data'
RESULTS_PATH   = '/pscratch/sd/p/pratsosa/GD1_SCREAM/results'
OUTPUT_PATH    = '/pscratch/sd/p/pratsosa/GD-1_gaia_x_decals_VRAD.fits'

# %% [markdown]
# ## Step 1 — Core catalog
# Load GD-1 Gaia×DECaLS fits and join model predictions on source_id.

# %%
cat = Table.read(f'{DATA_PATH}/GD-1_gaia_x_decals_040726.fits')

preds = pd.read_csv(f'{RESULTS_PATH}/cv_test_cv_predictions.csv')
# CSV source_id columns default to float64; coerce before joining.
preds['source_id'] = preds['source_id'].astype(np.int64)
preds = Table.from_pandas(preds)

cat = table.join(cat, preds, keys='source_id')

# %% [markdown]
# ## Step 2 — DESI VRAD
# Loads mwsall-pix-iron.fits RVTAB + GAIA extensions, deduplicates by
# lowest VRAD_ERR per source, then left-joins onto cat on source_id.

# %%
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
DR1_DESI_cat = Table.from_pandas(desi_df)
print(f'{len(DR1_DESI_cat)} unique Gaia sources with valid DESI VRAD measurements')

cat = table.join(cat, DR1_DESI_cat, keys='source_id', join_type='left')
cat.rename_column('VRAD',     'DESI_VRAD')
cat.rename_column('VRAD_ERR', 'DESI_VRAD_ERR')

# Convert masked columns to plain float arrays (NaN for unmatched rows).
desi_vrad = np.where(
    np.ma.getmaskarray(cat['DESI_VRAD']),
    np.nan,
    np.ma.getdata(cat['DESI_VRAD']).astype(float))
desi_vrad_err = np.where(
    np.ma.getmaskarray(cat['DESI_VRAD_ERR']),
    np.nan,
    np.ma.getdata(cat['DESI_VRAD_ERR']).astype(float))
cat['DESI_VRAD']     = desi_vrad
cat['DESI_VRAD_ERR'] = desi_vrad_err

n_matched = np.isfinite(desi_vrad).sum()
print(f'{n_matched} / {len(cat)} stars have a DESI VRAD measurement')

# %% [markdown]
# ## Save

# %%
cat.write(OUTPUT_PATH, overwrite=True)
print(f'Saved crossmatched catalog to {OUTPUT_PATH}')
