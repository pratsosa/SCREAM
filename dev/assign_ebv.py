"""
assign_ebv.py — Phase 2.6 post-processing: fill in EBV for generated stars.

After train_flow.py produces a CSV with ebv=-999 for generated stars, this script:
  1. Reads the CSV.
  2. Converts phi1, phi2 → ra, dec using the GD-1 stream frame.
  3. Queries the SFD dust map for all rows where ebv == -999 (batch query).
  4. Writes the updated CSV back in-place.

Usage:
    python dev/assign_ebv.py <path/to/generated.csv>

The script is idempotent: rows that already have a real ebv value are untouched.
"""

import sys
import logging
import numpy as np
import pandas as pd
import astropy.units as u
import galstreams

# Dustmaps data directory must be configured before importing SFDQuery.
from dustmaps.config import config as dustmaps_config
dustmaps_config['data_dir'] = '/pscratch/sd/p/pratsosa/dust_maps'

from astropy.coordinates import SkyCoord
from dustmaps.sfd import SFDQuery

import warnings
from astropy.utils.exceptions import AstropyDeprecationWarning
warnings.filterwarnings('ignore', category=AstropyDeprecationWarning)

from transforms import phi1_phi2_to_ra_dec

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

STREAM_TRACK = 'GD-1-I21'


def main(csv_path: str) -> None:
    log.info(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    log.info(f"Total rows: {len(df)}")

    mask = df['ebv'] == -999.0
    n_missing = mask.sum()
    log.info(f"Rows with ebv=-999 (generated stars): {n_missing}")

    if n_missing == 0:
        log.info("Nothing to do — all rows already have EBV assigned.")
        return

    # Build the GD-1 stream frame (same as in GD1_data_prep.py).
    log.info("Loading galstreams to get stream frame ...")
    mws = galstreams.MWStreams(verbose=False, implement_Off=True)
    stream_frame = mws[STREAM_TRACK].stream_frame

    phi1 = df.loc[mask, 'phi1'].to_numpy()
    phi2 = df.loc[mask, 'phi2'].to_numpy()

    log.info("Converting phi1/phi2 → ra/dec ...")
    ra, dec = phi1_phi2_to_ra_dec(stream_frame, phi1, phi2)

    log.info(f"Querying SFD dust map for {n_missing} coordinates ...")
    sfd = SFDQuery()
    coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')
    ebv_vals = sfd(coords)

    df.loc[mask, 'ebv'] = ebv_vals
    log.info(f"EBV stats — min: {ebv_vals.min():.4f}, max: {ebv_vals.max():.4f}, "
             f"mean: {ebv_vals.mean():.4f}")

    df.to_csv(csv_path, index=False)
    log.info(f"Updated CSV written to: {csv_path}")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <path/to/generated.csv>")
        sys.exit(1)
    main(sys.argv[1])
