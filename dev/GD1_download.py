import logging
from pathlib import Path
from fetcher_new import download_and_match_stream_parallel

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

OUTPUT_DIR = '/pscratch/sd/p/pratsosa/fetcher_output_new'
STREAM_TRACK = 'GD-1-I21'

if __name__ == '__main__':
    log.info(f"Starting download for stream: {STREAM_TRACK}")
    log.info(f"Output directory: {OUTPUT_DIR}")

    result = download_and_match_stream_parallel(
        stream_track=STREAM_TRACK,
        Nchunks=30,
        extension_deg=3.0,
        max_separation_arcsec=1.0,
        output_dir=OUTPUT_DIR,
        subsample_factor=15,
        max_workers=15,
    )

    if result is not None:
        out_path = Path(OUTPUT_DIR) / STREAM_TRACK / f"{STREAM_TRACK}_matched.hdf5"
        log.info(f"Download complete. {len(result)} rows saved to {out_path}")
    else:
        log.error("Download returned no data. Check DataLab connectivity and stream track name.")
