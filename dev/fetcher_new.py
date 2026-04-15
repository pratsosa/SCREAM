# Updated fetcher script from my collaborator which we will need to 
# adapt again in order to use parallel downloads. 
import os
import numpy as np
import pandas as pd
from pathlib import Path

import astropy.units as u
import astropy.coordinates as coord
from astropy.table import Table, vstack, unique, join
from astropy.coordinates import SkyCoord
from dl import queryClient as qc
import fitsio
import galstreams
from typing import Union
from concurrent.futures import ThreadPoolExecutor, as_completed


def track_length(track):
    """Calculate the spherical length of a track in degrees."""
    return np.sum(track[0:-1].separation(track[1:]).deg)


def extend_track(track, extension_deg):
    """
    Extend a stream track at both ends by extension_deg degrees along the local great-circle tangent.
    """
    n_points = int(np.ceil(extension_deg * 101.0))
    pa_start = track[0].position_angle(track[1])
    pa_end = track[-2].position_angle(track[-1])

    delta_arr_start = np.linspace(extension_deg, 0, n_points, endpoint=False) * u.deg
    ext_start = track[0].directional_offset_by(pa_start + 180 * u.deg, delta_arr_start)

    delta_arr_end = np.linspace(0, extension_deg, n_points, endpoint=False)[1:] * u.deg
    ext_end = track[-1].directional_offset_by(pa_end, delta_arr_end)

    ra_all = np.concatenate([ext_start.ra.deg, track.ra.deg, ext_end.ra.deg])
    dec_all = np.concatenate([ext_start.dec.deg, track.dec.deg, ext_end.dec.deg])

    return coord.SkyCoord(ra=ra_all * u.deg, dec=dec_all * u.deg, frame="icrs")


def get_dl_query_from_polygon(skycoo, base_query):
    """Construct Datalab SQL query using q3c polygon match."""
    sky_point_list = " ".join(skycoo.icrs.to_string()).replace(" ", ", ")
    query = f"""{base_query} q3c_poly_query(ra, dec, ARRAY[{sky_point_list}])
    """
    return query


def crossmatch_gaia_ls(
    gaia_table: Table, ls_table: Table, max_separation_arcsec: float = 1.0
) -> Table:
    """
    Perform a cross-match using astropy.coordinates.
    Matches Gaia stars to the closest LS DR9 object within `max_separation_arcsec`.
    This performs a Left Join (all Gaia stars are kept; missing LS matches have NaN).
    """
    if len(gaia_table) == 0:
        return gaia_table

    # We must rename columns so they don't overwrite Gaia's 'ra', 'dec', etc.
    ls_renamed = ls_table.copy()
    for col in ls_renamed.colnames:
        if not col.startswith("ls_"):
            ls_renamed.rename_column(col, f"ls_{col}")

    if len(ls_renamed) == 0:
        # If no LS objects, just append the columns with null values
        for col in ls_renamed.colnames:
            gaia_table[col] = np.nan
        return gaia_table

    c_gaia = SkyCoord(
        ra=gaia_table["ra"] * u.deg, dec=gaia_table["dec"] * u.deg, frame="icrs"
    )
    c_ls = SkyCoord(
        ra=ls_renamed["ls_ra"] * u.deg, dec=ls_renamed["ls_dec"] * u.deg, frame="icrs"
    )

    # Match gaia catalog to ls catalog
    idx, d2d, d3d = c_gaia.match_to_catalog_sky(c_ls)

    # Boolean array indicating matches within max distance
    valid_matches = d2d < max_separation_arcsec * u.arcsec
    match_indices_ls = idx[valid_matches]

    # Initialize all matched columns as NaNs in gaia table
    for col in ls_renamed.colnames:
        dtype = ls_renamed[col].dtype

        # Initialize a masked or NaN-filled array matching gaia length
        if dtype.kind in "if":
            new_col = np.full(len(gaia_table), np.nan, dtype=np.float64)
            new_col[valid_matches] = ls_renamed[col][match_indices_ls]
        else:
            new_col = np.full(len(gaia_table), None, dtype=object)
            new_col[valid_matches] = ls_renamed[col][match_indices_ls]

        gaia_table[col] = new_col

    # Also add match separation to final catalog for transparency
    gaia_table["match_dist_arcsec"] = np.full(len(gaia_table), np.nan, dtype=np.float64)
    gaia_table["match_dist_arcsec"][valid_matches] = (
        d2d[valid_matches].to(u.arcsec).value
    )

    return gaia_table


BASE_GAIA_QUERY = """
SELECT source_id,ra,dec, ra_error, dec_error, parallax,parallax_error,pmra,pmra_error,pmdec,pmdec_error,
phot_g_mean_flux,phot_g_mean_flux_error,phot_g_mean_mag,
phot_bp_mean_flux,phot_bp_mean_flux_error,phot_bp_mean_mag,
phot_rp_mean_flux,phot_rp_mean_flux_error,phot_rp_mean_mag,
bp_rp,radial_velocity,radial_velocity_error,l,b
FROM gaia_dr3.gaia_source
WHERE 
"""
BASE_LS_QUERY = """
SELECT release, brickid, objid, ra, dec, ebv, flux_g, flux_r, flux_z,
flux_ivar_g, flux_ivar_r, flux_ivar_z, 
mw_transmission_g, mw_transmission_r, mw_transmission_z
FROM ls_dr9.tractor 
WHERE type = 'PSF' AND brick_primary = 1 AND
"""


def download_and_match_stream(
    stream_track: str,
    extension_deg: float = 5.0,
    Nchunks: int = 1,
    subsample_factor: int = 1,
    max_separation_arcsec: float = 1.0,
    output_dir: Union[str, Path] = "./data",
    base_gaia_query: str = BASE_GAIA_QUERY,
    base_ls_query: str = BASE_LS_QUERY,
):
    """
    Given a galstreams track, downloads Gaia data, matches with Legacy Survey DR9 via Datalab,
    and saves output as a cross-matched HDF5 file.
    """
    mws = galstreams.MWStreams(verbose=False, implement_Off=True)

    try:
        track = mws[stream_track].track
    except KeyError:
        raise ValueError(f"Stream track {stream_track} not found in MWStreams library.")
    track_extended = extend_track(track, extension_deg=extension_deg)

    print(f"Total points in extended track: {len(track_extended)}")

    # Calculate chunk steps
    chunk_size = int(np.ceil(len(track_extended) * 2 / (Nchunks + 1)))
    step_size = chunk_size // 2

    data_path = Path(output_dir) / stream_track
    data_path.mkdir(parents=True, exist_ok=True)

    out_file = data_path / f"{stream_track}_matched.hdf5"
    if out_file.exists():
        print(f"Found existing dataset: {out_file}. Loading instead of downloading.")
        return Table.read(str(out_file))

    all_matched_data = []

    for i in range(Nchunks):
        start = i * step_size
        stop = start + chunk_size
        if i == Nchunks - 1:
            stop = len(track_extended)

        stop = min(stop, len(track_extended))

        if start >= stop:
            continue

        track_chunk = track_extended[start:stop]
        track_chunk = track_chunk[::subsample_factor]

        if len(track_chunk) < 2:
            print(f"Skipping chunk {i+1} because it has fewer than 2 points.")
            continue
        if 2 * len(track_chunk) > 100:
            raise ValueError(
                f"Polygon has {2*len(track_chunk)+1} vertices. More than 100 vertices are not supported"
            )

        chunk_gaia_list = []
        chunk_ls_list = []

        # We loop over slightly shifted polygons (-1, 0, 1 offsets)
        for j in [-1, 0, 1]:
            poly = galstreams.create_sky_polygon_footprint_from_track(
                track_chunk,
                mws[stream_track].stream_frame,
                width=extension_deg * u.deg,
                phi2_offset=j * extension_deg * u.deg,
            )

            chunk_name = f"chunk_{i+1}_{j}"
            print(f"Processing {chunk_name}")

            # 1. Fetch Gaia Data via NOIRLab DataLab
            gaia_dl_query = get_dl_query_from_polygon(poly, base_query=base_gaia_query)
            try:
                gaia_res = qc.query(sql=gaia_dl_query, fmt="table")
                print(f"  Got {len(gaia_res)} rows from Gaia")
                if len(gaia_res) > 0:
                    chunk_gaia_list.append(gaia_res)
            except Exception as e:
                print(f"  Querying Gaia via DataLab failed: {e}")

            # 2. Fetch LS DR9 Data via NOIRLab DataLab
            ls_dl_query = get_dl_query_from_polygon(poly, base_query=base_ls_query)
            try:
                ls_res = qc.query(sql=ls_dl_query, fmt="table")
                print(f"  Got {len(ls_res)} rows from LS DR9")
                if len(ls_res) > 0:
                    chunk_ls_list.append(ls_res)
            except Exception as e:
                print(f"  Querying LS DataLab failed: {e}")

        # Aggregate Gaia data across offsets
        if chunk_gaia_list:
            merged_gaia = vstack(chunk_gaia_list)
            if "source_id" in merged_gaia.colnames:
                merged_gaia = unique(merged_gaia, keys=["source_id"], keep="first")
        else:
            merged_gaia = Table()

        # Aggregate LS DR9 data across offsets
        if chunk_ls_list:
            merged_ls = vstack(chunk_ls_list)
            merged_ls = unique(
                merged_ls, keys=["release", "brickid", "objid"], keep="first"
            )
        else:
            merged_ls = Table()

        # 3. Crossmatch
        matched_chunk = crossmatch_gaia_ls(
            merged_gaia, merged_ls, max_separation_arcsec
        )
        if len(matched_chunk) > 0:
            all_matched_data.append(matched_chunk)

    if len(all_matched_data) == 0:
        print("No data extracted for any chunk.")
        return None

    final_table = vstack(all_matched_data)
    final_table = unique(final_table, keys=["source_id"], keep="first")

    final_table.write(str(out_file), path="data", overwrite=True, serialize_meta=True)
    print(f"Saved completed dataset: {out_file} ({len(final_table)} rows)")

    return final_table


def _process_chunk(i, track_extended, step_size, chunk_size, Nchunks, subsample_factor,
                   stream_track, extension_deg, base_gaia_query, base_ls_query,
                   max_separation_arcsec, mws):
    """Process a single chunk: query Gaia + LS for 3 polygon offsets, crossmatch, return table."""
    start = i * step_size
    stop = start + chunk_size
    if i == Nchunks - 1:
        stop = len(track_extended)
    stop = min(stop, len(track_extended))

    if start >= stop:
        return None

    track_chunk = track_extended[start:stop]
    track_chunk = track_chunk[::subsample_factor]

    if len(track_chunk) < 2:
        print(f"Skipping chunk {i+1}: fewer than 2 points.")
        return None

    chunk_gaia_list = []
    chunk_ls_list = []

    for j in [-1, 0, 1]:
        poly = galstreams.create_sky_polygon_footprint_from_track(
            track_chunk, mws[stream_track].stream_frame,
            width=extension_deg * u.deg,
            phi2_offset=j * extension_deg * u.deg,
        )
        chunk_name = f"chunk_{i+1}_{j}"
        print(f"Processing {chunk_name}")

        gaia_dl_query = get_dl_query_from_polygon(poly, base_query=base_gaia_query)
        try:
            gaia_res = qc.query(sql=gaia_dl_query, fmt="table")
            print(f"  [{chunk_name}] Got {len(gaia_res)} Gaia rows")
            if len(gaia_res) > 0:
                chunk_gaia_list.append(gaia_res)
        except Exception as e:
            print(f"  [{chunk_name}] Gaia query failed: {e}")

        ls_dl_query = get_dl_query_from_polygon(poly, base_query=base_ls_query)
        try:
            ls_res = qc.query(sql=ls_dl_query, fmt="table")
            print(f"  [{chunk_name}] Got {len(ls_res)} LS rows")
            if len(ls_res) > 0:
                chunk_ls_list.append(ls_res)
        except Exception as e:
            print(f"  [{chunk_name}] LS query failed: {e}")

    if chunk_gaia_list:
        merged_gaia = vstack(chunk_gaia_list)
        if "source_id" in merged_gaia.colnames:
            merged_gaia = unique(merged_gaia, keys=["source_id"], keep="first")
    else:
        merged_gaia = Table()

    if chunk_ls_list:
        merged_ls = vstack(chunk_ls_list)
        merged_ls = unique(merged_ls, keys=["release", "brickid", "objid"], keep="first")
    else:
        merged_ls = Table()

    matched = crossmatch_gaia_ls(merged_gaia, merged_ls, max_separation_arcsec)
    return matched if len(matched) > 0 else None


def download_and_match_stream_parallel(
    stream_track: str,
    base_gaia_query: str = BASE_GAIA_QUERY,
    base_ls_query: str = BASE_LS_QUERY,
    extension_deg: float = 5.0,
    Nchunks: int = 1,
    subsample_factor: int = 1,
    max_separation_arcsec: float = 1.0,
    output_dir: Union[str, Path] = "./data",
    max_workers: int = 10,
):
    """
    Parallel version of download_and_match_stream.
    Dispatches each chunk to a thread pool; the final dedup/save is sequential.
    max_workers controls concurrency — keep ≤ 15 to avoid DataLab rate limits.
    """
    mws = galstreams.MWStreams(verbose=False, implement_Off=True)

    try:
        track = mws[stream_track].track
    except KeyError:
        raise ValueError(f"Stream track {stream_track} not found in MWStreams library.")

    track_extended = extend_track(track, extension_deg=extension_deg)
    print(f"Total points in extended track: {len(track_extended)}")

    chunk_size = int(np.ceil(len(track_extended) * 2 / (Nchunks + 1)))
    step_size = chunk_size // 2

    data_path = Path(output_dir) / stream_track
    data_path.mkdir(parents=True, exist_ok=True)

    out_file = data_path / f"{stream_track}_matched.hdf5"
    if out_file.exists():
        print(f"Found existing dataset: {out_file}. Loading instead of downloading.")
        return Table.read(str(out_file))

    all_matched_data = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_chunk,
                i, track_extended, step_size, chunk_size, Nchunks, subsample_factor,
                stream_track, extension_deg, base_gaia_query, base_ls_query,
                max_separation_arcsec, mws,
            ): i
            for i in range(Nchunks)
        }
        for future in as_completed(futures):
            i = futures[future]
            try:
                result = future.result()
                if result is not None:
                    all_matched_data.append(result)
                    print(f"Chunk {i+1} complete: {len(result)} matched rows")
            except Exception as e:
                print(f"Chunk {i+1} raised an exception: {e}")

    if not all_matched_data:
        print("No data extracted for any chunk.")
        return None

    print("Stacking and removing duplicates...")
    final_table = vstack(all_matched_data)
    final_table = unique(final_table, keys=["source_id"], keep="first")

    final_table.write(str(out_file), path="data", overwrite=True, serialize_meta=True)
    print(f"Saved completed dataset: {out_file} ({len(final_table)} rows)")

    return final_table
