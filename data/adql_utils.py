import scipy
import pandas as pd
import numpy as np
import astropy
import astropy.coordinates as ac
import astropy.units as u
import pylab as plt
import seaborn as sns
import galstreams
from astroquery.gaia import Gaia
import astropy.coordinates as coord
from pathlib import Path
import fitsio
from astropy.table import Table, vstack, unique
from astropy.io import fits
import h5py

def track_length(track):

    return np.sum(track[0:-1].separation(track[1:]).deg)

def extend_track(track, extension_deg):
    """
    Extend a stream track at both ends by extension_deg degrees along the local great-circle tangent.

    Parameters
    ----------
    track : SkyCoord
        The original track points (must be ordered).
    extension_deg : float
        Number of degrees to extend at each end.
    n_points : int
        Number of points to add at each end.

    Returns
    -------
    SkyCoord
        The extended track.
    """
    n_points = int(np.ceil(extension_deg * 101.0))
    # Calculate position angle (tangent) at each end
    # Start: from first to second point
    pa_start = track[0].position_angle(track[1])
    # End: from last-1 to last point
    pa_end = track[-2].position_angle(track[-1])

    # Generate extension points at start (negative direction)
    # Note: linspace is from extension_deg to a small non-zero value to avoid duplicating the start point
    delta_arr_start = np.linspace(extension_deg, 0, n_points, endpoint=False) * u.deg
    ext_start = track[0].directional_offset_by(pa_start + 180*u.deg, delta_arr_start)
    ext_start = ext_start
    


    # Generate extension points at end (positive direction)
    # Note: linspace is from a small non-zero value to extension_deg
    delta_arr_end = np.linspace(0, extension_deg, n_points, endpoint=False)[1:] * u.deg
    ext_end = track[-1].directional_offset_by(pa_end  , delta_arr_end)
    


    # Concatenate all points
    ra_all = np.concatenate([ext_start.ra.deg, track.ra.deg, ext_end.ra.deg])
    dec_all = np.concatenate([ext_start.dec.deg, track.dec.deg, ext_end.dec.deg])

    return coord.SkyCoord(ra=ra_all*u.deg, dec=dec_all*u.deg, frame="icrs")

def max_polygon_edge_length(poly):
    # poly: SkyCoord of polygon vertices
    seps = poly[:-1].separation(poly[1:]).deg
    # Also check the closing edge
    seps = np.append(seps, poly[-1].separation(poly[0]).deg)
    return np.max(seps)

def ra_dec_to_phi1_phi2(frame, ra, dec):
    '''
    Given a frame, convert ra and dec to phi1 and phi2
    
    Input:
        frame: astropy.coordinates frame
        ra: right ascension in degrees
        dec: declination in degrees
        
    Output:
        phi1: stream phi1 coordinates in degrees
        phi2: stream phi2 coordinates in degrees
    '''
    skycoord_data = coord.SkyCoord(ra=ra, dec=dec, frame='icrs')
    transformed_skycoord = skycoord_data.transform_to(frame)
    phi1, phi2 = transformed_skycoord.phi1.deg, transformed_skycoord.phi2.deg
    return phi1, phi2   
def phi1_phi2_to_ra_dec(frame, phi1, phi2):
    '''
    Given a frame, convert phi1 and phi2 to ra and dec
    
    Input:
        frame: astropy.coordinates frame
        phi1: stream phi1 coordinates in degrees
        phi2: stream phi2 coordinates in degrees
        
    Output:
        ra: right ascension in degrees
        dec: declination in degrees
    '''
    skycoord_data = coord.SkyCoord(phi1=phi1*u.deg, phi2=phi2*u.deg, frame=frame)
    transformed_skycoord = skycoord_data.transform_to('icrs')
    ra, dec = transformed_skycoord.ra.deg, transformed_skycoord.dec.deg
    return ra, dec

def pmra_pmdec_to_pmphi12(frame, ra, dec, pmra, pmdec):
    '''
    Given a frame, convert ra, dec, pmra, and pmdec to pmphi1 and pmphi2
    
    Input:
        frame: astropy.coordinates frame
        ra: right ascension in degrees
        dec: declination in degrees
        pmra: proper motion in right ascension in mas/yr
        pmdec: proper motion in declination in mas/yr
        
    Output:
        pmphi1: stream pmphi1 in mas/yr
        pmphi2: stream pmphi2 in mas/yr
    '''
    skycoord_data = coord.SkyCoord(ra=ra, dec=dec, pm_ra_cosdec=pmra, pm_dec=pmdec, frame='icrs')
    transformed_skycoord = skycoord_data.transform_to(frame)
    pmphi1, pmphi2 = transformed_skycoord.pm_phi1_cosphi2.to(u.mas/u.yr).value, transformed_skycoord.pm_phi2.to(u.mas/u.yr).value
    return pmphi1, pmphi2