import numpy as np
import astropy.units as u
from astropy import coordinates as coord


def ra_dec_to_phi1_phi2(frame, ra, dec):
    """
    Given a frame, convert ra and dec to phi1 and phi2

    Input:
        frame: astropy.coordinates frame
        ra: right ascension in degrees
        dec: declination in degrees

    Output:
        phi1: stream phi1 coordinates in degrees
        phi2: stream phi2 coordinates in degrees
    """
    skycoord_data = coord.SkyCoord(ra=ra, dec=dec, frame="icrs")
    transformed_skycoord = skycoord_data.transform_to(frame)
    phi1, phi2 = transformed_skycoord.phi1.deg, transformed_skycoord.phi2.deg
    return phi1, phi2


def phi1_phi2_to_ra_dec(frame, phi1, phi2):
    """
    Given a frame, convert phi1 and phi2 to ra and dec

    Input:
        frame: astropy.coordinates frame
        phi1: stream phi1 coordinates in degrees
        phi2: stream phi2 coordinates in degrees

    Output:
        ra: right ascension in degrees
        dec: declination in degrees
    """
    skycoord_data = coord.SkyCoord(phi1=phi1 * u.deg, phi2=phi2 * u.deg, frame=frame)
    transformed_skycoord = skycoord_data.transform_to("icrs")
    ra, dec = transformed_skycoord.ra.deg, transformed_skycoord.dec.deg
    return ra, dec


def pmra_pmdec_to_pmphi12(frame, ra, dec, pmra, pmdec):
    """
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
    """
    skycoord_data = coord.SkyCoord(
        ra=ra, dec=dec, pm_ra_cosdec=pmra, pm_dec=pmdec, frame="icrs"
    )
    transformed_skycoord = skycoord_data.transform_to(frame)
    pmphi1, pmphi2 = (
        transformed_skycoord.pm_phi1_cosphi2.to(u.mas / u.yr).value,
        transformed_skycoord.pm_phi2.to(u.mas / u.yr).value,
    )
    return pmphi1, pmphi2


def flux_to_mag_ls(flux):
    return 22.5 - 2.5 * np.log10(flux)


def flux_to_mag_gaia(flux, band):
    zero_points = {"G": 25.6873668671, "BP": 25.3385422158, "RP": 24.7478955012}
    return -2.5 * np.log10(flux) + zero_points[band]


def mw_transmission_ls(cat, band):
    """
    Calculate the Milky Way transmission for a given band of legacy survey
    """
    band_coeffs = {
        "u": 3.995,
        "g": 3.214,
        "r": 2.165,
        "i": 1.592,
        "z": 1.211,
        "Y": 1.064,
        "W1": 0.184,
        "W2": 0.113,
        "W3": 0.0241,
        "W4": 0.00910,
    }
    A = band_coeffs[band] * cat["EBV"]
    return 10 ** (A / -2.5)


def mw_extinction_gaia(G, bp, rp, ebv, maxnit=100):
    """Compute the Gaia extinctions assuming relations from Babusieux
    Arguments: G, bp, rp, E(B-V)
    maxnit -- number of iterations
    Returns extinction in G,bp, rp
    Author: Sergey Koposov skoposov@cmu.edu
    """
    c1, c2, c3, c4, c5, c6, c7 = [
        0.9761,
        -0.1704,
        0.0086,
        0.0011,
        -0.0438,
        0.0013,
        0.0099,
    ]
    d1, d2, d3, d4, d5, d6, d7 = [
        1.1517,
        -0.0871,
        -0.0333,
        0.0173,
        -0.0230,
        0.0006,
        0.0043,
    ]
    e1, e2, e3, e4, e5, e6, e7 = [
        0.6104,
        -0.0170,
        -0.0026,
        -0.0017,
        -0.0078,
        0.00005,
        0.0006,
    ]
    A0 = 3.1 * ebv
    P1 = np.poly1d([c1, c2, c3, c4][::-1])

    def F1(bprp):
        return (
            np.poly1d([c1, c2, c3, c4][::-1])(bprp)
            + c5 * A0
            + c6 * A0**2
            + c7 * bprp * A0
        )

    def F2(bprp):
        return (
            np.poly1d([d1, d2, d3, d4][::-1])(bprp)
            + d5 * A0
            + d6 * A0**2
            + d7 * bprp * A0
        )

    def F3(bprp):
        return (
            np.poly1d([e1, e2, e3, e4][::-1])(bprp)
            + e5 * A0
            + e6 * A0**2
            + e7 * bprp * A0
        )

    xind = np.isfinite(bp + rp + G)
    curbp = bp - rp
    for i in range(maxnit):
        AG = F1(curbp) * A0
        Abp = F2(curbp) * A0
        Arp = F3(curbp) * A0
        curbp1 = bp - rp - Abp + Arp

        delta = np.abs(curbp1 - curbp)[xind]
        curbp = curbp1
    AG = F1(curbp) * A0
    Abp = F2(curbp) * A0
    Arp = F3(curbp) * A0
    return AG, Abp, Arp
