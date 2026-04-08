import torch
from torch import Tensor

# --- Gaia zero points (from dev/transforms.py) ---
ZP_G  = 25.6873668671
ZP_BP = 25.3385422158
ZP_RP = 24.7478955012

# --- Flux <-> magnitude conversions ---

def flux_to_mag_gaia(flux: Tensor, zp: float) -> Tensor:
    """-2.5 * log10(flux) + zp  (matches dev/transforms.py line 75)"""
    return -2.5 * torch.log10(flux.clamp(min=1e-10)) + zp


def mag_to_flux_gaia(mag: Tensor, zp: float) -> Tensor:
    """Inverse of flux_to_mag_gaia."""
    return 10.0 ** ((zp - mag) / 2.5)


def flux_to_mag_ls(flux: Tensor) -> Tensor:
    """22.5 - 2.5 * log10(flux)  (matches dev/transforms.py line 71)"""
    return 22.5 - 2.5 * torch.log10(flux.clamp(min=1e-10))


def mag_to_flux_ls(mag: Tensor) -> Tensor:
    """Inverse of flux_to_mag_ls."""
    return 10.0 ** ((22.5 - mag) / 2.5)


# --- Extinction corrections ---

def extinction_ls(ebv: Tensor):
    """
    Return (A_g, A_r, A_z) in magnitudes.
    Coefficients from dev/transforms.py mw_transmission_ls (g=3.214, r=2.165, z=1.211).
    Apply as: mag0 = mag - A
    """
    return 3.214 * ebv, 2.165 * ebv, 1.211 * ebv


def extinction_gaia(G: Tensor, Bp: Tensor, Rp: Tensor, ebv: Tensor,
                    n_iter: int = 10):
    """
    Iterative Gaia extinction using Babusiaux polynomial relations.
    Exact PyTorch port of mw_extinction_gaia in dev/transforms.py.
    Returns (A_G, A_Bp, A_Rp) in magnitudes.
    Apply as: G0 = G - A_G, etc.

    Coefficients (c for G, d for Bp, e for Rp):
    c: [0.9761, -0.1704, 0.0086, 0.0011, -0.0438, 0.0013, 0.0099]
    d: [1.1517, -0.0871, -0.0333, 0.0173, -0.0230, 0.0006, 0.0043]
    e: [0.6104, -0.0170, -0.0026, -0.0017, -0.0078, 0.00005, 0.0006]
    Polynomial: p(x) = c1 + c2*x + c3*x^2 + c4*x^3  (ascending powers)
    F(bprp) = poly(bprp) + c5*A0 + c6*A0^2 + c7*bprp*A0
    A_band = F(bprp) * A0,   A0 = 3.1 * ebv
    """
    A0 = 3.1 * ebv
    curbp = (Bp - Rp).clamp(min=-2.0, max=5.0)
    for _ in range(n_iter):
        AG  = (0.9761 + (-0.1704)*curbp + 0.0086*curbp**2 + 0.0011*curbp**3
               + (-0.0438)*A0 + 0.0013*A0**2 + 0.0099*curbp*A0) * A0
        ABp = (1.1517 + (-0.0871)*curbp + (-0.0333)*curbp**2 + 0.0173*curbp**3
               + (-0.0230)*A0 + 0.0006*A0**2 + 0.0043*curbp*A0) * A0
        ARp = (0.6104 + (-0.0170)*curbp + (-0.0026)*curbp**2 + (-0.0017)*curbp**3
               + (-0.0078)*A0 + 0.00005*A0**2 + 0.0006*curbp*A0) * A0
        curbp = ((Bp - Rp) - ABp + ARp).clamp(min=-2.0, max=5.0)
    return AG, ABp, ARp
