import math
from dataclasses import dataclass
from typing import List, Sequence
from math import pi

@dataclass
class Material:
    E: float  # Young's modulus (Pa)
    nu: float  # Poisson's ratio
    sigma_y: float  # Yield strength (Pa)

@dataclass
class Geometry:
    R: float  # Hull radius (m)
    t: float  # Shell thickness (m)
    L: float  # Hull length (m)

@dataclass
class Env:
    rho: float  # Water density (kg/m³)
    g: float  # Gravity (m/s²)

@dataclass
class Factors:
    eta_end: float  # End effect multiplier
    kdf_overall: float  # Overall buckling knockdown factor
    kdf_interframe: float  # Interframe buckling knockdown factor
    phi_yield: float  # Yield strength reduction factor
    gamma_global: float  # Global load factor

@dataclass
class Coefficients:
    K_if: float  # Interframe buckling coefficient
    C_overall: float  # Overall buckling coefficient

def von_mises_interframe_buckling(E, n, total_length, num_interframes, De, s, mu):
    """
    Calculate critical buckling pressure for a cylinder with multiple interframes.
    
    Parameters:
    E              : Young's modulus (MPa or consistent units).
    n              : Number of circumferential lobes (integer >= 2).
    total_length   : Total length of the cylinder (mm or consistent units).
    num_interframes: Number of equally spaced interframes (integer >= 0).
    De             : Equivalent diameter (mm).
    s              : Shell thickness (mm).
    mu             : Poisson's ratio.
    
    Returns:
    p_ee           : Critical buckling pressure (same units as E).
    """
    # Validate inputs
    if n < 2:
        raise ValueError("Number of lobes (n) must be >= 2.")
    if num_interframes < 0:
        raise ValueError("Number of interframes must be >= 0.")
    if total_length <= 0 or De <= 0 or s <= 0:
        raise ValueError("Dimensions must be positive.")
    if mu <= -1 or mu >= 0.5:
        raise ValueError("Poisson's ratio must be between -1 and 0.5.")

    # Calculate frame spacing (L)
    num_spacings = num_interframes + 1
    L = total_length / num_spacings  # Spacing between frames

    # Precompute recurring terms
    ratio = (2 * n * L) / (math.pi * De)
    ratio_sq = ratio ** 2
    s_De_ratio = s / De

    # Term 1: Membrane stress component
    denom1 = (n**2 - 1) * (1 + ratio_sq) ** 2
    term1 = (2 * E / denom1) * s_De_ratio

    # Term 2: Bending stress component
    denom2 = 3 * (1 - mu**2)
    fraction = (2 * n**2 - 1 - mu) / (1 - ratio_sq)
    bracket_term = n**2 - 1 - fraction * (s_De_ratio ** 3)
    term2 = (2 * E / denom2) * bracket_term

    return term1 + term2

def interframe_allowable_pressure(material: Material, geometry: Geometry, bay_length: float, coeffs: Coefficients, factors: Factors) -> float:
    """Calculate allowable pressure for interframe (local) buckling"""
    n = 4  # Typical number of circumferential waves
    De = 2 * geometry.R  # Effective diameter
    p_crit = von_mises_interframe_buckling(
        material.E, n, bay_length, 0, De, geometry.t, material.nu
    )
    return p_crit * coeffs.K_if * factors.kdf_interframe / factors.gamma_global

def overall_allowable_pressure(material: Material, geometry: Geometry, coeffs: Coefficients, factors: Factors) -> float:
    """Calculate allowable pressure for overall (global) buckling"""
    p_crit = coeffs.C_overall * material.E / math.sqrt(3 * (1 - material.nu**2)) * (geometry.t / geometry.R)**3
    return p_crit * factors.eta_end * factors.kdf_overall / factors.gamma_global

def yielding_allowable_pressure(material: Material, geometry: Geometry, factors: Factors) -> float:
    """Calculate allowable pressure based on yielding"""
    return material.sigma_y * factors.phi_yield * geometry.t / (geometry.R * factors.gamma_global)

def pressure_to_depth(p: float, env: Env) -> float:
    """Convert pressure to depth"""
    return p / (env.rho * env.g)

def _make_disk_positions_from_params(L: float, n_internal: int) -> List[float]:
    """Generate disk positions from number of internal disks"""
    if n_internal < 0:
        raise ValueError("Number of internal disks must be non-negative")
    if n_internal == 0:
        return [0, L]  # Just end disks
    step = L / (n_internal + 1)
    return [0] + [step * (i + 1) for i in range(n_internal)] + [L]

def find_optimal_n(E, total_length, num_interframes, De, s, mu, n_range=range(2, 21)):
    """Find the number of waves that gives the lowest critical pressure"""
    min_pressure = float('inf')
    optimal_n = None
    debug_info = []
    for n in n_range:
        try:
            # Calculate pressure for this n
            p = von_mises_interframe_buckling(E, n, total_length, num_interframes, De, s, mu)
            ratio = (2 * n * total_length) / (math.pi * De)
            ratio_sq = ratio ** 2
            debug_info.append(f"n={n}: ratio²={ratio_sq:.3f}, p={p:.2e} Pa")
            if p < min_pressure:
                min_pressure = p
                optimal_n = n
        except ValueError as e:
            debug_info.append(f"n={n}: {str(e)}")
            continue
    if optimal_n is None:
        debug_msg = "\n".join(debug_info)
        raise ValueError(f"No valid n value found in the given range. Debug info:\n{debug_msg}")
    return optimal_n, min_pressure

def evaluate_explicit(material: Material, geometry: Geometry, disk_positions: Sequence[float], env: Env, factors: Factors, coeffs: Coefficients, n_waves: int = None) -> dict:
    """
    Evaluate safe depths with explicit disk positions.
    
    Parameters:
        n_waves: Optional fixed number of circumferential waves. If None, finds optimal n.
    
    Returns:
    dict with:
        disk_positions: List[float] - Actual disk positions used (including ends)
        bay_lengths: List[float] - Length of each bay
        h_safe_interframe: float - Safe depth based on interframe buckling
        h_safe_overall: float - Safe depth based on overall buckling
        h_safe_yielding: float - Safe depth based on yielding
        h_safe_min: float - Minimum of all safe depths
        mode_controlling: str - Which mode controls the minimum depth
        optimal_n: int - Optimal number of waves (if n_waves was None)
        n_used: int - Number of waves actually used
    """
    # Validate and process disk positions
    disk_pos = sorted(set([0, geometry.L] + list(disk_positions)))
    bay_lengths = [disk_pos[i+1] - disk_pos[i] for i in range(len(disk_pos)-1)]
    
    # Calculate safe depths for each failure mode
    De = 2 * geometry.R  # Effective diameter
    
    # For interframe buckling, find optimal n if not specified
    if n_waves is None:
        # Find optimal n for each bay and use the most critical
        min_depth = float('inf')
        optimal_n = None
        for bay_length in bay_lengths:
            try:
                n, p_crit = find_optimal_n(
                    material.E, bay_length, 0, De, geometry.t, material.nu
                )
                depth = pressure_to_depth(
                    p_crit * coeffs.K_if * factors.kdf_interframe / factors.gamma_global,
                    env
                )
                if depth < min_depth:
                    min_depth = depth
                    optimal_n = n
            except ValueError:
                continue
        if optimal_n is None:
            raise ValueError("Could not find valid n for any bay")
        n_used = optimal_n
        h_interframe = min_depth
    else:
        n_used = n_waves
        h_interframe = min(
            pressure_to_depth(
                interframe_allowable_pressure(material, geometry, bay_length, coeffs, factors),
                env
            )
            for bay_length in bay_lengths
        )
    
    h_overall = pressure_to_depth(
        overall_allowable_pressure(material, geometry, coeffs, factors),
        env
    )
    
    h_yielding = pressure_to_depth(
        yielding_allowable_pressure(material, geometry, factors),
        env
    )
    
    # Find controlling mode
    h_min = min(h_interframe, h_overall, h_yielding)
    if h_min == h_interframe:
        mode = "Interframe Buckling"
    elif h_min == h_overall:
        mode = "Overall Buckling"
    else:
        mode = "Yielding"
    
    return {
        "disk_positions": disk_pos,
        "bay_lengths": bay_lengths,
        "h_safe_interframe": h_interframe,
        "h_safe_overall": h_overall,
        "h_safe_yielding": h_yielding,
        "h_safe_min": h_min,
        "mode_controlling": mode,
        "n_used": n_used,
        "optimal_n": n_used if n_waves is None else None
    }

def evaluate_from_params(material: Material, geometry: Geometry, n_internal: int, env: Env, factors: Factors, coeffs: Coefficients) -> dict:
    """Evaluate safe depths by generating disk positions from number of internal disks"""
    disk_positions = _make_disk_positions_from_params(geometry.L, n_internal)
    return evaluate_explicit(material, geometry, disk_positions, env, factors, coeffs)