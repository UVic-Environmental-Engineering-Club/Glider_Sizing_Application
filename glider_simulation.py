import numpy as np
from scipy.integrate import solve_ivp
from glider_physics import UnderwaterGlider

import os
import json
import csv
from scipy.interpolate import interp1d

def _normalize_cfd_rows(rows):
    """Normalize rows to ndarray with columns [AoA_deg, Cd_x, Cd_y, Cd_z, CL, CM] and sort by AoA."""
    arr = np.asarray(rows, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 6:
        raise ValueError(f"CFD table must be Nx6, got shape {arr.shape}")
    # sort by AoA column 0
    idx = np.argsort(arr[:, 0])
    return arr[idx]

def load_cfd_table_from_file(path: str) -> np.ndarray:
    """
    Load a CFD table from CSV or JSON.
    Expected columns/order: [AoA_deg, Cd_x, Cd_y, Cd_z, CL, CM]
    JSON can be:
      - list of lists [[AoA, Cd_x, ...], ...]
      - list of dicts [{"AoA_deg":..., "Cd_x":..., ...}, ...]
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in [".csv", ".txt"]:
        rows = []
        with open(path, newline="", encoding="utf-8") as f:
            sniffer = csv.Sniffer()
            sample = f.read(1024)
            f.seek(0)
            has_header = sniffer.has_header(sample)
            reader = csv.reader(f)
            header = None
            if has_header:
                header = next(reader)
                header_lower = [h.strip().lower() for h in header]
                try:
                    col_map = {
                        'aoa_deg': header_lower.index('aoa_deg'),
                        'cd_x': header_lower.index('cd_x'),
                        'cd_y': header_lower.index('cd_y'),
                        'cd_z': header_lower.index('cd_z'),
                        'cl': header_lower.index('cl'),
                        'cm': header_lower.index('cm'),
                    }
                    for row in reader:
                        if not row or all(c.strip()=='' for c in row):
                            continue
                        rows.append([
                            row[col_map['aoa_deg']], row[col_map['cd_x']], row[col_map['cd_y']],
                            row[col_map['cd_z']], row[col_map['cl']], row[col_map['cm']]
                        ])
                except ValueError:
                    # Fallback: treat as no header
                    rows = []
                    f.seek(0)
                    reader = csv.reader(f)
                    for row in reader:
                        if not row or all(c.strip()=='' for c in row):
                            continue
                        rows.append(row[:6])
            else:
                for row in reader:
                    if not row or all(c.strip()=='' for c in row):
                        continue
                    rows.append(row[:6])
        return _normalize_cfd_rows(rows)
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], dict):
                rows = []
                for d in data:
                    rows.append([
                        d.get('AoA_deg') if 'AoA_deg' in d else d.get('aoa_deg'),
                        d.get('Cd_x') if 'Cd_x' in d else d.get('cd_x'),
                        d.get('Cd_y') if 'Cd_y' in d else d.get('cd_y'),
                        d.get('Cd_z') if 'Cd_z' in d else d.get('cd_z'),
                        d.get('CL') if 'CL' in d else d.get('cl'),
                        d.get('CM') if 'CM' in d else d.get('cm'),
                    ])
                return _normalize_cfd_rows(rows)
            else:
                return _normalize_cfd_rows(data)
        raise ValueError("Unsupported JSON structure for CFD table")
    else:
        raise ValueError(f"Unsupported CFD table file extension: {ext}")

def apply_cfd_table_to_glider(glider: UnderwaterGlider, cfd_table: np.ndarray) -> None:
    """
    Apply a CFD table to the glider instance by setting the table and rebuilding interpolants.
    Mirrors the logic in the physics class without modifying it.
    """
    table = _normalize_cfd_rows(cfd_table)
    glider.cfd_table = table
    aoa_deg = table[:, 0]
    glider._interp_cd_x = interp1d(aoa_deg, table[:, 1], kind='linear', bounds_error=False, fill_value=(table[0, 1], table[-1, 1]))
    glider._interp_cd_y = interp1d(aoa_deg, table[:, 2], kind='linear', bounds_error=False, fill_value=(table[0, 2], table[-1, 2]))
    glider._interp_cd_z = interp1d(aoa_deg, table[:, 3], kind='linear', bounds_error=False, fill_value=(table[0, 3], table[-1, 3]))
    glider._interp_cl   = interp1d(aoa_deg, table[:, 4], kind='linear', bounds_error=False, fill_value=(table[0, 4], table[-1, 4]))
    glider._interp_cm   = interp1d(aoa_deg, table[:, 5], kind='linear', bounds_error=False, fill_value=(table[0, 5], table[-1, 5]))


def run_simulation(params, control_func=None, t_end=1, dt=1, 
                  init_depth=0, init_pitch=0, solver="Radau", progress_callback=None):
    """
    Run glider simulation with specified parameters and control
    
    Args:
        params: Dictionary of glider parameters
        control_func: Control function (t, state) -> (dm_dt, dx_dt)
        t_end: Simulation duration (seconds)
        dt: Time step (seconds)
        init_depth: Initial depth (meters, positive down)
        init_pitch: Initial pitch angle (degrees)
        solver: ODE solver method
        progress_callback: Optional callback function for progress updates
    
    Returns:
        solution: ODE solution object
    """
    # Initialize glider
    glider = UnderwaterGlider(params)

    # Optional: load CFD table from file or in-memory data without modifying physics file
    try:
        cfd_table_path = params.get('cfd_table_path') if isinstance(params, dict) else None
        if cfd_table_path:
            table = load_cfd_table_from_file(cfd_table_path)
            apply_cfd_table_to_glider(glider, table)
        elif 'cfd_table' in params:
            table = params['cfd_table']
            apply_cfd_table_to_glider(glider, np.asarray(table, dtype=float))
    except Exception as e:
        print(f"Warning: Failed to apply external CFD table: {e}")
    
    # Get initial state vector and create state0 for simulation
    state0 = glider.get_state_vector()
    
    # Ensure the glider has valid mass properties before simulation
    # This prevents NaN/inf values in inertia calculations
    if glider.mass is None or glider.mass <= 0:
        print("Warning: Invalid mass detected, recalculating...")
        glider._calculate_mass_properties()
        glider._calculate_inertia()
        state0 = glider.get_state_vector()
    
    # Verify state vector is valid
    if np.any(np.isnan(state0)) or np.any(np.isinf(state0)):
        print("Warning: Invalid state detected, resetting to defaults...")
        # Reset to safe defaults
        state0 = np.zeros(17)
        state0[3] = 1.0  # Identity quaternion
        glider.set_state_vector(state0)
        state0 = glider.get_state_vector()
    
    # Set initial depth (z position, index 2)
    state0[2] = init_depth
    
    # Set initial pitch (quaternion, indices 3:7)
    # Start with identity quaternion, then rotate about y by pitch (degrees)
    from scipy.spatial.transform import Rotation as R
    quat = R.from_euler('y', init_pitch, degrees=True).as_quat()  # [x, y, z, w]
    state0[3:7] = quat
    
    # Time points
    t_eval = np.arange(0, t_end, dt)
    
    # Progress tracking
    if progress_callback:
        progress_callback(10)  # Initialization complete
    
    # Custom progress tracking wrapper with control integration
    def dynamics_with_progress(t, y):
        # Apply control inputs if control function is provided
        if control_func is not None:
            try:
                # Call control function to get control inputs
                dm_dt, dx_dt = control_func(t, y)
                
                # Apply control inputs by updating the glider state
                # For ballast control, set the pump state (don't step the ballast here)
                if abs(dm_dt) > 1e-6:  # Only update if there's a significant rate
                    # Set pump state based on desired mass rate
                    glider.set_pump_from_mass_rate(dm_dt)
                    
                    # Update the glider's MVM offset to keep it in sync
                    glider.mvm_offset[0] = y[14]
                    glider.mvm_offset[1] = y[15] 
                    glider.mvm_offset[2] = y[16]
                
                # For MVM control, calculate the rate but don't update position here
                # The position will be updated by the ODE solver using the derivatives
                if abs(dx_dt) > 1e-6:  # Only update if there's a significant rate
                    # Store the desired rate in the glider for the derivative calculation
                    glider._desired_mvm_rate = dx_dt
                else:
                    glider._desired_mvm_rate = 0.0
                
                # # Debug output (uncomment to see control values)
                # if t % 1.0 < 0.1:  # Print every ~1 second
                #     current_depth = y[2]  # Depth is at index 2 in state vector
                    
                #     # Get current forces and mass for debug
                #     glider.compute_forces_and_moments()
                    
                #     # Call the detailed force debug method
                #     glider.debug_forces()
                    
                #     print(f"t={t:.1f}: Control applied - dm_dt={dm_dt:.4f}, dx_dt={dx_dt:.4f}")
                #     print(f"  Depth: {current_depth:.2f}m | Pump: {'ON' if glider.pump_on else 'OFF'}, Direction: {glider.pump_direction}")
                #     print(f"  Ballast fill: {glider.fill_fraction:.3f}, MVM x: {glider.mvm_offset[0]:.3f}")
                
            except Exception as e:
                print(f"Control function error at t={t}: {e}")
                # Fallback to no control - do nothing
        
        # Calculate progress based on current time
        if progress_callback:
            progress = int(10 + (t / t_end) * 80)  # 10-90% during simulation
            progress_callback(progress)
        
        # Safety check: ensure state vector is valid
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            print(f"Warning: Invalid state detected at t={t}, resetting...")
            # Return zero derivatives for invalid state
            return np.zeros_like(y)
        
        # Safety check: ensure ballast fill stays within 0-100%
        if len(y) > 13:
            y[13] = np.clip(y[13], 0.0, 1.0)
        
        # Safety check: ensure MVM offset stays within travel limits
        if len(y) > 14 and hasattr(glider, 'MVM_length'):
            max_offset = glider.MVM_length / 2
            y[14] = np.clip(y[14], -max_offset, max_offset)
            y[15] = np.clip(y[15], -max_offset, max_offset)
            y[16] = np.clip(y[16], -max_offset, max_offset)
        
        # Use the available compute_state_derivatives method
        try:
            derivatives = glider.compute_state_derivatives(t, y, dt)
            
            # # Debug output every 0.5 seconds to see forces
            # if t % 0.5 < 0.01:  # Print every ~0.5 seconds
            #     glider.debug_forces()
            
            # Safety check: ensure derivatives are valid
            if np.any(np.isnan(derivatives)) or np.any(np.isinf(derivatives)):
                print(f"Warning: Invalid derivatives detected at t={t}, using zeros...")
                return np.zeros_like(y)
            
            return derivatives
            
        except Exception as e:
            print(f"Error in dynamics at t={t}: {e}")
            # Return zero derivatives on error
            return np.zeros_like(y)
    
    # Run simulation
    sol = solve_ivp(
        dynamics_with_progress,
        [0, t_end],
        state0,
        t_eval=t_eval,
        method=solver
    )
    
    if progress_callback:
        progress_callback(100)  # Simulation complete
    
    return sol