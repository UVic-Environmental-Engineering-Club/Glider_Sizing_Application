import numpy as np
from scipy.spatial.transform import Rotation as R

def depth_pitch_control(t, state, set_depth=20, set_pitch=0, glider_params=None):
    """
    Control strategy to maintain desired depth and pitch
    set_depth: Target depth in meters (positive down)
    set_pitch: Target pitch angle in degrees
    glider_params: Optional glider parameters for limit checking
    """
    depth = state[2]
    quat = state[3:7]
    
    # Convert quaternion to euler angles using xyz order (roll, pitch, yaw)
    # For a glider: x=forward, y=right, z=down
    # Pitch is rotation about y-axis (side-to-side)
    try:
        euler = R.from_quat(quat).as_euler('xyz')
        pitch = euler[1]  # Pitch angle in radians (rotation about y-axis)
    except Exception as e:
        print(f"Quaternion conversion error: {e}, quat={quat}")
        pitch = 0.0
    
    # Ballast control with proportional control and physics-based limits
    depth_error = set_depth - depth
    
    # Get current ballast fill level from state
    current_fill = state[13] if len(state) > 13 else 0.5
    
    # Calculate ballast control with limits based on current fill
    if glider_params:
        max_ballast_flow = glider_params.get('max_ballast_flow', 1e-3)  # m³/s
        rho_water = glider_params.get('rho_water', 1025.0)  # kg/m³
        ballast_radius = glider_params.get('ballast_radius', 0.05)  # m
        ballast_length = glider_params.get('ballast_length', 0.2)  # m
        
        # Convert flow rate to mass rate (kg/s)
        max_mass_rate = max_ballast_flow * rho_water
        
        # Scale factor: 0.1 kg/s per meter of depth error, with physics limits
        dm_dt = np.clip(depth_error * 0.1, -max_mass_rate, max_mass_rate)
        
        # Additional limit: prevent over/under-filling
        if current_fill <= 0.01 and dm_dt < 0:  # Nearly empty, can't remove more
            dm_dt = 0.0
        elif current_fill >= 0.99 and dm_dt > 0:  # Nearly full, can't add more
            dm_dt = 0.0
    else:
        # Fallback to original limits if no params provided
        dm_dt = np.clip(depth_error * 0.1, -0.5, 0.5)
    
    # Pitch control with proportional control and physics-based limits
    pitch_error = np.radians(set_pitch) - pitch
    
    # Get current MVM offset from state
    current_mvm_x = state[14] if len(state) > 14 else 0.0
    
    # Calculate MVM control with limits based on current position
    if glider_params:
        mvm_length = glider_params.get('MVM_length', 0.5)  # m
        max_mvm_velocity = 0.02  # m/s (reasonable actuator speed)
        
        # Scale factor: 0.01 m/s per radian of pitch error, with physics limits
        dx_dt = np.clip(pitch_error * 0.01, -max_mvm_velocity, max_mvm_velocity)
        
        # Additional limit: prevent MVM from exceeding travel limits
        max_offset = mvm_length / 2
        if current_mvm_x <= -max_offset + 0.01 and dx_dt < 0:  # At left limit
            dx_dt = 0.0
        elif current_mvm_x >= max_offset - 0.01 and dx_dt > 0:  # At right limit
            dx_dt = 0.0
    else:
        # Fallback to original limits if no params provided
        dx_dt = np.clip(pitch_error * 0.01, -0.02, 0.02)
    
    # Debug output (uncomment for debugging)
    # if t % 1.0 < 0.1:  # Print every ~1 second
    #     print(f"t={t:.1f}: depth={depth:.2f}, target={set_depth}, error={depth_error:.2f}, dm_dt={dm_dt:.3f}")
    #     print(f"  pitch={np.degrees(pitch):.1f}°, target={set_pitch}°, error={np.degrees(pitch_error):.1f}°, dx_dt={dx_dt:.4f}")
    #     print(f"  ballast_fill={current_fill:.3f}, mvm_x={current_mvm_x:.3f}")
    
    return dm_dt, dx_dt

def trajectory_following_control(t, state, waypoints, glider_params=None):
    """
    Advanced control for following a trajectory
    waypoints: List of (x, y, depth) coordinates
    glider_params: Optional glider parameters for limit checking
    """
    # Simplified implementation - would include path following logic
    current_pos = state[0:3]
    
    # Find nearest waypoint (simplified)
    target = waypoints[0]
    if np.linalg.norm(current_pos - target) < 5:  # Within 5m radius
        target = waypoints[1] if len(waypoints) > 1 else waypoints[0]
    
    # Calculate desired pitch based on depth difference
    depth_error = target[2] - current_pos[2]
    desired_pitch = np.clip(depth_error * 5, -30, 30)  # Scale factor
    
    # Use depth_pitch control with calculated pitch
    return depth_pitch_control(t, state, set_depth=target[2], set_pitch=desired_pitch, glider_params=glider_params)

def simple_depth_control(t, state, set_depth=20, glider_params=None):
    """
    Simple proportional depth control only
    """
    depth = state[2]
    depth_error = set_depth - depth
    
    # Get current ballast fill level from state
    current_fill = state[13] if len(state) > 13 else 0.5
    
    # Calculate ballast control with physics-based limits
    if glider_params:
        max_ballast_flow = glider_params.get('max_ballast_flow', 1e-3)  # m³/s
        rho_water = glider_params.get('rho_water', 1025.0)  # kg/m³
        max_mass_rate = max_ballast_flow * rho_water
        
        # Proportional control with physics limits
        dm_dt = np.clip(depth_error * 0.05, -max_mass_rate, max_mass_rate)
        
        # Prevent over/under-filling
        if current_fill <= 0.01 and dm_dt < 0:
            dm_dt = 0.0
        elif current_fill >= 0.99 and dm_dt > 0:
            dm_dt = 0.0
    else:
        # Fallback to original limits
        dm_dt = np.clip(depth_error * 0.05, -0.3, 0.3)
    
    dx_dt = 0.0  # No pitch control
    
    return dm_dt, dx_dt

def neutral_buoyancy_control(t, state, target_depth=20, glider_params=None):
    """
    Control strategy focused on maintaining neutral buoyancy at target depth
    """
    depth = state[2]
    current_fill = state[13] if len(state) > 13 else 0.5
    
    # Calculate depth error
    depth_error = target_depth - depth
    
    # Ballast control: more aggressive for depth control
    if glider_params:
        max_ballast_flow = glider_params.get('max_ballast_flow', 1e-3)
        rho_water = glider_params.get('rho_water', 1025.0)
        max_mass_rate = max_ballast_flow * rho_water
        
        # Proportional control with higher gain for depth
        dm_dt = np.clip(depth_error * 0.2, -max_mass_rate, max_mass_rate)
        
        # Prevent over/under-filling
        if current_fill <= 0.01 and dm_dt < 0:
            dm_dt = 0.0
        elif current_fill >= 0.99 and dm_dt > 0:
            dm_dt = 0.0
    else:
        dm_dt = np.clip(depth_error * 0.2, -0.5, 0.5)
    
    # Minimal pitch control to maintain level flight
    quat = state[3:7]
    try:
        euler = R.from_quat(quat).as_euler('xyz')
        pitch = euler[1]
        pitch_error = 0 - pitch  # Target level flight (0° pitch)
        dx_dt = np.clip(pitch_error * 0.005, -0.01, 0.01)  # Gentle pitch correction
    except:
        dx_dt = 0.0
    
    return dm_dt, dx_dt

def validate_control_outputs(dm_dt, dx_dt, state, glider_params=None):
    """
    Validate control outputs against physical constraints
    Returns: (dm_dt_valid, dx_dt_valid, warnings)
    """
    warnings = []
    
    # Validate ballast control
    if glider_params:
        max_ballast_flow = glider_params.get('max_ballast_flow', 1e-3)
        rho_water = glider_params.get('rho_water', 1025.0)
        max_mass_rate = max_ballast_flow * rho_water
        
        if abs(dm_dt) > max_mass_rate:
            warnings.append(f"Ballast rate {dm_dt:.4f} exceeds max {max_mass_rate:.4f} kg/s")
            dm_dt = np.clip(dm_dt, -max_mass_rate, max_mass_rate)
    
    # Validate MVM control
    if glider_params:
        max_mvm_velocity = 1  # m/s
        if abs(dx_dt) > max_mvm_velocity:
            warnings.append(f"MVM velocity {dx_dt:.4f} exceeds max {max_mvm_velocity:.4f} m/s")
            dx_dt = np.clip(dx_dt, -max_mvm_velocity, max_mvm_velocity)
    
    # Validate against current state
    if len(state) > 13:
        current_fill = state[13]
        if current_fill <= 0.01 and dm_dt < 0:
            warnings.append("Cannot remove ballast: tank nearly empty")
            dm_dt = 0.0
        elif current_fill >= 0.99 and dm_dt > 0:
            warnings.append("Cannot add ballast: tank nearly full")
            dm_dt = 0.0
    
    if len(state) > 14:
        current_mvm_x = state[14]
        if glider_params:
            max_offset = glider_params.get('MVM_length', 0.5) / 2
            if current_mvm_x <= -max_offset + 0.01 and dx_dt < 0:
                warnings.append("Cannot move MVM left: at travel limit")
                dx_dt = 0.0
            elif current_mvm_x >= max_offset - 0.01 and dx_dt > 0:
                warnings.append("Cannot move MVM right: at travel limit")
                dx_dt = 0.0
    
    return dm_dt, dx_dt, warnings

def yo_yo_control(t, state, surface_depth=2, max_depth=50, cycle_time=300, glider_params=None):
    """
    Yo-yo control system that makes the glider alternate between ascending and descending
    
    Args:
        t: Current time (seconds)
        state: Current state vector [x, y, z, qx, qy, qz, qw, u, v, w, p, q, r, ballast_fill, mvm_offset_x, mvm_offset_y, mvm_offset_z]
        surface_depth: Depth considered "surface" (meters, positive down)
        max_depth: Maximum dive depth (meters, positive down)
        cycle_time: Time for one complete yo-yo cycle (seconds)
        glider_params: Optional glider parameters for limit checking
    
    Returns:
        dm_dt: Ballast mass rate (kg/s, positive = add ballast)
        dx_dt: MVM velocity (m/s, positive = move right)
    """
    depth = state[2]
    quat = state[3:7]
    current_fill = state[13] if len(state) > 13 else 0.5
    
    # Calculate phase in the yo-yo cycle (0 to 1)
    cycle_phase = (t % cycle_time) / cycle_time
    
    # Determine if we're in ascent or descent phase
    if cycle_phase < 0.5:
        # Descent phase (0 to 0.5): add ballast to sink
        target_depth = max_depth
        is_descent = True
    else:
        # Ascent phase (0.5 to 1.0): remove ballast to rise
        target_depth = surface_depth
        is_descent = False
    
    # Calculate depth error
    depth_error = target_depth - depth
    
    # Ballast control with proportional control and physics-based limits
    if glider_params:
        max_ballast_flow = glider_params.get('max_ballast_flow', 1e-3)  # m³/s
        rho_water = glider_params.get('rho_water', 1025.0)  # kg/m³
        max_mass_rate = max_ballast_flow * rho_water
        
        # Proportional control with higher gain for yo-yo behavior
        dm_dt = np.clip(depth_error * 0.15, -max_mass_rate, max_mass_rate)
        
        # Prevent over/under-filling
        if current_fill <= 0.01 and dm_dt < 0:  # Nearly empty, can't remove more
            dm_dt = 0.0
        elif current_fill >= 0.99 and dm_dt > 0:  # Nearly full, can't add more
            dm_dt = 0.0
    else:
        # Fallback to original limits if no params provided
        dm_dt = np.clip(depth_error * 0.15, -0.5, 0.5)
    
    # Pitch control to maintain efficient glide during yo-yo
    try:
        euler = R.from_quat(quat).as_euler('xyz')
        pitch = euler[1]  # Pitch angle in radians
        
        # Target pitch depends on phase
        if is_descent:
            target_pitch = np.radians(-15)  # Nose down for descent
        else:
            target_pitch = np.radians(15)   # Nose up for ascent
        
        pitch_error = target_pitch - pitch
        
        # Get current MVM offset from state
        current_mvm_x = state[14] if len(state) > 14 else 0.0
        
        # Calculate MVM control with limits
        if glider_params:
            mvm_length = glider_params.get('MVM_length', 0.5)  # m
            max_mvm_velocity = 0.02  # m/s (reasonable actuator speed)
            
            # Scale factor: 0.01 m/s per radian of pitch error, with physics limits
            dx_dt = np.clip(pitch_error * 0.01, -max_mvm_velocity, max_mvm_velocity)
            
            # Additional limit: prevent MVM from exceeding travel limits
            max_offset = mvm_length / 2
            if current_mvm_x <= -max_offset + 0.01 and dx_dt < 0:  # At left limit
                dx_dt = 0.0
            elif current_mvm_x >= max_offset - 0.01 and dx_dt > 0:  # At right limit
                dx_dt = 0.0
        else:
            # Fallback to original limits if no params provided
            dx_dt = np.clip(pitch_error * 0.01, -0.02, 0.02)
            
    except Exception as e:
        print(f"Pitch control error: {e}")
        dx_dt = 0.0
    
    # Debug output (uncomment for debugging)
    # if t % 10.0 < 0.1:  # Print every ~10 seconds
    #     phase_name = "DESCENT" if is_descent else "ASCENT"
    #     print(f"t={t:.1f}: {phase_name} | depth={depth:.2f}, target={target_depth}, error={depth_error:.2f}, dm_dt={dm_dt:.3f}")
    #     print(f"  pitch={np.degrees(pitch):.1f}°, target={np.degrees(target_pitch):.1f}°, dx_dt={dx_dt:.4f}")
    #     print(f"  ballast_fill={current_fill:.3f}, cycle_phase={cycle_phase:.2f}")
    
    return dm_dt, dx_dt