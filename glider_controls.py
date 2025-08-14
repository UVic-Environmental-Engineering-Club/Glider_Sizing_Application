import numpy as np
from scipy.spatial.transform import Rotation as R

def depth_pitch_control(t, state, set_depth=20, set_pitch=0):
    """
    Control strategy to maintain desired depth and pitch
    set_depth: Target depth in meters (positive down)
    set_pitch: Target pitch angle in degrees
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
    
    # Ballast control with proportional control and limits
    depth_error = set_depth - depth
    # Scale factor: 0.1 kg/s per meter of depth error, with limits
    dm_dt = np.clip(depth_error * 0.1, -0.5, 0.5)
    
    # Pitch control with proportional control and limits
    pitch_error = np.radians(set_pitch) - pitch
    # Scale factor: 0.01 m/s per radian of pitch error, with limits
    dx_dt = np.clip(pitch_error * 0.01, -0.02, 0.02)
    
    # Debug output (uncomment for debugging)
    # if t % 1.0 < 0.1:  # Print every ~1 second
    #     print(f"t={t:.1f}: depth={depth:.2f}, target={set_depth}, error={depth_error:.2f}, dm_dt={dm_dt:.3f}")
    #     print(f"  pitch={np.degrees(pitch):.1f}°, target={set_pitch}°, error={np.degrees(pitch_error):.1f}°, dx_dt={dx_dt:.4f}")
    
    return dm_dt, dx_dt

def trajectory_following_control(t, state, waypoints):
    """
    Advanced control for following a trajectory
    waypoints: List of (x, y, depth) coordinates
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
    return depth_pitch_control(t, state, set_depth=target[2], set_pitch=desired_pitch)

def simple_depth_control(t, state, set_depth=20):
    """
    Simple proportional depth control only
    """
    depth = state[2]
    depth_error = set_depth - depth
    
    # Proportional control with limits
    dm_dt = np.clip(depth_error * 0.05, -0.3, 0.3)
    dx_dt = 0.0  # No pitch control
    
    return dm_dt, dx_dt