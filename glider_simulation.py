import numpy as np
from scipy.integrate import solve_ivp
from glider_physics import UnderwaterGlider

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
    glider.reset()  # Ensure state0 is initialized
    
    # Set initial depth (z position, index 2)
    glider.state0[2] = init_depth
    
    # Set initial pitch (quaternion, indices 3:7)
    # Start with identity quaternion, then rotate about y by pitch (degrees)
    from scipy.spatial.transform import Rotation as R
    quat = R.from_euler('y', init_pitch, degrees=True).as_quat()  # [x, y, z, w]
    glider.state0[3:7] = quat
    
    # Time points
    t_eval = np.arange(0, t_end, dt)
    
    # Progress tracking
    if progress_callback:
        progress_callback(10)  # Initialization complete
    
    # Custom progress tracking wrapper
    def dynamics_with_progress(t, y):
        # Calculate progress based on current time
        if progress_callback:
            progress = int(10 + (t / t_end) * 80)  # 10-90% during simulation
            progress_callback(progress)
        return glider.dynamics(t, y)
    
    # Run simulation
    sol = solve_ivp(
        dynamics_with_progress,
        [0, t_end],
        glider.state0,
        t_eval=t_eval,
        method=solver
    )
    
    if progress_callback:
        progress_callback(100)  # Simulation complete
    
    return sol