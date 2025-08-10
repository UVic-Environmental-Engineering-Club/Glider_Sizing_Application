import numpy as np
from scipy.integrate import solve_ivp
from glider_physics import UnderwaterGlider

def run_simulation(params, control_func=None, t_end=1, dt=1, 
                  init_depth=0, init_pitch=0):
    """
    Run glider simulation with specified parameters and control
    
    Args:
        params: Dictionary of glider parameters
        control_func: Control function (t, state) -> (dm_dt, dx_dt)
        t_end: Simulation duration (seconds)
        dt: Time step (seconds)
        init_depth: Initial depth (meters, positive down)
        init_pitch: Initial pitch angle (degrees)
    
    Returns:
        solution: ODE solution object
    """
    # Initialize glider
    glider = UnderwaterGlider(params)
    glider.set_initial_conditions(depth=init_depth, pitch=init_pitch)
    
    # Time points
    t_eval = np.arange(0, t_end, dt)
    
    # Run simulation
    sol = solve_ivp(
        lambda t, y: glider.dynamics_wrapper(t, y, control_func),
        [0, t_end],
        glider.state0,
        t_eval=t_eval,
        method='Radau'
    )
    
    return sol