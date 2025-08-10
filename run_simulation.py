import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from glider_simulation import run_simulation
from glider_controls import depth_pitch_control

# Define glider parameters
params = {
    # Geometry
    'nose_length': 0.1,           # m
    'nose_radius': 0.12,          # m
    'cyl_length': 1.2,            # m
    'hull_radius': 0.,          # m
    'tail_length': 0.1,           # m
    'tail_radius': 0.09,          # m
    'hull_thickness': 0.005,      # m
    'hull_density': 2700.0,       # kg/m^3 (aluminum)

    # Ballast system
    'ballast_radius': 0.05,       # m
    'ballast_length': 0.3,        # m
    'tank_thickness': 0.004,      # m
    'tank_density': 2700.0,       # kg/m^3
    'ballast_position': np.array([0.7, 0, 0]),

    # Mass properties
    'piston_mass': 3.0,           # kg
    'piston_position': np.array([0.6, 0, 0]),
    'piston_travel': 0.3,         # m
    'actuator_mass': 3.0,         # kg
    'actuator_position': np.array([0.6, 0, 0]),
    'fixed_mass': 2,           # kg
    'fixed_position': np.array([0.7, 0, 0]),
    'I_dry_base': np.diag([2.0, 3.0, 1.5]),

    # Hydrodynamics
    'wing_span': 0.3,             # m
    'wing_chord': 0.075,           # m
    'wing_area': 0.0225,            # m^2
    'wing_position': np.array([0.9, 0, 0]),
    'glider_length': 1.8,         # m
    'CD0': 0.1,
    'CL_alpha': 2 * np.pi,        # ~6.28
    'CD_alpha': 0.5,
    'CM0': -0.02,
    'CM_alpha': -0.1
}

# Calculate extra mass needed for neutral buoyancy at 50% ballast fill
rho_water = 1025.0
V_hull = (1/3) * np.pi * params['nose_radius']**2 * params['nose_length'] \
    + np.pi * params['hull_radius']**2 * params['cyl_length'] \
    + (1/3) * np.pi * params['tail_radius']**2 * params['tail_length']
ballast_mass_50 = rho_water * (np.pi * params['ballast_radius']**2 * params['ballast_length']) * 0.5
# Estimate dry mass (without extra mass)
base_dry_mass = (
    (np.pi * params['nose_radius'] * np.sqrt(params['nose_radius']**2 + params['nose_length']**2)
     + 2 * np.pi * params['hull_radius'] * params['cyl_length']
     + np.pi * params['tail_radius'] * np.sqrt(params['tail_radius']**2 + params['tail_length']**2))
    * params['hull_thickness'] * params['hull_density']
    + 2 * np.pi * params['ballast_radius'] * params['ballast_length'] * params['tank_thickness'] * params['tank_density']
    + params['fixed_mass'] + params['actuator_mass'] + params['piston_mass']
)
total_mass_50 = base_dry_mass + ballast_mass_50
buoyant_mass = rho_water * V_hull
extra_mass = buoyant_mass - total_mass_50
if extra_mass > 0:
    params['fixed_mass'] += extra_mass
    print(f"[Buoyancy Calc] Added {extra_mass:.2f} kg to fixed_mass for neutral buoyancy at 50% ballast.")
else:
    params['fixed_mass'] -= extra_mass
    print(f"[Buoyancy Calc] Glider is already heavier than neutral at 50% ballast.")
    print(f"[Buoyancy Calc] removed {extra_mass:.2f} kg to fixed_mass for neutral buoyancy at 50% ballast.")

# Run simulation
solution = run_simulation(
    params, 
    control_func=lambda t, s: depth_pitch_control(t, s, 20, 0),
    t_end=60,
    dt=1,
    init_depth=10,
    init_pitch=5,  # Start with less than neutral ballast and piston at one end
    # These will be handled in glider_physics.py if you add the arguments
    # For now, let's just set depth and pitch, but you can also edit set_initial_conditions for more effect
)
# Debug information
print('--- Simulation Debug Info ---')
print('Simulation success:', solution.success)
print('Solver message:', solution.message)
print('solution.t (time steps):', solution.t)
print('solution.y shape (states x time):', solution.y.shape)
print('Initial state:', solution.y[:, 0] if solution.y.shape[1] > 0 else 'N/A')
if solution.y.shape[1] > 1:
    print('Final state:', solution.y[:, -1])
else:
    print('No state progression (only initial state)')
print('-------------------------------')
# Display final neutral buoyancy calculation
print(f"[Buoyancy Calc] Final fixed_mass: {params['fixed_mass']:.2f} kg")
print(f"[Buoyancy Calc] Neutral buoyancy mass at 50% ballast: {buoyant_mass:.2f} kg")
print(f"[Buoyancy Calc] Total mass at 50% ballast: {total_mass_50 + max(0, extra_mass):.2f} kg")
# Plot results
plt.figure(figsize=(12, 8))

# Depth plot
plt.subplot(2, 1, 1)
plt.plot(solution.t, solution.y[2, :], label='Actual Depth')
plt.plot(solution.t, [20]*len(solution.t), 'r--', label='Desired Depth (Setpoint)')
plt.title('Glider Trajectory')
plt.ylabel('Depth (m)')
plt.gca().invert_yaxis()
plt.grid(True)
plt.legend()

# Pitch plot
plt.subplot(2, 1, 2)
pitch_angles = []
for i in range(len(solution.t)):
    quat = solution.y[3:7, i]
    pitch = R.from_quat(quat).as_euler('zyx')[1]
    pitch_angles.append(np.degrees(pitch))
plt.plot(solution.t, pitch_angles, label='Actual Pitch')
plt.plot(solution.t, [10]*len(solution.t), 'r--', label='Desired Pitch (Setpoint)')
plt.xlabel('Time (s)')
plt.ylabel('Pitch (deg)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()