import numpy as np
from scipy.spatial.transform import Rotation as R
import warnings

class UnderwaterGlider:
    def __init__(self, params):
        self.params = params
        self.rho_water = 1025.0
        self.g = 9.81
        self.dm_ballast_dt = 0.0
        self.dx_p_dt = 0.0
        self.state0 = np.zeros(16)

        # Hydrodynamic coefficients (simplified)
        self.D_lin = np.diag([50, 100, 100])  # Linear damping
        self.D_ang = np.diag([20, 30, 20])    # Angular damping

        # Assign hydrodynamic parameters from params
        self.CL_alpha = self.params['CL_alpha']
        self.CD0 = self.params['CD0']
        self.CD_alpha = self.params['CD_alpha']
        self.CM0 = self.params['CM0']
        self.CM_alpha = self.params['CM_alpha']
        self.wing_area = self.params['wing_area']
        
        # Initialize physical properties
        self._calculate_mass_properties()
        self._calculate_volume_properties()

        # Power and work tracking
        self.pump_power = 0.0   # W, updated every dynamics call
        self.pump_work = 0.0    # J, integrated over time
        self._last_t = None     # for time step tracking

        
    def _calculate_mass_properties(self):
        """Calculate all mass properties from geometry"""
        p = self.params
        
        # Hull mass calculations
        nose_area = np.pi * p['nose_radius'] * np.sqrt(p['nose_radius']**2 + p['nose_length']**2)
        cyl_area = 2 * np.pi * p['hull_radius'] * p['cyl_length']
        tail_area = np.pi * p['tail_radius'] * np.sqrt(p['tail_radius']**2 + p['tail_length']**2)
        
        self.m_hull = (nose_area + cyl_area + tail_area) * p['hull_thickness'] * p['hull_density']
        
        # Ballast tank mass
        tank_area = 2 * np.pi * p['ballast_radius'] * p['ballast_length']
        self.m_tank = tank_area * p['tank_thickness'] * p['tank_density']
        
        # Total dry mass
        self.m_dry = (self.m_hull + self.m_tank + p['fixed_mass'] + 
                     p['actuator_mass'] + p['piston_mass'])
        
        # Calculate component CGs
        self._calculate_centers_of_mass()
        
    def _calculate_centers_of_mass(self):
        """Calculate centers of mass for all components"""
        p = self.params
        
        # Hull CG components
        nose_cg = (2/3) * p['nose_length']
        cyl_cg = p['nose_length'] + 0.5 * p['cyl_length']
        tail_cg = p['nose_length'] + p['cyl_length'] + (1/3) * p['tail_length']
        
        # Weighted average hull CG
        self.hull_cg = (nose_cg * (self.m_hull/3) + cyl_cg * (self.m_hull/3) + 
                       tail_cg * (self.m_hull/3))
        
        # Dry CG (without piston)
        self.cg_dry = (
            self.hull_cg * self.m_hull +
            p['ballast_position'] * self.m_tank +
            p['fixed_position'] * p['fixed_mass'] +
            p['actuator_position'] * p['actuator_mass']
        ) / (self.m_dry - p['piston_mass'])
        
    def _calculate_volume_properties(self):
        """Calculate all volume properties"""
        p = self.params
        
        # Hull volumes
        self.V_nose = (1/3) * np.pi * p['nose_radius']**2 * p['nose_length']
        self.V_cyl = np.pi * p['hull_radius']**2 * p['cyl_length']
        self.V_tail = (1/3) * np.pi * p['tail_radius']**2 * p['tail_length']
        self.V_hull = self.V_nose + self.V_cyl + self.V_tail
        
        # Ballast tank
        self.V_ballast_max = np.pi * p['ballast_radius']**2 * p['ballast_length']
        self.V_ballast_neutral = 0.5 * self.V_ballast_max
        
        # Center of buoyancy
        nose_cb = (3/4) * p['nose_length']
        cyl_cb = p['nose_length'] + 0.5 * p['cyl_length']
        tail_cb = p['nose_length'] + p['cyl_length'] + (1/4) * p['tail_length']
        
        self.cb = np.array([
            (self.V_nose * nose_cb + self.V_cyl * cyl_cb + self.V_tail * tail_cb) / self.V_hull,
            0, 
            0
        ])

    def set_initial_conditions(self, depth=0, pitch=0):
        """Set initial state"""
        self.state0[2] = depth  # z-down
        quat = R.from_euler('y', pitch, degrees=True).as_quat()
        self.state0[3:7] = quat
        self.state0[13] = self.rho_water * self.V_ballast_neutral
        self.state0[14] = 0.5 * self.params['piston_travel']
     

    def dynamics(self, t, state):
        """6-DOF dynamics implementation with debug output"""
        # Unpack state
        pos, quat, vel, omega, m_ballast, x_p = (
            state[0:3], state[3:7], state[7:10], 
            state[10:13], state[13], state[14]
        )
        #quaternion normalization 
        quat_norm = np.linalg.norm(quat)
        if quat_norm > 0:
            quat = quat / quat_norm

        # Apply physical limits
        m_ballast = np.clip(m_ballast, 0, self.rho_water * self.V_ballast_max)
        x_p = np.clip(x_p, 0, self.params['piston_travel'])
        
        # Calculate total mass properties
        m_total = self.m_dry + m_ballast
        denom = (self.m_dry + m_ballast)
        if denom < 1e-6:
            warnings.warn(f"CG denominator near zero at t={t}, m_dry={self.m_dry}, m_ballast={m_ballast}")
            return np.zeros_like(state)
        cg = self.calculate_cg(m_ballast, x_p)
        I_total = self.calculate_inertia(m_ballast, x_p, cg)
        
        # Clamp velocities and angular rates
        vel = np.clip(vel, -10, 10)
        omega = np.clip(omega, -10, 10)
        
        # Rotation matrix
        rot = R.from_quat(quat)
        R_ib = rot.as_matrix()
        
        # Forces and moments
        F_hydro, M_hydro = self.hydrodynamic_forces(vel, omega)
        F_grav, F_buoy, M_buoy = self.buoyancy_gravity_forces(R_ib, m_total, m_ballast, cg)
        # Debug log for forces and moments
        print(f'[Forces Debug] t={t:.4f}, F_grav={F_grav}, F_buoy={F_buoy}, F_hydro={F_hydro}, M_buoy={M_buoy}, M_hydro={M_hydro}')
        
        # Rigid-body dynamics
        F_total = F_grav + F_buoy + F_hydro
        M_total = M_buoy + M_hydro
        dvel_dt = F_total / m_total - np.cross(omega, vel)
        domega_dt = np.linalg.inv(I_total) @ (M_total - np.cross(omega, I_total @ omega))
        
        # Kinematics
        dpos_dt = R_ib.T @ vel
        # Standard quaternion derivative (fixed)
        dquat_dt = self.quaternion_kinematics(quat, omega)
        
        # Actuator limits
        dm_dt, dx_dt = self.apply_actuator_limits(m_ballast, x_p)
        
        # Pad with zeros to match state size if needed
        deriv = np.concatenate((
            dpos_dt, dquat_dt, dvel_dt, domega_dt, [dm_dt, dx_dt]
        ))
        if deriv.shape[0] < state.shape[0]:
            deriv = np.concatenate((deriv, np.zeros(state.shape[0] - deriv.shape[0])))
        elif deriv.shape[0] > state.shape[0]:
            deriv = deriv[:state.shape[0]]
        
        # Check for NaN/inf in state
        if not np.all(np.isfinite(state)):
            warnings.warn(f"NaN or inf detected in state at t={t}")
            return np.zeros_like(state)
        
        # Check I_total condition number
        cond_I = np.linalg.cond(I_total)
        if cond_I > 1e8:
            warnings.warn(f"I_total nearly singular at t={t}, cond={cond_I}")
            return np.zeros_like(state)
        
        # Check for NaN/inf in derivative
        if not np.all(np.isfinite(dvel_dt)) or not np.all(np.isfinite(domega_dt)):
            warnings.warn(f"NaN or inf detected in derivatives at t={t}")
            return np.zeros_like(state)
        
        # Pump work and power calculation
        depth = pos[2]  
        external_pressure = self.rho_water * self.g * abs(depth)  
        ballast_flow_rate = self.dm_ballast_dt / self.rho_water   

        # Instantaneous pump power (positive for pumping in or out)
        self.pump_power = external_pressure * ballast_flow_rate  

        # Integrate pump work over time
        if self._last_t is not None:
            dt = t - self._last_t
            if dt > 0 and np.isfinite(dt):
                self.pump_work += self.pump_power * dt  
        self._last_t = t


        # Print a summary of the state and derivative for debugging
        print(f'[Dynamics Debug] t={t:.4f}, state[0:7]={state[0:7]}, deriv[0:7]={deriv[0:7]}')
        return deriv
    
    def calculate_cg(self, m_ballast, x_p):
        """Compute center of gravity with movable masses"""
        p = self.params
        
        # Ballast water CG (at tank position)
        cg_ballast = m_ballast * p['ballast_position']
        
        # Piston position (along actuator track)
        piston_pos = p['piston_position'] + np.array([x_p, 0, 0])
        cg_piston = p['piston_mass'] * piston_pos
        
        # Total CG (mass-weighted average)
        return (
            self.cg_dry * (self.m_dry - p['piston_mass']) + 
            cg_ballast + 
            cg_piston
        ) / (self.m_dry + m_ballast)
    
    def calculate_inertia(self, m_ballast, x_p, cg_total):
        """Compute inertia tensor about CG"""
        p = self.params
        
        # Base inertia (dry components without piston)
        I_total = p['I_dry_base'].copy()
        
        # Add piston inertia using parallel axis theorem
        piston_pos = p['piston_position'] + np.array([x_p, 0, 0])
        r_piston = piston_pos - cg_total
        I_piston = p['piston_mass'] * (np.linalg.norm(r_piston)**2 * np.eye(3) - 
                   np.outer(r_piston, r_piston))
        
        # Add ballast water inertia
        r_ballast = p['ballast_position'] - cg_total
        I_ballast = m_ballast * (np.linalg.norm(r_ballast)**2 * np.eye(3) - 
                    np.outer(r_ballast, r_ballast))
        
        return I_total + I_piston + I_ballast
    
    def buoyancy_gravity_forces(self, R_ib, m_total, m_ballast, cg):
        """Compute gravity and buoyancy forces"""
        # Gravity force in inertial frame (NED: z-down)
        F_grav_inertial = np.array([0, 0, m_total * self.g])
        F_grav_body = R_ib @ F_grav_inertial
        
        # Buoyancy force (archimedes principle)
        # Note: Ballast water is INSIDE hull so doesn't add displacement
        buoy_force = -self.rho_water * self.V_hull * self.g
        F_buoy_body = R_ib @ np.array([0, 0, buoy_force])
        
        # Buoyancy moment (about CG)
        r_b = self.cb - cg  # Vector from CG to CB
        M_buoy = np.cross(r_b, F_buoy_body)
        
        return F_grav_body, F_buoy_body, M_buoy
    
    # Revised hydrodynamic_forces method
    def hydrodynamic_forces(self, vel, omega):
        V = np.linalg.norm(vel)
        if V < 1e-6:
            return np.zeros(3), np.zeros(3)
        # Clamp V to avoid extreme forces
        V = np.clip(V, 0, 10)
        # Angle of attack and sideslip
        alpha = np.arctan2(vel[2], vel[0])
        # Clamp alpha to [-pi/2, pi/2]
        alpha = np.clip(alpha, -np.pi/2, np.pi/2)
        beta = np.arcsin(np.clip(vel[1]/V, -1, 1))
    
        # Dynamic pressure
        q_dyn = 0.5 * self.rho_water * V**2
    
        # Lift and drag coefficients (clamped)
        CL = np.clip(self.CL_alpha * alpha, -5, 5)
        CD = np.clip(self.CD0 + self.CD_alpha * alpha**2, 0, 5)
    
        # Lift and drag forces in stability axes
        lift = q_dyn * self.wing_area * CL
        drag = q_dyn * self.wing_area * CD
    
        # Convert to body axes
        F_lift = np.array([-lift * np.sin(alpha), 0, -lift * np.cos(alpha)])
        F_drag = np.array([-drag * np.cos(alpha), 0, drag * np.sin(alpha)])
    
        # Damping forces (viscous + added mass approximation)
        F_damp = -self.D_lin @ vel
        M_damp = -self.D_ang @ omega
    
        # Pitch moment
        M_pitch = np.array([0, q_dyn * self.wing_area * self.params['glider_length'] * 
                           (self.CM0 + self.CM_alpha * alpha), 0])
    
        F_hydro = F_lift + F_drag + F_damp
        M_hydro = M_pitch + M_damp
        # Clamp hydrodynamic forces/moments
        F_hydro = np.clip(F_hydro, -1e4, 1e4)
        M_hydro = np.clip(M_hydro, -1e4, 1e4)
    
        return F_hydro, M_hydro
    
    def newton_euler(self, vel, omega, F_total, M_total, mass, I):
        """Rigid-body dynamics equations"""
        # Linear acceleration
        dvel_dt = F_total / mass - np.cross(omega, vel)
        
        # Angular acceleration
        I_inv = np.linalg.inv(I)
        domega_dt = I_inv @ (M_total - np.cross(omega, I @ omega))
        
        return dvel_dt, domega_dt
    
    # Corrected quaternion kinematics
    def quaternion_kinematics(self, quat, omega):
        qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]
        p, q, r = omega
    
        # Quaternion derivative (normalized)
        dqx = 0.5 * (p*qw + q*qz - r*qy)
        dqy = 0.5 * (q*qw - p*qz + r*qx)
        dqz = 0.5 * (r*qw + p*qy - q*qx)
        dqw = 0.5 * (-p*qx - q*qy - r*qz)
    
        return np.array([dqx, dqy, dqz, dqw])
    
    def apply_actuator_limits(self, m_ballast, x_p):
        """Enforce physical actuator constraints"""
        # Ballast pump limits
        dm_dt = self.dm_ballast_dt
        if m_ballast <= 0 and dm_dt < 0:
            dm_dt = 0
        elif m_ballast >= self.rho_water * self.V_ballast_max and dm_dt > 0:
            dm_dt = 0
            
        # Piston travel limits
        dx_dt = self.dx_p_dt
        if x_p <= 0 and dx_dt < 0:
            dx_dt = 0
        elif x_p >= self.params['piston_travel'] and dx_dt > 0:
            dx_dt = 0
            
        return dm_dt, dx_dt

    def dynamics_wrapper(self, t, state, control_func=None):
        """Wrapper for control input handling with debug output"""
        if control_func:
            controls = control_func(t, state)
            self.dm_ballast_dt, self.dx_p_dt = controls
        # Debug print for wrapper
        print(f'[Wrapper Debug] t={t:.4f}, state[0:7]={state[0:7]}, ...')
        deriv = self.dynamics(t, state)
        print(f'[Wrapper Debug] t={t:.4f}, deriv[0:7]={deriv[0:7]}, ...')
        return deriv