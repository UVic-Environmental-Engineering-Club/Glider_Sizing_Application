from __future__ import annotations
from typing import Dict, Any
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
import warnings


class UnderwaterGlider:
    # ---------------- defaults ----------------
    # ------- Static method to call perams------
    @staticmethod
    def default_params() -> Dict[str, Any]:
        return {
            # Geometry (m)
            'nose_length': 0.4,
            'nose_radius': 0.08,
            'cyl_length': 0.9,
            'hull_radius': 0.08,
            'tail_length': 0.3,
            'tail_radius': 0.04,
            'glider_length': 1.6,

            # Ballast tank geometry
            'ballast_radius': 0.05,
            'ballast_length': 0.2,

            # Structural
            'hull_thickness': 0.005,
            'hull_density': 2700.0,
            'tank_thickness': 0.004,
            'tank_density': 1200.0,

            # Masses
            'fixed_mass': 3.0,
            'MVM_mass': 5,
            'piston_mass': 0.8,

            # Inertia override (not in use atm)
            'I_dry_base': None,

            # Wing area (used in force calculations)
            'wing_area': 0.04,

            # velocity initial conditions
            'u' : 0.0,
            'v' : 0.0,  
            'w' : 0.0,
            
            # angular velocity initial conditions
            'p' : 0.0,  # Roll rate
            'q' : 0.0,  # Pitch rate
            'r' : 0.0,  # Yaw rate

            # Added-mass placeholders
            'added_mass_x': 5.0,
            'added_mass_y': 12.0,
            'added_mass_z': 12.0,
            'added_Ixx': 0.05,
            'added_Iyy': 0.12,
            'added_Izz': 0.12,
            'added_mass_matrix': None,

            # Control limits
            'max_ballast_flow': 1e-3,
            'MVM_length': 0.5,
            'current_fill':0,

            # Watertight options for the nose and tail
            'nose_Watertight': True,
            'tail_Watertight': True,

            # Environment
            'rho_water': 1025.0,
            'g': 9.81,
        }

    # ---------------- construction ----------------
    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        merged = self.default_params().copy()
        if params:
            merged.update(params)
        self.params = merged
        
        # Convert frequently accessed parameters to instance attributes for speed
        # Environment
        self.rho_water = float(self.params['rho_water'])
        self.g = float(self.params['g'])
        
        # Geometry (frequently used in calculations)
        self.nose_length = float(self.params['nose_length'])
        self.nose_radius = float(self.params['nose_radius'])
        self.cyl_length = float(self.params['cyl_length'])
        self.hull_radius = float(self.params['hull_radius'])
        self.tail_length = float(self.params['tail_length'])
        self.tail_radius = float(self.params['tail_radius'])
        self.glider_length = float(self.params['glider_length'])
        
        # Ballast tank geometry
        self.ballast_radius = float(self.params['ballast_radius'])
        self.ballast_length = float(self.params['ballast_length'])
        
        # Structural properties
        self.hull_thickness = float(self.params['hull_thickness'])
        self.hull_density = float(self.params['hull_density'])
        self.tank_thickness = float(self.params['tank_thickness'])
        self.tank_density = float(self.params['tank_density'])
        
        # Masses
        self.fixed_mass = float(self.params['fixed_mass'])
        self.MVM_mass = float(self.params['MVM_mass'])
        self.piston_mass = float(self.params['piston_mass'])
        
        # Wing area (used in force calculations)
        self.wing_area = float(self.params.get('wing_area', 0.04))
        
        # Velocity initial conditions
        self.u = float(self.params.get('u', 0.0))
        self.v = float(self.params.get('v', 0.0))
        self.w = float(self.params.get('w', 0.0))
        
        # Angular velocity initial conditions (p, q, r)
        self.p = float(self.params.get('p', 0.0))  # Roll rate
        self.q = float(self.params.get('q', 0.0))  # Pitch rate  
        self.r = float(self.params.get('r', 0.0))  # Yaw rate
        
        # Added mass parameters
        self.added_mass_x = float(self.params.get('added_mass_x', 5.0))
        self.added_mass_y = float(self.params.get('added_mass_y', 12.0))
        self.added_mass_z = float(self.params.get('added_mass_z', 12.0))
        self.added_Ixx = float(self.params.get('added_Ixx', 0.05))
        self.added_Iyy = float(self.params.get('added_Iyy', 0.12))
        self.added_Izz = float(self.params.get('added_Izz', 0.12))
        
        # Control limits
        self.max_ballast_flow = float(self.params.get('max_ballast_flow', 1e-3))
        self.MVM_length = float(self.params.get('MVM_length', 0.5))
        
        # Watertight sections (for buoyancy calculations)
        self.nose_Watertight = bool(self.params.get('nose_Watertight', True))
        self.tail_Watertight = bool(self.params.get('tail_Watertight', True))
        
        # Base positions for dynamic components
        # These positions define the geometric center of each component in 3D space
        # 
        # Moving Mass (MVM): Base position is the geometric center of the MVM when at neutral offset
        # - X: Distance from nose tip to center of MVM
        # - Y: Lateral offset from centerline (0 = centered)
        # - Z: Vertical offset from centerline (0 = centered)
        # The MVM can move from this base position by the offset amount
        self.mvm_base_position = self._vec(self.params.get('Moving_Mass_base_position', np.array([0.5, 0.0, 0.0])))
        
        # Ballast Tank: Base position is the geometric center of the ballast tank
        # - X: Distance from nose tip to center of ballast tank
        # - Y: Lateral offset from centerline (0 = centered)
        # - Z: Vertical offset from centerline (0 = centered)
        # The ballast tank itself doesn't move, only the water level changes
        self.ballast_base_position = self._vec(self.params.get('ballast_base_position', np.array([0.6, 0.0, 0.0])))
        
        # Current offsets from base positions
        self.mvm_offset = np.array([0.0, 0.0, 0.0])  # Offset from base position
        
        # Pre-calculate frequently used values
        self._hull_radius_sq = self.hull_radius**2
        self._ballast_radius_sq = self.ballast_radius**2
        self._pi = np.pi

        # Ballast/pump state
        self.fill_fraction = float(self.params.get('current_fill', 0.0))  # 0..1
        self.pump_on = False
        self.pump_direction = +1  # +1 = fill, -1 = empty

        # Diagnostics
        self._last_t = None
        self.pump_power = 0.0
        self.pump_work = 0.0

        # Derived placeholders
        self.mass = None
        self.cg = None
        self.I_body = None  # dynamic inertia about CG
        self.V_hull = None
        self.cb = None
        
        # Attitude (quaternion: [x, y, z, w])
        self.attitude = np.array([0.0, 0.0, 0.0, 1.0])  # Default: level attitude

        # Position and orientation state (inertial frame)
        self.position_inertial = np.array([0.0, 0.0, 0.0])  # [x, y, z] in inertial frame
        
        # State vector for 6-DOF dynamics
        self.state_vector = np.zeros(17)  # [x, y, z, qx, qy, qz, qw, u, v, w, p, q, r, ballast_fill, mvm_offset_x, mvm_offset_y, mvm_offset_z]

        # Simulation state
        self.time = 0.0
        self.state_history = []
        self.time_history = []

        # start calculations
        self._setup_cfd_table()
        self._calculate_mass_properties()
        self._calculate_inertia()
        self._build_added_mass()

    def _setup_cfd_table(self):
        """Setup CFD table for drag coefficients vs angle of attack."""
        # Default CFD table
        # Format: [AoA_deg, Cd, CL, CM]
        # AoA in degrees, positive = nose up
        self.cfd_table = np.array([
            [-90,  1.2,  0.0,  0.0],   # Nose down
            [-60,  1.1, -0.5,  0.1],
            [-45,  1.0, -0.8,  0.2],
            [-30,  0.9, -1.0,  0.3],
            [-15,  0.85, -1.1, 0.4],
            [0,    0.8,  0.0,  0.0],   # Level flight
            [15,   0.85, 1.1,  -0.4],
            [30,   0.9,  1.0,  -0.3],
            [45,   1.0,  0.8,  -0.2],
            [60,   1.1,  0.5,  -0.1],
            [90,   1.2,  0.0,  0.0],   # Nose up
        ])
        
        # Create interpolation functions for each coefficient
        aoa_deg = self.cfd_table[:, 0]
        self._interp_cd = interp1d(aoa_deg, self.cfd_table[:, 1], kind='linear', 
                                  bounds_error=False, fill_value=(self.cfd_table[0, 1], self.cfd_table[-1, 1]))
        self._interp_cl = interp1d(aoa_deg, self.cfd_table[:, 2], kind='linear', 
                                  bounds_error=False, fill_value=(self.cfd_table[0, 2], self.cfd_table[-1, 2]))
        self._interp_cm = interp1d(aoa_deg, self.cfd_table[:, 3], kind='linear', 
                                  bounds_error=False, fill_value=(self.cfd_table[0, 3], self.cfd_table[-1, 3]))




    # ---------------- mass & CG ----------------
    def _calculate_mass_properties(self):
        """
        Calculate mass properties and 3D center of gravity for the underwater glider.
        
        Coordinate System:
        - Origin (0,0,0): Nose tip of the glider
        - X-axis: Forward direction (positive = forward from nose)
        - Y-axis: Lateral direction (positive = right when looking forward)
        - Z-axis: Vertical direction (positive = up when level)
        
        All positions are calculated relative to the nose tip and can be offset
        for asymmetric geometries using custom CG offset parameters.
        """
        # --- Hull (thin-shell approximations) ---
        # Nose cone: Use elliptical cone approximation for more accurate geometry
        # Surface area of elliptical cone: π * r * √(r² + L²) where r is base radius
        nose_area = self._pi * self.nose_radius * np.sqrt(self.nose_radius**2 + self.nose_length**2)
        cyl_area  = 2.0 * self._pi * self.hull_radius * self.cyl_length
        tail_area = self._pi * self.tail_radius * np.sqrt(self.tail_radius**2 + self.tail_length**2)

        m_nose = nose_area * self.hull_thickness * self.hull_density
        m_cyl  = cyl_area  * self.hull_thickness * self.hull_density
        m_tail = tail_area * self.hull_thickness * self.hull_density
        m_hull = m_nose + m_cyl + m_tail

        # --- Ballast tank geometry ---
        r_inner = self.ballast_radius - self.tank_thickness
        tank_volume = self._pi * (r_inner**2) * self.ballast_length   # internal fillable volume
        shell_area  = 2.0 * self._pi * self.ballast_radius * self.ballast_length
        
        
        end_cap_area = self._pi * self.ballast_radius**2
        total_tank_area = shell_area + end_cap_area
        
        m_tank_shell = total_tank_area * self.tank_thickness * self.tank_density

        # --- Dynamic ballast water mass ---
        V_ballast = np.clip(self.fill_fraction, 0.0, 1.0) * tank_volume
        m_ballast_water = self.rho_water * V_ballast

        # --- Other fixed bits ---
        m_fixed = self.fixed_mass
        m_MVM   = self.MVM_mass

        mass = m_hull + m_tank_shell + m_ballast_water + m_fixed + m_MVM

        # --- 3D Positions for all components ---
        # Calculate actual 3D CG positions for hull components based on geometry
        nose_cg = self._calculate_hull_component_cg('nose')
        cyl_cg  = self._calculate_hull_component_cg('cyl')
        tail_cg = self._calculate_hull_component_cg('tail')

        # Calculate hull CG in 3D space
        hull_cg = (nose_cg * m_nose + cyl_cg * m_cyl + tail_cg * m_tail) / max(m_hull, 1e-9)

        # Ballast position
        ballast_pos = self._calculate_ballast_cg()
        
        # Fixed mass position in 3D space
        # This represents the geometric center of the fixed mass component
        # - X: Distance from nose tip to center of fixed mass
        # - Y: Lateral offset from centerline (0 = centered)
        # - Z: Vertical offset from centerline (0 = centered)
        fixed_pos = self._vec(self.params.get('fixed_position', np.array([0.4, 0.0, 0.0])))
        
        # Moving mass position in 3D space (base + current offset)
        mvm_pos = self._calculate_mvm_cg()

        # Calculate 3D CG using mass-weighted average of all component positions
        numerator = (hull_cg * m_hull +
                    ballast_pos * (m_tank_shell + m_ballast_water) +
                    fixed_pos * m_fixed +
                    mvm_pos * m_MVM)
        cg = numerator / mass

        # Store for inertia step
        self._m_parts = dict(
            m_nose=m_nose, m_cyl=m_cyl, m_tail=m_tail, m_hull=m_hull,
            m_tank_shell=m_tank_shell, m_ballast_water=m_ballast_water,
            m_fixed=m_fixed, m_MVM=m_MVM
        )
        self._cg_parts = dict(
            nose_cg=nose_cg, cyl_cg=cyl_cg, tail_cg=tail_cg, hull_cg=hull_cg,
            ballast_pos=ballast_pos, fixed_pos=fixed_pos, Moving_Mass_pos=mvm_pos
        )

        self.mass = float(mass)
        self.cg = cg 
        self._tank_volume = tank_volume

    # The control system is only able to issue two commands to the pump
    # Its can change the pump rate and  and turn it on or off
    # to simplify this we will consider only on or off state

    # ---------------- Ballast integration to controls  ----------------
    # this is a work around but should be fixed in the future
    
    def step_ballast(self, dt: float):
        """Advance ballast state given pump ON/OFF."""
        if not self.pump_on:
            return

        # Rate of change of volume (m³/s), limited by pump capacity
        # NOTE: Pump performance is depth-dependent
        # need to implement depth dependent pump performance
        dV_dt = self.pump_direction * self.max_ballast_flow
        d_fill = (dV_dt * dt) / self._tank_volume

        # Calculate new fill fraction
        new_fill = self.fill_fraction + d_fill
        
        # Apply proper clipping to prevent negative values
        if new_fill < 0.0:
            # If trying to remove more than available, stop at 0 and turn pump off
            self.fill_fraction = 0.0
            self.pump_on = False
            print(f"Warning: Ballast tank empty, stopping pump. Fill: {self.fill_fraction:.3f}")
        elif new_fill > 1.0:
            # If trying to add more than capacity, stop at 1 and turn pump off
            self.fill_fraction = 1.0
            self.pump_on = False
            print(f"Warning: Ballast tank full, stopping pump. Fill: {self.fill_fraction:.3f}")
        else:
            self.fill_fraction = new_fill
        
        # Update pump diagnostics
        self.pump_power = abs(dV_dt) * 101325  # Power = flow rate * atmospheric pressure (Pa)
        self.pump_work += self.pump_power * dt  # Cumulative work done

    def set_pump_state(self, pump_on: bool, pump_direction: int = None):
        """
        Set pump state and direction.
        
        Args:
            pump_on: Whether pump is active
            pump_direction: +1 for fill, -1 for empty (only set if pump_on is True)
        """
        self.pump_on = pump_on
        if pump_on and pump_direction is not None:
            self.pump_direction = np.clip(pump_direction, -1, 1)
        elif not pump_on:
            self.pump_direction = 0  # No direction when pump is off

    def set_pump_from_mass_rate(self, dm_dt: float):
        """
        Set pump state based on desired mass rate.
        
        Args:
            dm_dt: Desired mass rate in kg/s (positive = add ballast, negative = remove)
        """
        if abs(dm_dt) < 1e-6:  # Very small rate, turn pump off
            self.set_pump_state(False)
            return
        
        # Determine pump direction from mass rate
        if dm_dt > 0:  # Adding ballast
            self.set_pump_state(True, +1)
        else:  # Removing ballast
            self.set_pump_state(True, -1)
        
        # Note: The actual flow rate is limited by max_ballast_flow in step_ballast()
    
    def reset_ballast_system(self):
        """Reset ballast system to safe state if it gets corrupted."""
        if self.fill_fraction < 0.0 or self.fill_fraction > 1.0:
            print(f"Warning: Resetting corrupted ballast fill from {self.fill_fraction:.3f} to 0.5")
            self.fill_fraction = 0.5
            self.pump_on = False
            self.pump_direction = 0
            self.pump_power = 0.0
            self.pump_work = 0.0


    # ---------------- Inertia ----------------
    def _calculate_inertia(self):
        parts = self._m_parts
        cgs   = self._cg_parts

        I_total = np.zeros((3, 3))

        # Hull sections as thin cylindrical shells
        for m_key, cg_key, geom in [
            ('m_nose', 'nose_cg', (self.nose_radius, self.nose_length)),
            ('m_cyl',  'cyl_cg',  (self.hull_radius, self.cyl_length)),
            ('m_tail', 'tail_cg', (self.tail_radius, self.tail_length)),
        ]:
            m = parts[m_key]
            r, L = geom
            I_local = self._thin_cyl_inertia(m, r, L)
            d = cgs[cg_key] - self.cg
            I_total += self._parallel_axis(I_local, m, d)

        # Tank shell
        m = parts['m_tank_shell']
        d = cgs['ballast_pos'] - self.cg
        I_total += self._parallel_axis(np.zeros((3,3)), m, d)

        # Ballast water
        m = parts['m_ballast_water']
        d = cgs['ballast_pos'] - self.cg
        I_total += self._parallel_axis(np.zeros((3,3)), m, d)

        # Fixed block
        m = parts['m_fixed']
        d = cgs['fixed_pos'] - self.cg
        I_total += self._parallel_axis(np.zeros((3,3)), m, d)

        # Moving mass (MVM)
        m = parts['m_MVM']
        d = cgs['Moving_Mass_pos'] - self.cg
        I_total += self._parallel_axis(np.zeros((3,3)), m, d)

        self.I_body = I_total

    def _build_added_mass(self):
        """Build the added mass matrix for the glider."""
        # Create diagonal added mass matrix
        self.added_mass_matrix = np.diag([
            self.added_mass_x,
            self.added_mass_y, 
            self.added_mass_z,
            self.added_Ixx,
            self.added_Iyy,
            self.added_Izz
        ])

    # ---------------- Bouyancy ----------------
    def _estimate_hull_volume_and_cb(self):
        """Estimate external hull displaced volume and center of buoyancy (in body coords)."""
        # Cylinder volume
        V_cyl = self._pi * self._hull_radius_sq * self.cyl_length
        cyl_cg_x = self.nose_length + 0.5 * self.cyl_length

        # Nose approximated as cone for volume
        V_nose = (1.0/3.0) * self._pi * self.nose_radius**2 * self.nose_length
        nose_cg_x = (1.0/3.0) * self.nose_length  

        # Tail approximated as cone for volume
        V_tail = (1.0/3.0) * self._pi * self.tail_radius**2 * self.tail_length
        tail_cg_x = self.nose_length + self.cyl_length + (2.0/3.0) * self.tail_length

        
        # NOTE: Buoyancy is based on external geometry displacement
        # The nose and tail could not be watertight, but the cylinder is
        # self.nose_Watertight and self.tail_Watertight are boolean flags that are set in the constructor/GUI
        if self.nose_Watertight == False:
            V_nose = 0.0
        if self.tail_Watertight == False:
            V_tail = 0.0
        
        V_total = V_cyl + V_nose + V_tail
        
        # center of buoyancy x-coordinate (in body frame) - use external hull for CB calculation
        x_cb = (V_nose * nose_cg_x + V_cyl * cyl_cg_x + V_tail * tail_cg_x) / max(V_total, 1e-12)
        self.cb = np.array([x_cb, 0.0, 0.0])
        
        # Store volume for later use
        self.V_hull = V_total

    
    # ---------------- Static forces ----------------
    def compute_forces_and_moments(self):
        """
        Compute external force & moment on the body (in body frame).
        Uses current velocity and angular velocity stored in the object.
        Returns dict with F_body, M_body and components.
        """
        rho = self.rho_water
        vel_body = np.array([self.u, self.v, self.w])
        omega_body = np.array([self.p, self.q, self.r])

        # convert to body frame using current rotation
        # create rotation matrix from quaternion
        R_ib = R.from_quat(self.attitude).as_matrix().T  # inertial->body = R^T


        #Weight (acts at CG) in inertial downwards -> convert to body
        W = self.mass * self.g
        F_weight_inertial = np.array([0.0, 0.0, W])  # N (z-down in inertial frame)
        
        F_weight_body = R_ib @ F_weight_inertial

        # Buoyancy: 
        # assume full immersion of hull (static forces at semi submerged state arent needed to be modeled))
        self._estimate_hull_volume_and_cb()  # Updates self.V_hull and self.cb
        F_buoy_inertial = np.array([0.0, 0.0, -rho * self.V_hull * self.g])  # upward in inertial (+z up)
        # convert to body
        F_buoy_body = R_ib @ F_buoy_inertial

        # CFD table-based hydrodynamic drag and lift forces
        # Reference area: use frontal area for drag calculations
        A_ref = self._pi * self._hull_radius_sq  # Frontal area
        
        # Calculate angle of attack from body velocity
        aoa_deg = self.calculate_angle_of_attack(vel_body)
        
        # Get coefficients from CFD table based on AoA
        coeffs = self.get_drag_coefficients(aoa_deg)
        Cd = coeffs['Cd']
        CL = coeffs['CL']
        CM = coeffs['CM']

        u, v, w = vel_body
        V_mag = np.linalg.norm(vel_body)
        
        # Calculate drag force using single coefficient and proper flow direction projection
        if V_mag > 1e-6:
            # Drag force acts opposite to velocity direction
            # F_drag = -0.5 * rho * Cd * A_ref * V_mag^2 * (velocity_unit_vector)
            vel_unit = vel_body / V_mag
            F_drag_magnitude = 0.5 * rho * Cd * A_ref * V_mag**2
            F_drag_body = -F_drag_magnitude * vel_unit
            
            # Lift force (acts perpendicular to flow direction)
            # For underwater gliders, lift primarily acts in the vertical plane
            # Lift coefficient is based on AoA and acts perpendicular to the flow
            F_lift_magnitude = 0.5 * rho * CL * A_ref * V_mag**2
            
            # Calculate lift direction: perpendicular to velocity in the vertical plane
            # For a glider, lift acts in the z-direction relative to the body frame
            # when there's a component of velocity in the x-direction
            if abs(u) > 1e-6:  # Only apply lift if there's forward motion
                # Lift acts in z-direction (up/down relative to body)
                F_lift_body = np.array([0.0, 0.0, F_lift_magnitude])
            else:
                F_lift_body = np.array([0.0, 0.0, 0.0])
        else:
            F_drag_body = np.array([0.0, 0.0, 0.0])
            F_lift_body = np.array([0.0, 0.0, 0.0])
        
        F_hydro_body = F_drag_body + F_lift_body

        # NOTE: rotational damping needs to be improved this is bad approximation
        rot_damping = 0.1
        M_damp = -rot_damping * omega_body

        # Restoring moment from offset between CB and CG
        #
        # Buoyancy acts at CB (body coords), weight acts at CG (body coords => location zero relative to CG)
        # moment = (r_cb - r_cg) x F_buoy_body
        r_cb_rel = self.cb - self.cg  # both in body coords (self.cg is in body coords too if you use body frame)
    
        M_buoy = np.cross(r_cb_rel, F_buoy_body)

        # 6) Sum forces and moments
        F_total_body = F_weight_body + F_buoy_body + F_hydro_body
        M_total_body = M_buoy + M_damp

        # Store  for reference
        self.F_body = F_total_body
        self.M_body = M_total_body
        self.F_weight_body = F_weight_body
        self.F_buoy_body = F_buoy_body
        self.F_drag_body = F_drag_body
        self.F_lift_body = F_lift_body
        self.F_hydro_body = F_hydro_body
        self.M_buoy = M_buoy
        self.M_damp = M_damp
        self.aoa_deg = aoa_deg
        self.drag_coeffs = coeffs
        self.vel_body = vel_body
        self.omega_body = omega_body

    #method for printing all in inertial frame forces and moments

    def debug_forces(self):
        """Debug output showing all forces and their directions"""
        print("\n=== FORCE DEBUG OUTPUT ===")
        print(f"Mass: {self.mass:.3f} kg")
        print(f"Gravity: {self.g:.3f} m/s²")
        print(f"Hull Volume: {self.V_hull:.6f} m³")
        print(f"Water Density: {self.rho_water:.1f} kg/m³")
        
        # Weight force
        W = self.mass * self.g
        print(f"\nWEIGHT FORCE:")
        print(f"  Magnitude: {W:.3f} N")
        print(f"  Inertial direction: [0, 0, -{W:.3f}] (z-down)")
        print(f"  Body frame: [{self.F_weight_body[0]:.3f}, {self.F_weight_body[1]:.3f}, {self.F_weight_body[2]:.3f}]")
        
        # Buoyancy force
        B = self.rho_water * self.V_hull * self.g
        print(f"\nBUOYANCY FORCE:")
        print(f"  Magnitude: {B:.3f} N")
        print(f"  Inertial direction: [0, 0, {B:.3f}] (z-up)")
        print(f"  Body frame: [{self.F_buoy_body[0]:.3f}, {self.F_buoy_body[1]:.3f}, {self.F_buoy_body[2]:.3f}]")
        
        # Net vertical force
        net_vertical = B - W
        print(f"\nNET VERTICAL FORCE:")
        print(f"  Buoyancy - Weight = {B:.3f} - {W:.3f} = {net_vertical:.3f} N")
        print(f"  {'POSITIVE (should float up)' if net_vertical > 0 else 'NEGATIVE (should sink down)'}")
        
        # Total force
        print(f"\nTOTAL FORCE (Body Frame):")
        print(f"  F_total = [{self.F_body[0]:.3f}, {self.F_body[1]:.3f}, {self.F_body[2]:.3f}] N")
        print(f"  Magnitude: {np.linalg.norm(self.F_body):.3f} N")
        
        # Current attitude
        euler = R.from_quat(self.attitude).as_euler('xyz', degrees=True)
        print(f"\nCURRENT ATTITUDE:")
        print(f"  Roll: {euler[0]:.1f}°")
        print(f"  Pitch: {euler[1]:.1f}°")
        print(f"  Yaw: {euler[2]:.1f}°")
        
        # CG and CB positions
        print(f"\nPOSITIONS:")
        print(f"  CG (body frame): [{self.cg[0]:.3f}, {self.cg[1]:.3f}, {self.cg[2]:.3f}] m")
        print(f"  CB (body frame): [{self.cb[0]:.3f}, {self.cb[1]:.3f}, {self.cb[2]:.3f}] m")
        print(f"  CB-CG offset: [{self.cb[0]-self.cg[0]:.3f}, {self.cb[1]-self.cg[1]:.3f}, {self.cb[2]-self.cg[2]:.3f}] m")
        
        # Current velocity
        print(f"\nCURRENT VELOCITY (Body Frame):")
        print(f"  [u, v, w] = [{self.u:.3f}, {self.v:.3f}, {self.w:.3f}] m/s")
        
        print("=" * 40)

    # ---------------- 6-DOF Physics Dynamics ----------------
    def _compute_acceleration(self) -> np.ndarray:
        """Compute linear acceleration in body frame."""
        # Total force in body frame
        F_total = self.F_body
        
        # Added mass effects
        F_added_mass = -self.added_mass_matrix[0:3, 0:3] @ np.array([self.u, self.v, self.w])
        
        # Coriolis and centripetal forces
        omega = np.array([self.p, self.q, self.r])
        F_coriolis = -np.cross(omega, np.array([self.u, self.v, self.w])) * self.mass
        
        # Total acceleration
        a_total = (F_total + F_added_mass + F_coriolis) / self.mass
        
        return a_total

    def _compute_angular_acceleration(self) -> np.ndarray:
        """Compute angular acceleration in body frame."""
        # Total moment in body frame
        M_total = self.M_body
        
        # Added mass moment effects
        M_added_mass = -self.added_mass_matrix[3:6, 3:6] @ np.array([self.p, self.q, self.r])
        
        # Gyroscopic effects: -ω × I_body * ω
        omega = np.array([self.p, self.q, self.r])
        M_gyro = -np.cross(omega, self.I_body @ omega)
        
        # Total angular acceleration
        alpha_total = np.linalg.solve(self.I_body, M_total + M_added_mass + M_gyro)
        
        return alpha_total

    def _quaternion_rate(self, q: np.ndarray, omega: np.ndarray) -> np.ndarray:
        """
        Compute quaternion rate from angular velocity.
        Args:
            q: Quaternion [x, y, z, w]
            omega: Angular velocity [p, q, r]
        Returns:
            Quaternion derivative [q̇x, q̇y, q̇z, q̇w]
        """
        # Quaternion rate equation: q̇ = 0.5 * Q * ω
        Q = np.array([
            [ q[3], -q[2],  q[1]],
            [ q[2],  q[3], -q[0]],
            [-q[1],  q[0],  q[3]],
            [-q[0], -q[1], -q[2]]
        ])
        
        return 0.5 * Q @ omega

    def is_mvm_at_limit(self, direction: str = 'any') -> bool:
        """
        Check if the MVM is at its travel limit.
        
        Args:
            direction: 'x', 'y', 'z', or 'any' to check all directions
            
        Returns:
            True if MVM is at limit in specified direction(s)
        """
        max_offset = self.MVM_length / 2
        
        if direction == 'any':
            return np.any(np.abs(self.mvm_offset) >= max_offset)
        elif direction == 'x':
            return abs(self.mvm_offset[0]) >= max_offset
        elif direction == 'y':
            return abs(self.mvm_offset[1]) >= max_offset
        elif direction == 'z':
            return abs(self.mvm_offset[2]) >= max_offset
        else:
            raise ValueError("direction must be 'x', 'y', 'z', or 'any'")


    #---------------- generalized calculations and setup ----------------

    @staticmethod
    def _thin_cyl_inertia(m: float, r: float, L: float) -> np.ndarray:
        """Calculate inertia tensor for thin cylindrical shell."""
        Ixx = m * r**2
        Iyy = 0.5 * m * r**2 + (1.0/12.0) * m * L**2
        Izz = Iyy
        return np.diag([Ixx, Iyy, Izz])

    @staticmethod
    def _parallel_axis(I_local: np.ndarray, m: float, d: np.ndarray) -> np.ndarray:
        """Apply parallel axis theorem to transfer inertia to new point."""
        d = np.asarray(d, dtype=float).reshape(3)
        d2 = float(d @ d)
        return I_local + m * ((d2) * np.eye(3) - np.outer(d, d))

    def _vec(self, v):
        """Convert input to 3D vector, handling scalars and arrays."""
        a = np.asarray(v, dtype=float)
        if a.size == 1:
            return np.array([a.item(), 0.0, 0.0])
        return a.reshape(3)


    def calculate_angle_of_attack(self, vel_body: np.ndarray) -> float:
        """
        Calculate angle of attack from body velocity.
        
        Args:
            vel_body: Velocity vector in body frame [u, v, w]
            
        Returns:
            Angle of attack in degrees (positive = nose up)
        """
        u, v, w = vel_body
        
        # Calculate total velocity magnitude
        V_mag = np.linalg.norm(vel_body)
        if V_mag < 1e-6:  # Very slow or stationary
            return 0.0
            
        # Angle of attack is arctan(w/u) in body frame
        # Positive w = upward motion = nose up = positive AoA
        aoa_rad = np.arctan2(w, u)
        aoa_deg = np.degrees(aoa_rad)
        
        return aoa_deg

    
    def compute_state_derivatives(self, t: float, state: np.ndarray, dt: float = 0.1) -> np.ndarray:
        """
        Compute derivatives for ODE integration.
        Args:
            t: Current time
            state: [x, y, z, qx, qy, qz, qw, u, v, w, p, q, r, ballast_fill, mvm_offset_x, mvm_offset_y, mvm_offset_z]
            dt: Time step for rate limiting calculations
        Returns:
            Derivatives [ẋ, ẏ, ż, q̇x, q̇y, q̇z, q̇w, u̇, v̇, ẇ, ṗ̇, q̇, ṙ̇, ballast_fill_rate, mvm_offset_rate_x, mvm_offset_rate_y, mvm_offset_rate_z]
        """
        # Set current state
        self.set_state_vector(state)
        
        # Compute forces and moments
        self.compute_forces_and_moments()
        
        # Calculate accelerations (including added mass effects)
        accel_body = self._compute_acceleration()
        alpha_body = self._compute_angular_acceleration()
        
        # Quaternion rate equation
        q_dot = self._quaternion_rate(self.attitude, np.array([self.p, self.q, self.r]))
        
        # Transform body velocity to inertial frame
        R_bi = R.from_quat(self.attitude).as_matrix()
        pos_dot = R_bi @ np.array([self.u, self.v, self.w])
        
        # Ballast fill rate (from pump state)
        ballast_fill_rate = 0.0
        if self.pump_on and self.pump_direction != 0:
            dV_dt = self.pump_direction * self.max_ballast_flow
            ballast_fill_rate = (dV_dt) / self._tank_volume
            
            # Apply additional limits to prevent over/under-filling
            current_fill = self.fill_fraction
            if current_fill <= 0.01 and ballast_fill_rate < 0:
                # Nearly empty, can't remove more
                ballast_fill_rate = 0.0
            elif current_fill >= 0.99 and ballast_fill_rate > 0:
                # Nearly full, can't add more
                ballast_fill_rate = 0.0
            
            # Final safety check: ensure the rate won't cause fill to exceed bounds
            if current_fill + ballast_fill_rate * dt > 1.0:
                ballast_fill_rate = max(0.0, (1.0 - current_fill) / dt)
            elif current_fill + ballast_fill_rate * dt < 0.0:
                ballast_fill_rate = min(0.0, -current_fill / dt)
        
        # MVM offset rate (assume controlled externally, default to zero)
        mvm_offset_rate = np.array([0.0, 0.0, 0.0])
        
        # Use the desired rate from control system if available
        if hasattr(self, '_desired_mvm_rate'):
            mvm_offset_rate[0] = self._desired_mvm_rate
        else:
            self._desired_mvm_rate = 0.0
        
        # Enforce travel limits by clipping the rate
        current_offset = self.mvm_offset
        max_offset = self.MVM_length / 2
        
        # If at limits, prevent further movement in that direction
        for i in range(3):
            if current_offset[i] >= max_offset and mvm_offset_rate[i] > 0:
                mvm_offset_rate[i] = 0.0
            elif current_offset[i] <= -max_offset and mvm_offset_rate[i] < 0:
                mvm_offset_rate[i] = 0.0
        
        return np.concatenate([pos_dot, q_dot, accel_body, alpha_body, [ballast_fill_rate], mvm_offset_rate])

    def _calculate_hull_component_cg(self, component_type: str) -> np.ndarray:
        """
        Calculate the 3D CG position of a hull component based on its actual geometry.
        
        Args:
            component_type: 'nose', 'cyl', or 'tail'
            
        Returns:
            np.ndarray: 3D CG position [x, y, z] in body frame (meters)
        """
        if component_type == 'nose':
            # Nose cone: elliptical cone with base radius = hull_radius, tip radius = 0
            # Using elliptical cone approximation for more accurate geometry modeling
            # Geometry: Starts at X=0 (nose tip) and extends to X=nose_length
            # CG of a cone is at 1/4 height from base (3/4 from tip)
            # - X: 0.75 * nose_length (measured from nose tip)
            # - Y: 0.0 (assumed centered on centerline, can be offset for asymmetry)
            # - Z: 0.0 (assumed centered on centerline, can be offset for asymmetry)
            x = 0.75 * self.nose_length
            
            # For a cone, if there's any asymmetry in Y or Z, it would be here
            # For now, assuming the cone is perfectly symmetric around the X-axis
            # but this can be extended for actual asymmetric geometries
            y = 0.0
            z = 0.0
            
            base_cg = np.array([x, y, z])
            
        elif component_type == 'cyl':
            # Cylindrical section: CG is at geometric center
            # Geometry: Extends from X=nose_length to X=nose_length+cyl_length
            # - X: nose_length + 0.5*cyl_length (center of cylinder)
            # - Y: 0.0 (centered on axis of symmetry)
            # - Z: 0.0 (centered on axis of symmetry)
            x = self.nose_length + 0.5 * self.cyl_length
            y = 0.0  # Centered on axis of symmetry
            z = 0.0  # Centered on axis of symmetry
            
            base_cg = np.array([x, y, z])
            
        elif component_type == 'tail':
            # Tail cone: truncated cone with base radius = hull_radius, tip radius = tail_radius
            # Geometry: Extends from X=nose_length+cyl_length to X=glider_length
            # CG of a truncated cone is approximately 1/4 from the base (3/4 from tip)
            # - X: nose_length + cyl_length + 0.25*tail_length (measured from nose tip)
            # - Y: 0.0 (assumed centered on centerline, can be offset for asymmetry)
            # - Z: 0.0 (assumed centered on centerline, can be offset for asymmetry)
            x = self.nose_length + self.cyl_length + 0.25 * self.tail_length
            
            # For a truncated cone, if there's any asymmetry in Y or Z, it would be here
            # This can be extended for actual asymmetric geometries
            y = 0.0
            z = 0.0
            
            base_cg = np.array([x, y, z])
            
        else:
            raise ValueError(f"Unknown component type: {component_type}")
        
        # Check if custom CG offsets are specified in parameters
        custom_key = f'{component_type}_cg_offset'
        if custom_key in self.params:
            custom_offset = self._vec(self.params[custom_key])
        else:
            custom_offset = np.array([0.0, 0.0, 0.0])
        
        # Apply custom offset if specified
        return base_cg + custom_offset
    
    def _calculate_ballast_cg(self) -> np.ndarray:
        """
        Calculate the 3D CG position of the ballast system.
        
        Returns:
            np.ndarray: 3D CG position [x, y, z] in body frame (meters)
        """
        # Base position from parameters
        base_pos = self.ballast_base_position
        
        # If the ballast tank has any asymmetry or offset from the base position,
        # it would be calculated here based on actual tank geometry
        # For now, using the base position directly
        return base_pos.copy()
    
    def _calculate_mvm_cg(self) -> np.ndarray:
        """
        Calculate the 3D CG position of the moving mass system.
        
        Returns:
            np.ndarray: 3D CG position [x, y, z] in body frame (meters)
        """
        # Base position plus current offset
        return self.mvm_base_position + self.mvm_offset

    #---------------- Getters ----------------
    def get_position_inertial(self) -> np.ndarray:
        """Get current position in inertial frame."""
        return self.position_inertial.copy()

    def get_attitude_euler(self) -> np.ndarray:
        """Get current attitude as Euler angles [roll, pitch, yaw] in radians."""
        r = R.from_quat(self.attitude)
        return r.as_euler('xyz')

    def get_attitude_quaternion(self) -> np.ndarray:
        """Get current attitude as quaternion [x, y, z, w]."""
        return self.attitude.copy()

    def get_velocity_body(self) -> np.ndarray:
        """Get current velocity in body frame."""
        return np.array([self.u, self.v, self.w])

    def get_angular_velocity_body(self) -> np.ndarray:
        """Get current angular velocity in body frame."""
        return np.array([self.p, self.q, self.r])

    def get_velocity_inertial(self) -> np.ndarray:
        """Get current velocity in inertial frame."""
        R_bi = R.from_quat(self.attitude).as_matrix()
        return R_bi @ np.array([self.u, self.v, self.w])
    
    def get_velocity_state(self) -> Dict[str, float]:
        """Get current velocity state as a dictionary."""
        return {
            'u': self.u, 'v': self.v, 'w': self.w,
            'p': self.p, 'q': self.q, 'r': self.r
        }
    
    def get_pump_state(self) -> Dict[str, Any]:
        """Get current pump state and diagnostics."""
        return {
            'pump_on': self.pump_on,
            'pump_direction': self.pump_direction,
            'pump_power': self.pump_power,
            'pump_work': self.pump_work,
            'ballast_fill': self.fill_fraction,
            'max_flow_rate': self.max_ballast_flow
        }
    
    def get_drag_coefficients(self, aoa_deg: float) -> Dict[str, float]:
        """
        Get drag coefficients for a given angle of attack.
        
        Args:
            aoa_deg: Angle of attack in degrees (positive = nose up)
            
        Returns:
            Dictionary with Cd, CL, CM coefficients
        """
        return {
            'Cd': float(self._interp_cd(aoa_deg)),
            'CL': float(self._interp_cl(aoa_deg)),
            'CM': float(self._interp_cm(aoa_deg))
        }
    
    def get_state_vector(self) -> np.ndarray:
        """
        Get the complete state vector for 6-DOF dynamics.
        Returns: [x, y, z, qx, qy, qz, qw, u, v, w, p, q, r, ballast_fill, mvm_offset_x, mvm_offset_y, mvm_offset_z]
        """
        self.state_vector = np.concatenate([
            self.position_inertial,      # [x, y, z] - inertial position
            self.attitude,               # [qx, qy, qz, qw] - quaternion attitude
            [self.u, self.v, self.w],   # [u, v, w] - body frame velocity
            [self.p, self.q, self.r],   # [p, q, r] - body frame angular velocity
            [self.fill_fraction],        # ballast fill level (0..1)
            self.mvm_offset             # [mvm_offset_x, mvm_offset_y, mvm_offset_z]
        ])
        return self.state_vector

    
    def get_mvm_travel_remaining(self, direction: str = 'x') -> float:
        """
        Get remaining travel distance in specified direction.
        
        Args:
            direction: 'x', 'y', or 'z'
            
        Returns:
            Remaining travel distance in meters
        """
        max_offset = self.MVM_length / 2
        
        if direction == 'x':
            return max_offset - abs(self.mvm_offset[0])
        elif direction == 'y':
            return max_offset - abs(self.mvm_offset[1])
        elif direction == 'z':
            return max_offset - abs(self.mvm_offset[2])
        else:
            raise ValueError("direction must be 'x', 'y', or 'z'")
    
    def get_cg_3d(self) -> np.ndarray:
        """
        Get the center of gravity in 3D space.
        
        Returns:
            np.ndarray: 3D CG position [x, y, z] in body frame (meters)
        """
        return self.cg.copy()
    
    def get_cg_components(self) -> Dict[str, np.ndarray]:
        """
        Get the CG positions of individual mass components for analysis.
        
        Returns:
            Dict containing CG positions of all mass components:
            - 'hull': Combined hull CG
            - 'ballast': Ballast tank CG (shell + water)
            - 'fixed': Fixed mass CG
            - 'mvm': Moving mass CG
        """
        return {
            'hull': self._cg_parts['hull_cg'].copy(),
            'ballast': self._cg_parts['ballast_pos'].copy(),
            'fixed': self._cg_parts['fixed_pos'].copy(),
            'mvm': self._cg_parts['Moving_Mass_pos'].copy()
        }
    
    def get_cg_mass_breakdown(self) -> Dict[str, float]:
        """
        Get the mass breakdown of all components for analysis.
        
        Returns:
            Dict containing masses of all components:
            - 'hull': Total hull mass
            - 'ballast_shell': Ballast tank shell mass
            - 'ballast_water': Ballast water mass
            - 'fixed': Fixed mass
            - 'mvm': Moving mass
        """
        return {
            'hull': self._m_parts['m_hull'],
            'ballast_shell': self._m_parts['m_tank_shell'],
            'ballast_water': self._m_parts['m_ballast_water'],
            'fixed': self._m_parts['m_fixed'],
            'mvm': self._m_parts['m_MVM']
        }
    
    def get_cg_inertial(self) -> np.ndarray:
        """
        Get the center of gravity in inertial frame coordinates.
        
        Returns:
            np.ndarray: 3D CG position [x, y, z] in inertial frame (meters)
        """
        # Transform body CG to inertial frame using current position and attitude
        R_bi = R.from_quat(self.attitude).as_matrix()
        cg_inertial = self.position_inertial + R_bi @ self.cg
        return cg_inertial
    
    def get_fixed_mass_position(self) -> np.ndarray:
        """
        Get the current 3D position of the fixed mass component.
        
        Returns:
            np.ndarray: 3D position [x, y, z] in body frame (meters)
        """
        return self._vec(self.params.get('fixed_position', np.array([0.4, 0.0, 0.0])))
    
    def get_cg_offset_from_geometric_center(self) -> np.ndarray:
        """
        Get the CG offset from the geometric center of the glider.
        
        Returns:
            np.ndarray: 3D offset [dx, dy, dz] from geometric center (meters)
        """
        # Geometric center is at the middle of the glider length
        geometric_center = np.array([self.glider_length / 2, 0.0, 0.0])
        cg_offset = self.cg - geometric_center
        return cg_offset
    
    def get_cg_stability_info(self) -> Dict[str, Any]:
        """
        Get comprehensive CG information for stability analysis.
        
        Returns:
            Dict containing:
            - 'cg_body': CG in body frame
            - 'cg_inertial': CG in inertial frame
            - 'cg_offset': CG offset from geometric center
            - 'mass': Total mass
            - 'components': Individual component CGs
            - 'mass_breakdown': Mass breakdown
        """
        return {
            'cg_body': self.get_cg_3d(),
            'cg_inertial': self.get_cg_inertial(),
            'cg_offset': self.get_cg_offset_from_geometric_center(),
            'mass': self.mass,
            'components': self.get_cg_components(),
            'mass_breakdown': self.get_cg_mass_breakdown()
        }
    
    def get_hull_component_cg_offset(self, component_type: str) -> np.ndarray:
        """
        Get the current CG offset for a hull component.
        
        Args:
            component_type: 'nose', 'cyl', or 'tail'
            
        Returns:
            np.ndarray: 3D CG offset [dx, dy, dz] from geometric CG (meters)
        """
        if component_type not in ['nose', 'cyl', 'tail']:
            raise ValueError("component_type must be 'nose', 'cyl', or 'tail'")
        
        custom_key = f'{component_type}_cg_offset'
        if custom_key in self.params:
            return self._vec(self.params[custom_key]).copy()
        else:
            return np.array([0.0, 0.0, 0.0])

    #---------------- Setters ----------------
    def set_velocity_arrays(self, vel_body: np.ndarray, omega_body: np.ndarray):
        """Set velocity state from numpy arrays."""
        if vel_body is not None and len(vel_body) == 3:
            self.u, self.v, self.w = vel_body
        if omega_body is not None and len(omega_body) == 3:
            self.p, self.q, self.r = omega_body

    def set_mvm_offset(self, offset: np.ndarray):
        """Set the MVM offset from its base position with travel limits."""
        offset = np.asarray(offset, dtype=float).reshape(3)
        
        # Enforce travel limits (symmetric around base position)
        max_offset = self.MVM_length / 2
        offset_clipped = np.clip(offset, -max_offset, max_offset)
        
        self.mvm_offset = offset_clipped
        
        # Recalculate mass properties since MVM position changed
        self._calculate_mass_properties()
        self._calculate_inertia()

    def set_ballast_fill(self, fill_fraction: float):
        """Set ballast tank fill level directly."""
        self.fill_fraction = np.clip(fill_fraction, 0.0, 1.0)
        # Recalculate mass properties since ballast changed
        self._calculate_mass_properties()
    
    def set_fixed_mass_position(self, position: np.ndarray):
        """
        Set the 3D position of the fixed mass component.
        
        Args:
            position: 3D position [x, y, z] in body frame (meters)
        """
        # Update the parameter and recalculate mass properties
        self.params['fixed_position'] = self._vec(position)
        self._calculate_mass_properties()
    
    def set_hull_component_cg_offset(self, component_type: str, offset: np.ndarray):
        """
        Set a custom CG offset for a hull component to account for asymmetric geometry.
        
        Args:
            component_type: 'nose', 'cyl', or 'tail'
            offset: 3D offset [dx, dy, dz] from calculated geometric CG (meters)
        """
        if component_type not in ['nose', 'cyl', 'tail']:
            raise ValueError("component_type must be 'nose', 'cyl', or 'tail'")
        
        # Store the offset and recalculate mass properties
        self.params[f'{component_type}_cg_offset'] = self._vec(offset)
        self._calculate_mass_properties()

    def set_state_vector(self, state: np.ndarray):
        """
        Set the complete state vector for 6-DOF dynamics.
        Args:
            state: [x, y, z, qx, qy, qz, qw, u, v, w, p, q, r, ballast_fill, mvm_offset_x, mvm_offset_y, mvm_offset_z]
        """
        if len(state) != 17:
            raise ValueError(f"State vector must have 17 elements, got {len(state)}")
        
        self.position_inertial = state[0:3]
        self.attitude = state[3:7]
        self.u, self.v, self.w = state[7:10]
        self.p, self.q, self.r = state[10:13]
        self.fill_fraction = np.clip(state[13], 0.0, 1.0)  # Ensure ballast fill stays within 0-100%
        self.mvm_offset = state[14:17]
        
        # Enforce MVM travel limits
        max_offset = self.MVM_length / 2
        self.mvm_offset = np.clip(self.mvm_offset, -max_offset, max_offset)
        
        # Recalculate mass properties since ballast or MVM position changed
        self._calculate_mass_properties()
        self._calculate_inertia()