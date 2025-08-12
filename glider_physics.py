import numpy as np
from scipy.spatial.transform import Rotation as R
import warnings

# ============================================================================
# UnderwaterGlider — improved physics model (well-commented)
#
# This file implements a practical 6-DOF model for an underwater glider with
# step-by-step explanatory comments for every physics block. The goal is to
# make the model readable and auditable by engineers who need to understand
# where each term comes from and what approximations were used.
#
# Conventions and coordinate frames:
#  - Body frame: origin at nose tip for geometry/inertia construction; axes:
#      + x: pointing from nose to tail (forward)
#      + y: starboard/right
#      + z: down (so right-handed body frame)
#  - Inertial frame: NED-like (North-East-Down) used for gravity/buoyancy.
#  - Quaternion ordering: SciPy style [x, y, z, w] (vector part first, scalar last).
#  - Velocities and angular rates are expressed in body coordinates.
#
# Major physics sections (annotated):
#  1) Geometry and mass properties (thin-shell approximations)
#  2) Inertia computed from geometry with parallel-axis shifts
#  3) Added mass and added inertia (simple diagonal estimates)
#  4) Hydrodynamics: quadratic per-axis drag and simple lift/moment
#  5) Hydrostatics: buoyancy magnitude and buoyancy moment from CB-CG offset
#  6) Ballast engine model: volumetric/mass flow, plumbing loss, pump efficiency
#  7) Moving-mass coupling: reaction forces from internal piston accelerations
#  8) Rigid-body equations with added-mass approximations
#
# Approximations and limits (explicit):
# - Hull is modelled as thin shell: mass and inertia from perimeter*thickness.
# - Ballast fluid treated either as incompressible (liquid) or compressible
#   (not modelled here in detail). For incompressible liquid we model the
#   fluid as a solid cylindrical segment whose centroid & inertia change with fill
#   level (this is a physically-meaningful approximation for axial filling).
# - Added mass is diagonal here. Off-diagonal terms may be required for
#   asymmetric shapes or for high-fidelity coupling.
# - Hydrodynamic coefficients (Cd, Cl, Cm) are placeholders — replace with
#   towing-tank or CFD-derived tables for accurate results.
# ============================================================================


class UnderwaterGlider:
    """Improved Underwater Glider Physics Model with detailed physics comments.

    The class is intentionally verbose in comments to document every modeling
    decision and the physical meaning of each computed quantity.
    """

    @staticmethod
    def default_params():
        """Return default geometric, material and hydrodynamic parameters.

        These defaults are placeholders and should be replaced with CAD-derived
        geometry, measured masses, and experimental hydrodynamic coefficients
        for accurate predictions.
        """
        return {
            # Geometry (m)
            'nose_length': 0.4,
            'nose_radius': 0.08,
            'cyl_length': 0.9,
            'hull_radius': 0.08,
            'tail_length': 0.3,
            'tail_radius': 0.04,
            'glider_length': 1.6,

            # Ballast tank geometry (assumed cylindrical and aligned with body x)
            'ballast_radius': 0.05,
            'ballast_length': 0.2,

            # Structural properties (thin-shell approximations)
            'hull_thickness': 0.005,
            'hull_density': 2700.0,   # aluminium-like; adjust to your material
            'tank_thickness': 0.004,
            'tank_density': 1200.0,

            # Masses — fixed components that do not change during run
            'fixed_mass': 3.0,        # e.g., batteries, electronics
            'actuator_mass': 0.5,
            'piston_mass': 0.8,

            # Optionally supply an explicit inertia matrix to override geometry calc
            'I_dry_base': None,

            # Hydrodynamic surfaces
            'wing_area': 0.04,

            # Approx hydrodynamic coeffs (replace with experimental values!)
            'Cd_x': 0.8,
            'Cd_y': 1.0,
            'Cd_z': 1.0,
            'CL_alpha': 2.0,  # lift slope per rad
            'CM0': 0.0,
            'CM_alpha': -0.1,

            # Added-mass defaults (diagonal) — must be tuned to shape
            'added_mass_x': 5.0,
            'added_mass_y': 12.0,
            'added_mass_z': 12.0,
            'added_Ixx': 0.05,
            'added_Iyy': 0.12,
            'added_Izz': 0.12,

            # Pump & ballast engine properties
            'pump_efficiency': 0.5,
            'plumbing_loss_coeff': 1e5,  # Pa per (m^3/s) crude linear approx
            'ballast_mode': 'incompressible',

            # Control limits
            'max_ballast_flow': 1e-3,
            'max_piston_speed': 0.02,

            # Piston actuator servo settings (for option A servo-model)
            'piston_travel': 0.2,        # default max travel (m)
            'piston_servo_tau': 0.1,     # servo time-constant (s)
            'max_piston_accel': 0.5,     # m/s^2

            # Misc
            'rho_water': 1025.0,
            'g': 9.81,
        }

    def __init__(self, params=None):
        # Merge user params with defaults
        if params is None:
            params = {}
        self.params = self.default_params()
        self.params.update(params)

        # Store frequently-used constants
        self.rho_water = float(self.params.get('rho_water', 1025.0))
        self.g = float(self.params.get('g', 9.81))

        # Actuator command variables — these are set by control code before calling dynamics
        # dm_ballast_dt (kg/s) is an explicit mass-flow command. dm_vol_dt (m^3/s) is a
        # volumetric command convenient for incompressible flows. dx_p_dt (m/s) is piston
        # speed command for the servo. ddx_p (m/s^2) will be computed by the servo and
        # stored for reaction force computation.
        self.dm_ballast_dt = 0.0
        self.dm_vol_dt = 0.0
        self.dx_p_dt = 0.0
        self.ddx_p = 0.0

        # piston_set_pos is captured on the first dynamics() call to implement
        # travel limits relative to the system's set position, as you requested.
        # If you want to override it explicitly, set params['piston_set_position'].
        self.piston_set_pos = None

        # Diagnostics for pump energy accounting
        self._last_t = None
        self.pump_power = 0.0
        self.pump_work = 0.0

        # Build geometry, mass and inertia properties. These calls compute
        # self.m_dry, self.cg_dry (dry center of gravity), self.I_dry_base (inertia
        # about dry CG), volumes, and added mass defaults.
        self._calculate_mass_properties()
        self._calculate_volume_properties()
        self._build_added_mass()

    # ------------------------------ geometry / mass ------------------------------
    def _calculate_mass_properties(self):
        """Compute approximate shell masses and centers for hull and tank.

        Steps/assumptions:
        - Hull split into three segments: nose cone, cylinder, tail cone.
        - Each segment treated as thin shell with lateral area = perimeter * length
          (for cones we use lateral area formula approximation with slant length).
        - Mass of slice = area * thickness * material density.
        - This yields segment masses which are then used to compute the hull CG.
        - Fixed components and tank shell are added as point/analytic masses.
        """
        p = self.params

        # Lateral areas (approximate for conical nose/tail)
        nose_area = np.pi * p['nose_radius'] * np.sqrt(p['nose_radius']**2 + p['nose_length']**2)
        cyl_area = 2.0 * np.pi * p['hull_radius'] * p['cyl_length']
        tail_area = np.pi * p['tail_radius'] * np.sqrt(p['tail_radius']**2 + p['tail_length']**2)

        # Segment masses (thin-shell)
        m_nose = nose_area * p['hull_thickness'] * p['hull_density']
        m_cyl = cyl_area * p['hull_thickness'] * p['hull_density']
        m_tail = tail_area * p['hull_thickness'] * p['hull_density']

        # Total hull shell mass
        self.m_hull = m_nose + m_cyl + m_tail

        # Tank shell mass (thin cylindrical shell analytic)
        tank_area = 2.0 * np.pi * p['ballast_radius'] * p['ballast_length']
        self.m_tank = tank_area * p['tank_thickness'] * p['tank_density']

        # Dry mass: hull + tank shell + fixed items + actuators + piston
        self.m_dry = self.m_hull + self.m_tank + p['fixed_mass'] + p['actuator_mass'] + p['piston_mass']

        # Store segment masses for CG/inertia calculations
        self._m_nose = m_nose
        self._m_cyl = m_cyl
        self._m_tail = m_tail

        # Compute segment centroid locations along body x (from nose tip)
        # Cone centroid distance from apex = 3/4 * h (solid cone formula used here)
        nose_cg_x = 0.75 * p['nose_length']
        cyl_cg_x = p['nose_length'] + 0.5 * p['cyl_length']
        tail_cg_x = p['nose_length'] + p['cyl_length'] + 0.25 * p['tail_length']

        nose_cg = np.array([nose_cg_x, 0.0, 0.0])
        cyl_cg = np.array([cyl_cg_x, 0.0, 0.0])
        tail_cg = np.array([tail_cg_x, 0.0, 0.0])

        # Weighted average for hull CG
        self.hull_cg = (nose_cg * self._m_nose + cyl_cg * self._m_cyl + tail_cg * self._m_tail) / (
            self._m_nose + self._m_cyl + self._m_tail
        )

        # Utility to accept scalar or vector positions in params (backwards compatible)
        def vec(v, fallback_x=0.5 * p['glider_length']):
            a = np.asarray(v, dtype=float)
            if a.size == 1:
                return np.array([a.item(), 0.0, 0.0])
            return a

        # Positions of components in body coordinates (defaults if not provided)
        self.ballast_pos = vec(self.params.get('ballast_position', np.array([0.6, 0.0, 0.0])))
        self.fixed_pos = vec(self.params.get('fixed_position', np.array([0.4, 0.0, 0.0])))
        self.actuator_pos = vec(self.params.get('actuator_position', np.array([0.5, 0.0, 0.0])))
        self.piston_nominal_pos = vec(self.params.get('piston_position', np.array([0.55, 0.0, 0.0])))

        # Compute dry CG including piston mass (this is the reference CG for I_dry_base)
        numerator = (self.hull_cg * self.m_hull + self.ballast_pos * self.m_tank +
                     self.fixed_pos * p['fixed_mass'] + self.actuator_pos * p['actuator_mass'] +
                     self.piston_nominal_pos * p['piston_mass'])
        if self.m_dry <= 0:
            raise ValueError('Invalid dry mass')
        self.cg_dry = numerator / self.m_dry

        # Compute inertia tensor from geometry (thin-shell integration + analytic tank + point masses)
        I_geom = self._compute_inertia_from_geometry()

        # If user provided an explicit I_dry_base use it; otherwise use computed geometry inertia
        I_user = self.params.get('I_dry_base', None)
        if I_user is not None:
            self.I_dry_base = np.asarray(I_user, dtype=float)
        else:
            self.I_dry_base = I_geom

    def _compute_inertia_from_geometry(self, n_slices=400):
        """Numerically compute the dry inertia tensor using thin-shell slices.

        Detailed steps and reasoning:
        - We discretize the hull length into many thin slices (rings).
        - For each ring at x: compute local radius r(x) based on whether it is
          in the nose cone, cylinder, or tail cone.
        - Treat each ring as a thin ring (mass dm = perimeter * thickness * rho * dx).
        - The ring's moment of inertia about the body x-axis is dm * r^2.
        - The ring's moment about transverse axes through the ring center is 0.5*dm*r^2.
        - Shift transverse contributions of each slice to the nose-tip origin using dm*x^2
          (parallel axis for translation along x).
        - Sum contributions for hull slices, add analytic tank shell inertia and point masses.
        - Finally shift the combined inertia from origin to the dry CG using parallel-axis.

        Approximation notes:
        - This treats the shell as thin and axesymmetric; any major asymmetry (heavy
        batteries off-center) must be added as point masses at their true positions.
        - Use sufficiently many slices (default 400) to get good accuracy for cones.
        """
        p = self.params
        x_nose_end = p['nose_length']
        x_cyl_end = x_nose_end + p['cyl_length']
        x_tail_end = x_cyl_end + p['tail_length']
        L = x_tail_end

        xs = np.linspace(0.0, L, n_slices)
        dx = xs[1] - xs[0]

        thickness = p['hull_thickness']
        rho = p['hull_density']

        # accumulate inertia components about origin (nose tip)
        I_x_origin = 0.0
        I_y_origin = 0.0
        I_z_origin = 0.0
        m_sum = 0.0

        for x in xs:
            # Piecewise radius profile r(x)
            if x <= x_nose_end:
                # nose cone: linear increase in radius from 0 to nose_radius
                r = (x / max(x_nose_end, 1e-12)) * p['nose_radius']
            elif x <= x_cyl_end:
                r = p['hull_radius']
            else:
                # tail cone: linear taper from hull_radius to tail_radius
                xi = (x - x_cyl_end) / max(p['tail_length'], 1e-12)
                r = (1.0 - xi) * p['hull_radius'] + xi * p['tail_radius']

            r = max(r, 1e-8)
            # mass of thin slice (circumference * thickness * density * dx)
            dm = (2.0 * np.pi * r) * thickness * rho * dx
            m_sum += dm

            # ring inertia about its own center
            I_ring_x = dm * r**2
            I_ring_y_center = 0.5 * dm * r**2
            I_ring_z_center = I_ring_y_center

            # shift transverse ring inertias to nose-tip origin using x^2 term
            I_x_origin += I_ring_x
            I_y_origin += I_ring_y_center + dm * (x**2)
            I_z_origin += I_ring_z_center + dm * (x**2)

        # scale inertia to match analytic hull mass (handles discretization error)
        if m_sum > 0:
            scale = self.m_hull / m_sum
            I_x_origin *= scale
            I_y_origin *= scale
            I_z_origin *= scale
            m_sum = self.m_hull

        # Tank shell analytic thin-cylinder contributions about its center
        m_tank = self.m_tank
        r_t = p['ballast_radius']
        L_t = p['ballast_length']
        x_tank = self.ballast_pos[0]
        I_tank_xc = m_tank * (r_t**2)
        I_tank_yc = m_tank * ((1.0 / 12.0) * (L_t**2) + 0.5 * r_t**2)
        I_tank_zc = I_tank_yc

        # shift tank shell contributions to nose-tip origin (parallel axis) - add to origin inertias
        I_x_origin += I_tank_xc
        I_y_origin += I_tank_yc + m_tank * (x_tank**2)
        I_z_origin += I_tank_zc + m_tank * (x_tank**2)
        m_sum += m_tank

        # Add point-mass contributions for fixed components at their provided positions
        for mass_name, pos in [('fixed_mass', self.fixed_pos),
                               ('actuator_mass', self.actuator_pos),
                               ('piston_mass', self.piston_nominal_pos)]:
            m_pt = p.get(mass_name, 0.0)
            pos = np.asarray(pos, dtype=float)
            x_pt, y_pt, z_pt = pos
            I_x_origin += m_pt * (y_pt**2 + z_pt**2)
            I_y_origin += m_pt * (z_pt**2 + x_pt**2)
            I_z_origin += m_pt * (x_pt**2 + y_pt**2)
            m_sum += m_pt

        # Form inertia tensor about origin
        I_origin = np.diag([I_x_origin, I_y_origin, I_z_origin])

        # Shift inertia from origin (nose tip) to dry CG using parallel-axis theorem
        d_vec = np.asarray(self.cg_dry, dtype=float)
        m_total = m_sum
        d_norm2 = np.dot(d_vec, d_vec)
        I_shift = m_total * (d_norm2 * np.eye(3) - np.outer(d_vec, d_vec))
        I_cm = I_origin - I_shift

        # Symmetrize for numerical cleanliness
        I_cm = 0.5 * (I_cm + I_cm.T)
        I_cm[np.abs(I_cm) < 1e-12] = 0.0
        return I_cm

    def _calculate_volume_properties(self):
        """Compute hull volumes and center of buoyancy for full-submergence.

        Note: For partial submergence the displaced volume and CB location depend
        on the immersion plane and would need a separate hydrostatic routine.
        """
        p = self.params
        self.V_nose = (1.0 / 3.0) * np.pi * p['nose_radius']**2 * p['nose_length']
        self.V_cyl = np.pi * p['hull_radius']**2 * p['cyl_length']
        self.V_tail = (1.0 / 3.0) * np.pi * p['tail_radius']**2 * p['tail_length']
        self.V_hull = self.V_nose + self.V_cyl + self.V_tail

        # Ballast volume capabilities
        self.V_ballast_max = np.pi * p['ballast_radius']**2 * p['ballast_length']
        self.V_ballast_neutral = 0.5 * self.V_ballast_max

        # CB for a fully-submerged symmetric hull - computed as volume-weighted centroid
        nose_cb_x = 0.75 * p['nose_length']
        cyl_cb_x = p['nose_length'] + 0.5 * p['cyl_length']
        tail_cb_x = p['nose_length'] + p['cyl_length'] + 0.25 * p['tail_length']
        self.cb = np.array([
            (self.V_nose * nose_cb_x + self.V_cyl * cyl_cb_x + self.V_tail * tail_cb_x) / max(self.V_hull, 1e-9),
            0.0,
            0.0
        ])

    # --------------------------- added-mass / hydrodynamic ------------------------
    def _build_added_mass(self):
        """Simple diagonal added mass estimates.

        Added mass represents the inertia of the fluid that must be accelerated
        with the vehicle. For slender bodies the added mass in the transverse
        directions is larger than in surge. Off-diagonals are neglected here.
        For higher fidelity, fill a full 6x6 added-mass matrix from CFD or
        empirical data.
        """
        p = self.params
        self.MA_linear = np.diag([p['added_mass_x'], p['added_mass_y'], p['added_mass_z']])
        self.MA_rot = np.diag([p['added_Ixx'], p['added_Iyy'], p['added_Izz']])

    # --------------------------- helper utilities -------------------------------
    @staticmethod
    def _vec(v):
        a = np.asarray(v, dtype=float)
        if a.size == 1:
            return np.array([a.item(), 0.0, 0.0])
        return a

    def reset(self):
        """Reset transient diagnostics and default state vector."""
        self._last_t = None
        self.pump_power = 0.0
        self.pump_work = 0.0
        self.dm_ballast_dt = 0.0
        self.dm_vol_dt = 0.0
        self.dx_p_dt = 0.0
        self.ddx_p = 0.0
        # state layout changed: add piston velocity state at index 15
        self.state0 = np.zeros(17)
        self.state0[13] = self.rho_water * self.V_ballast_neutral
        self.state0[14] = 0.0  # piston position
        self.state0[15] = 0.0  # piston velocity
        # reset piston_set_pos so it's captured on first dynamics call
        self.piston_set_pos = self.params.get('piston_set_position', None)

    # --------------------------- primary dynamics --------------------------------
    def dynamics(self, t, state):
        """Compute state derivative using a quasi-6DOF model with added-mass.

        Steps (annotated):
        1. Unpack state and normalize quaternion to avoid drift.
        2. Clip physical states (ballast mass and piston travel) to valid ranges.
        3. Compute total mass and effective masses with diagonal added-mass.
        4. Compute CG and inertia that account for current ballast fill and piston.
        5. Compute hydrodynamic forces and buoyancy/gravity in body frame.
        6. Compute moving-mass reaction forces from piston acceleration.
        7. Sum forces/moments and compute accelerations. For linear accel we divide
           by axis-wise effective masses (approx). For angular accel solve I*wdot = M - w x Iw.
        8. Kinematics: integrate body velocities to inertial position using rotation.
        9. Return derivative vector.

        Important numerical notes:
        - If you replace diagonal added-mass with a full 6x6 matrix you'll need
          to solve a 6x6 linear system coupling linear and angular accelerations.
        - Quaternion derivative uses body rates and SciPy ordering.
        """
        p = self.params
        pos = np.asarray(state[0:3], dtype=float)
        quat = np.asarray(state[3:7], dtype=float)
        vel = np.asarray(state[7:10], dtype=float)
        omega = np.asarray(state[10:13], dtype=float)
        m_ballast = float(state[13])
        x_p = float(state[14])
        piston_vel = float(state[15])

        # 1) Quaternion normalization
        qnorm = np.linalg.norm(quat)
        if qnorm > 0:
            quat = quat / qnorm

        # 2) State clipping for safety (ballast mass only here)
        m_ballast = np.clip(m_ballast, 0.0, self.rho_water * self.V_ballast_max)

        # Capture piston_set_pos on first call if not explicitly provided.
        if self.piston_set_pos is None:
            # the "system set position" is the piston position at simulation start
            self.piston_set_pos = float(x_p)

        # compute allowable piston range relative to the captured set position
        min_pos = float(self.piston_set_pos)
        max_pos = float(self.piston_set_pos + p.get('piston_travel', 0.2))

        # enforce range (clip absolute position)
        x_p = np.clip(x_p, min_pos, max_pos)

        # 3) Effective mass including added-mass (diagonal approximation)
        m_total = self.m_dry + m_ballast
        m_eff_x = m_total + self.MA_linear[0, 0]
        m_eff_y = m_total + self.MA_linear[1, 1]
        m_eff_z = m_total + self.MA_linear[2, 2]

        # rotational inertia baseline + added rotational inertia
        I_rb = self.I_dry_base.copy()
        I_eff = I_rb + self.MA_rot

        # 4) Compute current CG and inertia (ballast fluid distribution included)
        cg = self.calculate_cg(m_ballast, x_p)         # body-frame vector
        I_total = self.calculate_inertia(m_ballast, x_p, cg)
        I_total_eff = I_total + self.MA_rot

        # 5) Rotation matrix body->inertial
        rot = R.from_quat(quat)
        R_ib = rot.as_matrix()

        # Hydrodynamic forces and moments in body frame
        F_hydro, M_hydro = self.hydrodynamic_forces(vel, omega)

        # Buoyancy and gravity: compute in inertial and rotate to body frame
        F_grav_body, F_buoy_body, M_buoy = self.buoyancy_gravity_forces(R_ib, m_total, m_ballast, cg)

        # ---------------- piston servo modelling (Option A) ---------------------
        # We implement a simple first-order servo driven by the commanded piston
        # speed (self.dx_p_dt). This produces a physically consistent piston
        # velocity and acceleration (which are used to compute reaction forces).
        dm_dt, dx_cmd = self.apply_actuator_limits(m_ballast, x_p)
        # servo time-constant and acceleration limits
        tau = max(1e-6, p.get('piston_servo_tau', 0.1))
        max_acc = p.get('max_piston_accel', 0.5)

        # compute piston acceleration as (command_vel - current_vel)/tau
        ddx = (float(dx_cmd) - piston_vel) / tau
        # saturate acceleration
        ddx = np.clip(ddx, -abs(max_acc), abs(max_acc))

        # Safety: if we're at limit and velocity would push us out-of-bounds, zero the accel/vel
        eps = 1e-9
        if x_p <= min_pos + eps and (piston_vel < 0 or dx_cmd < 0):
            ddx = 0.0
            piston_vel = max(piston_vel, 0.0)
        if x_p >= max_pos - eps and (piston_vel > 0 or dx_cmd > 0):
            ddx = 0.0
            piston_vel = min(piston_vel, 0.0)

        # store computed piston acceleration for reaction force calculation
        self.ddx_p = float(ddx)

        # 6) Moving-mass reaction forces & moments (now uses internally computed ddx)
        F_mm, M_mm = self.moving_mass_reaction(m_ballast, x_p, cg)

        # 7) Sum forces/moments
        F_total = F_grav_body + F_buoy_body + F_hydro + F_mm
        M_total = M_buoy + M_hydro + M_mm

        # Linear acceleration (approximate per-axis effective mass)
        # Note: this is an approximation. Exact treatment uses a full 6x6 mass
        # matrix and solves for linear + angular accelerations simultaneously.
        dvel_dt = np.zeros(3)
        dvel_dt[0] = F_total[0] / m_eff_x - (omega[1] * vel[2] - omega[2] * vel[1])
        dvel_dt[1] = F_total[1] / m_eff_y - (omega[2] * vel[0] - omega[0] * vel[2])
        dvel_dt[2] = F_total[2] / m_eff_z - (omega[0] * vel[1] - omega[1] * vel[0])

        # Angular acceleration: use full inertia tensor including added rotational inertia
        try:
            domega_dt = np.linalg.solve(I_total_eff, (M_total - np.cross(omega, I_total_eff @ omega)))
        except np.linalg.LinAlgError:
            warnings.warn('Singular inertia matrix; returning zeros')
            return np.zeros_like(state)

        # 8) Kinematics: inertial position derivative and quaternion derivative
        dpos_dt = R_ib @ vel
        dquat_dt = self.quaternion_kinematics(quat, omega)

        # 9) Actuator derivatives and pump power accounting
        # dm_dt computed above from apply_actuator_limits
        # piston position derivative = piston_vel; piston velocity derivative = ddx
        # We must ensure we do not integrate the piston beyond min/max: if we are at
        # the bounds and acceleration is zero, velocity will be clamped here for safety.
        if x_p <= min_pos + eps and piston_vel < 0:
            piston_vel = 0.0
        if x_p >= max_pos - eps and piston_vel > 0:
            piston_vel = 0.0

        # compute pump power/work
        self._compute_pump_power_work(pos, m_ballast, dm_dt, t)

        # Form derivative vector — state has length 17 now
        deriv = np.zeros_like(state, dtype=float)
        deriv[0:3] = dpos_dt
        deriv[3:7] = dquat_dt
        deriv[7:10] = dvel_dt
        deriv[10:13] = domega_dt
        deriv[13] = dm_dt
        deriv[14] = piston_vel
        deriv[15] = ddx

        # keep last state for diagnostics
        self._last_state = state.copy()
        return deriv

    # ------------------------ hydrodynamics implementations ----------------------
    def hydrodynamic_forces(self, vel, omega):
        """Compute hydrodynamic forces and moments in body coordinates.

        Physics explained step-by-step:
        - Viscous/pressure drag on bodies scales ~ 0.5 * rho * U^2 * Cd * A.
          We implement axis-wise quadratic drag using reference areas for frontal
          vs cross-sectional directions. This is a coarse but robust approach.
        - Lift in the x-z plane is approximated using CL = CL_alpha * alpha where
          alpha is the small-angle attack approximated by atan2(-w, u).
        - Pitching moment is modelled as a simple CL/CM-derived term located at
          an effective moment arm (here we use glider_length as reference).
        - Small linear damping terms are added to help numerical stability near zero speed.

        Important caveats:
        - This is not a full panel / potential-flow + viscous model. For
          precision use CFD or experimental CL/CD tables as functions of alpha
          and Reynolds number.
        """
        p = self.params
        U = np.linalg.norm(vel)
        if U < 1e-6:
            # At near-zero speed, use light linear damping to avoid singularities
            F_damp = -0.1 * vel
            M_damp = -0.05 * omega
            return F_damp, M_damp

        # Reference areas
        A_x = max(0.02, 0.5 * self.params.get('hull_radius', 0.08) * self.params.get('glider_length', 1.0))
        A_yz = max(self.params.get('wing_area', 0.04), 0.01)

        q = 0.5 * self.rho_water * U**2
        Cd_x = p.get('Cd_x', 0.8)
        Cd_y = p.get('Cd_y', 1.0)
        Cd_z = p.get('Cd_z', 1.0)

        # Quadratic drag on each axis: sign-preserving formula
        Fdx = -q * Cd_x * A_x * (vel[0] / (abs(vel[0]) + 1e-9)) * abs(vel[0])
        Fdy = -q * Cd_y * A_yz * (vel[1] / (abs(vel[1]) + 1e-9)) * abs(vel[1])
        Fdz = -q * Cd_z * A_yz * (vel[2] / (abs(vel[2]) + 1e-9)) * abs(vel[2])
        F_drag = np.array([Fdx, Fdy, Fdz])

        # Approx lift model in x-z plane
        u = vel[0]
        w = vel[2]
        alpha = np.arctan2(-w, u)  # note sign: vel[2] positive down
        CL = self.params.get('CL_alpha', 2.0) * alpha
        lift = q * self.params.get('wing_area', 0.04) * CL
        # lift produces upward force (negative body z)
        F_lift = np.array([0.0, 0.0, -lift])

        # Simple pitching moment proportional to dynamic pressure and CL/CM
        M_pitch = np.array([0.0, q * self.params.get('wing_area', 0.04) * self.params.get('glider_length', 1.0) *
                            (self.params.get('CM0', 0.0) + self.params.get('CM_alpha', -0.1) * alpha), 0.0])

        # small linear/angular damping
        F_damp = -0.05 * vel
        M_damp = -0.02 * omega

        F_total = F_drag + F_lift + F_damp
        M_total = M_pitch + M_damp
        return F_total, M_total

    # ----------------------- buoyancy & gravity ----------------------------------
    def buoyancy_gravity_forces(self, R_ib, m_total, m_ballast, cg):
        """Compute gravity and buoyancy as forces in the body frame.

        Steps and reasoning:
        - Gravity acts downward in inertial NED coordinates: Fg = [0,0,m_total*g]
        - Buoyancy magnitude (fully submerged) = rho_water * displaced_volume * g
          and acts upward in inertial coordinates: Fb = [0,0,-rho*V*g]
        - Convert both vectors to body frame using R_ib.T because R_ib maps
          body->inertial; its transpose maps inertial->body.
        - Buoyancy moment computed as cross(r_cb, Fb_body) where r_cb is vector
          from CG to CB expressed in body coordinates. This generates the restoring
          torque that tends to align CB below CG (statical stability).

        Important: CB is assumed fixed in body coordinates for a fully submerged
        hull. For partial-submergence CB location depends on immersion and must be
        recomputed with hydrostatic integration.
        """
        # Gravity in inertial (NED): +z down
        Fg_inertial = np.array([0.0, 0.0, m_total * self.g])
        Fg_body = R_ib.T @ Fg_inertial

        # Buoyancy magnitude and inertial vector
        Fb_mag = self.rho_water * self.V_hull * self.g
        Fb_inertial = np.array([0.0, 0.0, -Fb_mag])
        Fb_body = R_ib.T @ Fb_inertial

        # Buoyancy torque about CG (CB defined in body coords for full-submersion)
        r_cb = self.cb - cg
        M_buoy = np.cross(r_cb, Fb_body)
        return Fg_body, Fb_body, M_buoy

    # ----------------------- moving mass coupling -------------------------------
    def moving_mass_reaction(self, m_ballast, x_p, cg):
        """Compute reaction force/moment produced by accelerating internal piston.

        Rationale and physics:
        - The piston is a discrete mass inside the body. When it accelerates
          relative to the body, Newton’s 3rd law produces an equal and opposite
          reaction force on the vehicle hull.
        - We treat the piston as located at piston_nominal_pos + x_p along x.
        - The piston acceleration used here is the servo-computed self.ddx_p.
        - Moment is computed about the CG using cross(r_piston-cg, F_reaction).

        Notes on fidelity:
        - This is a quasi-static coupling: if you want motor/gearbox dynamics
          simulate the piston mass with its own second-order ODE and return
          commanded acceleration instead of directly setting ddx_p.
        """
        m_piston = self.params.get('piston_mass', 0.0)
        piston_pos = self.piston_nominal_pos + np.array([x_p, 0.0, 0.0])
        a_piston_body = np.array([self.ddx_p, 0.0, 0.0])
        F_reaction = - m_piston * a_piston_body
        M_reaction = np.cross(piston_pos - cg, F_reaction)
        return F_reaction, M_reaction

    # ------------------------- inertia & cg calculations ------------------------
    def calculate_cg(self, m_ballast, x_p):
        """Compute the total center of gravity taking ballast fill into account.

        Steps:
        1) Determine ballast fluid volume Vf = m_ballast / rho and the axial
           filled length Lf = Vf / (pi*r^2) (clamped to tank length).
        2) Compute centroid of filled segment: if fluid occupies the first Lf
           from the tank start, centroid is at x_start + 0.5*Lf.
        3) Compute mass-weighted sum for hull shell, tank shell, fixed items,
           piston (nominal), and the fluid mass at the computed centroid.
        4) Divide by total mass to obtain CG.

        This method assumes axial contiguous filling (no baffling or slosh)
        """
        p = self.params
        r = p['ballast_radius']
        L_t = p['ballast_length']
        V_max = np.pi * r**2 * L_t
        m_ballast = float(np.clip(m_ballast, 0.0, self.rho_water * V_max))
        Vf = m_ballast / self.rho_water if self.rho_water > 0 else 0.0
        Lf = np.clip(Vf / (np.pi * r**2), 0.0, L_t)

        x_tank_center = self.ballast_pos[0]
        x_start = x_tank_center - 0.5 * L_t
        if Lf <= 0:
            ballast_centroid_x = x_tank_center
        else:
            ballast_centroid_x = x_start + 0.5 * Lf

        ballast_centroid = np.array([ballast_centroid_x, self.ballast_pos[1], self.ballast_pos[2]])

        piston_pos = self.piston_nominal_pos + np.array([x_p, 0.0, 0.0])
        m_piston = p.get('piston_mass', 0.0)

        m_hull = self.m_hull
        m_tank_shell = self.m_tank
        m_fixed = p.get('fixed_mass', 0.0)
        m_act = p.get('actuator_mass', 0.0)

        sum_m_pos = (self.hull_cg * m_hull +
                     self.ballast_pos * m_tank_shell +
                     self.fixed_pos * m_fixed +
                     self.actuator_pos * m_act +
                     self.piston_nominal_pos * m_piston)

        # add fluid mass located at its centroid
        sum_m_pos = sum_m_pos + ballast_centroid * m_ballast + piston_pos * 0.0  # piston nominal included above

        total_mass = m_hull + m_tank_shell + m_fixed + m_act + m_piston + m_ballast
        if total_mass <= 0:
            raise ValueError('Non-positive total mass when computing CG')
        cg_total = sum_m_pos / total_mass
        return cg_total

    def calculate_inertia(self, m_ballast, x_p, cg_total):
        """Compute inertia about CG including ballast fluid distribution.

        Steps:
        1) Start from structural dry inertia I_dry_base (about dry CG).
        2) Compute fluid segment inertia about its own centroid (analytical formula).
           For a solid cylinder segment approximated as a solid cylinder of length Lf:
             I_x = 0.5*m*r^2 (about cylinder axis)
             I_perp = (1/12) * m * (3*r^2 + Lf^2) (transverse axes)
        3) Shift fluid centroid inertia to vehicle CG via parallel-axis theorem.
        4) Add piston point-mass inertia similarly.
        5) Return the combined inertia tensor used in angular dynamics.

        Notes:
        - We assume that I_dry_base already contains hull+tank shell+other structural items.
        - If I_dry_base was computed about dry CG then adding shifted point masses keeps
        consistency. If the user supplies I_dry_base about a different point they must
        provide it about the dry CG or the code must be adapted.
        """
        p = self.params
        I = self.I_dry_base.copy()

        r = p['ballast_radius']
        L_t = p['ballast_length']
        V_max = np.pi * r**2 * L_t
        m_ballast = float(np.clip(m_ballast, 0.0, self.rho_water * V_max))
        Vf = m_ballast / self.rho_water if self.rho_water > 0 else 0.0
        Lf = np.clip(Vf / (np.pi * r**2), 0.0, L_t)

        if m_ballast > 0 and Lf > 0:
            x_tank_center = self.ballast_pos[0]
            x_start = x_tank_center - 0.5 * L_t
            ballast_centroid_x = x_start + 0.5 * Lf
            ballast_centroid = np.array([ballast_centroid_x, self.ballast_pos[1], self.ballast_pos[2]])

            # fluid inertia about its own centroid
            I_xc = 0.5 * m_ballast * r**2
            I_perp = (1.0 / 12.0) * m_ballast * (3.0 * r**2 + Lf**2)
            I_ballast_centroid = np.diag([I_xc, I_perp, I_perp])

            # Shift to vehicle CG using parallel-axis theorem
            r_vec = ballast_centroid - cg_total
            r_norm2 = np.dot(r_vec, r_vec)
            I_shift = m_ballast * (r_norm2 * np.eye(3) - np.outer(r_vec, r_vec))
            I_ballast_about_cg = I_ballast_centroid + I_shift
        else:
            I_ballast_about_cg = np.zeros((3, 3))

        # piston as point mass (parallel axis)
        piston_pos = self.piston_nominal_pos + np.array([x_p, 0.0, 0.0])
        r_p = piston_pos - cg_total
        m_p = p.get('piston_mass', 0.0)
        I_p = m_p * (np.dot(r_p, r_p) * np.eye(3) - np.outer(r_p, r_p))

        # We assume tank shell already included in I_dry_base so we only add fluid & piston
        return I + I_ballast_about_cg + I_p

    # ------------------------- actuator / pump modelling ------------------------
    def apply_actuator_limits(self, m_ballast, x_p):
        """Clip and translate actuator commands to physical dm/dx rates.

        Reasoning:
        - For incompressible liquid, dm_vol_dt is convenient: dm_dt = vol_cmd * rho
        - We allow users to either set a mass-flow command (dm_ballast_dt kg/s) or
        a volumetric command (dm_vol_dt m^3/s). The code chooses volumetric
        conversion unless an explicit mass-flow is provided.
        - Limits are enforced so the simulation remains physically plausible.
        """
        p = self.params
        # We return a commanded piston speed dx_cmd (m/s) which the servo will try to follow.
        dx_cmd = float(self.dx_p_dt)
        max_piston = p.get('max_piston_speed', 0.02)
        dx_cmd = np.clip(dx_cmd, -max_piston, max_piston)

        # volumetric/mass flow handling for ballast pump
        max_vol = p.get('max_ballast_flow', 1e-3)
        vol_cmd = float(self.dm_vol_dt)
        vol_cmd = np.clip(vol_cmd, -max_vol, max_vol)
        dm_dt = vol_cmd * self.rho_water
        if abs(self.dm_ballast_dt) > 1e-12:
            dm_dt = float(self.dm_ballast_dt)
        if m_ballast <= 0.0 and dm_dt < 0.0:
            dm_dt = 0.0
        if m_ballast >= self.rho_water * self.V_ballast_max and dm_dt > 0.0:
            dm_dt = 0.0
        return dm_dt, dx_cmd

    # ------------------------- quaternion kinematics ---------------------------
    def quaternion_kinematics(self, quat, omega):
        """Compute quaternion time derivative from body angular rates.

        - Quaternion ordering follows SciPy: [x,y,z,w].
        - Kinematic equation: q_dot = 0.5 * Omega(omega) * q
        - Using component form ensures we match SciPy's ordering and avoid sign errors.
        """
        qx, qy, qz, qw = quat
        p_, q_, r_ = omega
        dqx = 0.5 * ( qw * p_ + qz * r_ - qy * q_ )
        dqy = 0.5 * ( qw * q_ - qz * p_ + qx * r_ )
        dqz = 0.5 * ( qw * r_ + qy * p_ - qx * q_ )
        dqw = 0.5 * ( - qx * p_ - qy * q_ - qz * r_ )
        return np.array([dqx, dqy, dqz, dqw])

    # ------------------------- simple diagnostics / trim solver -----------------
    def trim_to_buoyancy(self, desired_buoyancy=0.0):
        """Find ballast mass that produces a desired net buoyant force (N).

        - For fully-submerged hull: Fb - m_total*g = desired_buoyancy
        - Solve for m_total then subtract dry mass to obtain ballast mass.
        - Clamp to available ballast volume.
        """
        Fb = self.rho_water * self.V_hull * self.g
        m_total_required = (Fb - desired_buoyancy) / self.g
        m_ballast_required = m_total_required - self.m_dry
        m_ballast_required = np.clip(m_ballast_required, 0.0, self.rho_water * self.V_ballast_max)
        return m_ballast_required


# ================================ unit test harness ==============================
if __name__ == '__main__':
    # Quick sanity checks: run dynamics with nominal defaults and print results.
    glider = UnderwaterGlider()
    glider.reset()

    state = np.zeros(17)
    state[2] = 2.0
    quat = R.from_euler('y', 0.0, degrees=True).as_quat()
    state[3:7] = quat
    state[7:10] = np.array([0.1, 0.0, 0.0])
    state[10:13] = np.zeros(3)
    state[13] = glider.rho_water * glider.V_ballast_neutral
    state[14] = 0.0
    state[15] = 0.0

    glider.dm_ballast_dt = 0.0
    glider.dx_p_dt = 0.0
    glider.ddx_p = 0.0

    deriv = glider.dynamics(0.0, state)
    print('derivative sample (pos_dot, vel_dot, quat_dot, pump_power, pump_work):')
    print('pos_dot =', deriv[0:3])
    print('vel_dot =', deriv[7:10])
    print('quat_dot =', deriv[3:7])
    print('pump_power =', glider.pump_power, 'W')
    print('pump_work =', glider.pump_work, 'J')

    mb = glider.trim_to_buoyancy(0.0)
    print(f'Required ballast mass for neutral buoyancy: {mb:.3f} kg')

    glider.dm_vol_dt = 5e-4
    deriv2 = glider.dynamics(1.0, state)
    print('after pump command: pump_power =', glider.pump_power, 'W')

    print('Done unit test.')
