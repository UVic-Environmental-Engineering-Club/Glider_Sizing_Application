import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                            QTabWidget, QGroupBox, QSlider, QComboBox, QSpinBox, QDoubleSpinBox)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from glider_simulation import run_simulation
import json
import os
import matplotlib.patches as patches
from scipy.spatial.transform import Rotation as R
from unit_converter import UnitConverterWidget

class GliderGUI(QMainWindow):
    PARAMS_FILE = "glider_gui_params.json"
    def __init__(self):
        super().__init__()
        self.param_fields = {}
        self.setWindowTitle("Underwater Glider Simulator")
        self.setGeometry(100, 100, 1200, 800)
        # Main tab widget
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        # --- Design Tab ---
        design_tab = QWidget()
        design_layout = QHBoxLayout(design_tab)
        # Left: Parameter panel in a scroll area for neatness
        from PyQt5.QtWidgets import QScrollArea, QGroupBox, QFormLayout
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        param_widget = QWidget()
        param_layout = QVBoxLayout(param_widget)
        self.param_panel = QTabWidget()
        self._setup_parameter_tabs()
        param_layout.addWidget(self.param_panel)
        # --- Metric/Imperial Converter ---
        converter_group = QGroupBox("Metric/Imperial Converter")
        converter_layout = QFormLayout()
        self.metric_input = QLineEdit()
        self.imperial_input = QLineEdit()
        self.metric_input.setPlaceholderText("Meters or Kilograms")
        self.imperial_input.setPlaceholderText("Feet or Pounds")
        converter_layout.addRow("Metric:", self.metric_input)
        converter_layout.addRow("Imperial:", self.imperial_input)
        converter_group.setLayout(converter_layout)
        param_layout.addWidget(converter_group)
        # Connect conversion logic
        self.metric_input.textChanged.connect(self.metric_to_imperial)
        self.imperial_input.textChanged.connect(self.imperial_to_metric)
        scroll.setWidget(param_widget)
        # Make the scroll area a fixed width for better balance
        scroll.setMinimumWidth(350)
        scroll.setMaximumWidth(400)
        design_layout.addWidget(scroll, stretch=0)
        # Right: Feedback and summary stacked above render
        right_panel = QVBoxLayout()
        # Summary on top
        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        right_panel.addWidget(QLabel("Key Metrics:"))
        right_panel.addWidget(self.summary_label)
        # Feedback below summary
        self.feedback_label = QLabel()
        self.feedback_label.setWordWrap(True)
        right_panel.addWidget(QLabel("Design Feedback:"))
        right_panel.addWidget(self.feedback_label)
        # Render at the bottom
        cross_preview = Figure(figsize=(3.5, 1.5))
        self.cross_preview_canvas = FigureCanvas(cross_preview)
        self.cross_preview_fig = cross_preview
        right_panel.addWidget(self.cross_preview_canvas, stretch=1)
        # Apply Changes button at the very bottom
        self.apply_btn = QPushButton("Apply Changes")
        self.apply_btn.clicked.connect(self.on_apply_changes)
        right_panel.addWidget(self.apply_btn)
        design_layout.addLayout(right_panel, stretch=1)
        self.tabs.addTab(design_tab, "Design")
        # --- Cross Section Tab ---
        cross_tab = QWidget()
        cross_layout = QVBoxLayout(cross_tab)
        self.cross_fig = Figure(figsize=(8, 3))
        self.cross_canvas = FigureCanvas(self.cross_fig)
        cross_layout.addWidget(self.cross_canvas)
        self.tabs.addTab(cross_tab, "Cross Section")
        # --- Simulation Tab ---
        sim_tab = QWidget()
        sim_layout = QVBoxLayout(sim_tab)
        # Simulation controls
        sim_ctrl_layout = QHBoxLayout()
        # Time step
        sim_ctrl_layout.addWidget(QLabel("Time Step (dt):"))
        self.dt_spin = QDoubleSpinBox()
        self.dt_spin.setDecimals(3)
        self.dt_spin.setRange(0.001, 10.0)
        self.dt_spin.setValue(0.1)
        sim_ctrl_layout.addWidget(self.dt_spin)
        # Duration
        sim_ctrl_layout.addWidget(QLabel("Duration (s):"))
        self.tend_spin = QDoubleSpinBox()
        self.tend_spin.setDecimals(1)
        self.tend_spin.setRange(1, 10000)
        self.tend_spin.setValue(60)
        sim_ctrl_layout.addWidget(self.tend_spin)
        # Solver method
        sim_ctrl_layout.addWidget(QLabel("Solver:"))
        self.solver_combo = QComboBox()
        self.solver_combo.addItems(["RK45", "RK23", "DOP853", "LSODA"])
        sim_ctrl_layout.addWidget(self.solver_combo)
        # Initial depth
        sim_ctrl_layout.addWidget(QLabel("Init Depth (m):"))
        self.init_depth_spin = QDoubleSpinBox()
        self.init_depth_spin.setDecimals(2)
        self.init_depth_spin.setRange(-1000, 1000)
        self.init_depth_spin.setValue(10)
        sim_ctrl_layout.addWidget(self.init_depth_spin)
        # Initial pitch
        sim_ctrl_layout.addWidget(QLabel("Init Pitch (deg):"))
        self.init_pitch_spin = QDoubleSpinBox()
        self.init_pitch_spin.setDecimals(2)
        self.init_pitch_spin.setRange(-90, 90)
        self.init_pitch_spin.setValue(5)
        sim_ctrl_layout.addWidget(self.init_pitch_spin)
        # Control system
        sim_ctrl_layout.addWidget(QLabel("Control System:"))
        self.control_combo = QComboBox()
        self.control_combo.addItems(["Depth/Pitch Control", "Trajectory Following Control"])
        sim_ctrl_layout.addWidget(self.control_combo)
        sim_layout.addLayout(sim_ctrl_layout)
        self.sim_fig = Figure(figsize=(8, 6))
        self.sim_canvas = FigureCanvas(self.sim_fig)
        sim_layout.addWidget(self.sim_canvas)
        btn_run = QPushButton("Run Simulation")
        btn_run.clicked.connect(self.run_simulation)
        sim_layout.addWidget(btn_run)
        self.tabs.addTab(sim_tab, "Simulation")
        # --- Converter Tab ---
        converter_tab = UnitConverterWidget()
        self.tabs.addTab(converter_tab, "Unit Converter")
        # Load params, feedback, and initial cross section
        self.load_params()
        self.update_feedback()
        self.update_cross_section()
        self.update_cross_preview()
        
    def _setup_parameter_tabs(self):
        """Create parameter input tabs"""
        # Geometry tab
        geom_tab = QWidget()
        geom_layout = QVBoxLayout()
        
        # Hull group
        hull_group = QGroupBox("Hull Parameters")
        hull_layout = QVBoxLayout()
        self._add_parameter_field(hull_layout, "nose_length", "Nose Length (m)", "0.3")
        self._add_parameter_field(hull_layout, "nose_radius", "Nose Radius (m)", "0.15")
        self._add_parameter_field(hull_layout, "cyl_length", "Cylinder Length (m)", "1.4")
        self._add_parameter_field(hull_layout, "hull_radius", "Hull Radius (m)", "0.15")
        self._add_parameter_field(hull_layout, "tail_length", "Tail Length (m)", "0.3")
        self._add_parameter_field(hull_layout, "tail_radius", "Tail Radius (m)", "0.12")
        self._add_parameter_field(hull_layout, "hull_thickness", "Hull Thickness (m)", "0.005")
        self._add_parameter_field(hull_layout, "hull_density", "Hull Density (kg/m³)", "2700")
        hull_group.setLayout(hull_layout)
        geom_layout.addWidget(hull_group)
        
        # Ballast group
        ballast_group = QGroupBox("Ballast System")
        ballast_layout = QVBoxLayout()
        self._add_parameter_field(ballast_layout, "ballast_radius", "Ballast Radius (m)", "0.08")
        self._add_parameter_field(ballast_layout, "ballast_length", "Ballast Length (m)", "0.4")
        self._add_parameter_field(ballast_layout, "tank_thickness", "Tank Thickness (m)", "0.004")
        self._add_parameter_field(ballast_layout, "tank_density", "Tank Density (kg/m³)", "2700")
        self._add_parameter_field(ballast_layout, "ballast_position", "Ballast Position (x,y,z)", "0.8,0,0")
        ballast_group.setLayout(ballast_layout)
        geom_layout.addWidget(ballast_group)
        
        geom_tab.setLayout(geom_layout)
        self.param_panel.addTab(geom_tab, "Geometry")
        
        # Mass tab
        mass_tab = QWidget()
        mass_layout = QVBoxLayout()
        
        # Mass properties group
        mass_group = QGroupBox("Mass Properties")
        mass_group_layout = QVBoxLayout()
        self._add_parameter_field(mass_group_layout, "fixed_mass", "Fixed Mass (kg)", "20.0")
        self._add_parameter_field(mass_group_layout, "fixed_position", "Fixed Mass Position (x,y,z)", "0.7,0,0")
        self._add_parameter_field(mass_group_layout, "actuator_mass", "Actuator Mass (kg)", "2.0")
        self._add_parameter_field(mass_group_layout, "actuator_position", "Actuator Position (x,y,z)", "0.6,0,0")
        self._add_parameter_field(mass_group_layout, "piston_mass", "Piston Mass (kg)", "5.0")
        self._add_parameter_field(mass_group_layout, "piston_position", "Piston Position (x,y,z)", "0.6,0,0")
        self._add_parameter_field(mass_group_layout, "piston_travel", "Piston Travel (m)", "0.5")
        self._add_parameter_field(mass_group_layout, "I_dry_base", "Dry Inertia (diag) kg·m²", "2.0,3.0,1.5")
        mass_group.setLayout(mass_group_layout)
        mass_layout.addWidget(mass_group)
        
        mass_tab.setLayout(mass_layout)
        self.param_panel.addTab(mass_tab, "Mass")
        
        # Hydrodynamics tab
        hydro_tab = QWidget()
        hydro_layout = QVBoxLayout()
        
        hydro_group = QGroupBox("Hydrodynamic Properties")
        hydro_group_layout = QVBoxLayout()
        self._add_parameter_field(hydro_group_layout, "wing_span", "Wing Span (m)", "1.0")
        self._add_parameter_field(hydro_group_layout, "wing_chord", "Wing Chord (m)", "0.15")
        self._add_parameter_field(hydro_group_layout, "wing_position", "Wing Position (x,y,z)", "0.9,0,0")
        self._add_parameter_field(hydro_group_layout, "glider_length", "Glider Length (m)", "2.0")
        self._add_parameter_field(hydro_group_layout, "CD0", "Zero-Lift Drag", "0.1")
        self._add_parameter_field(hydro_group_layout, "CL_alpha", "Lift Slope", "6.28")
        self._add_parameter_field(hydro_group_layout, "CD_alpha", "Drag Alpha Coeff", "0.5")
        self._add_parameter_field(hydro_group_layout, "CM0", "Zero-Alpha Moment", "-0.02")
        self._add_parameter_field(hydro_group_layout, "CM_alpha", "Moment Alpha Slope", "-0.1")
        hydro_group.setLayout(hydro_group_layout)
        hydro_layout.addWidget(hydro_group)
        
        hydro_tab.setLayout(hydro_layout)
        self.param_panel.addTab(hydro_tab, "Hydrodynamics")
        
        # Add all params to GUI, including those currently hardcoded in run_simulation.py
        # Geometry tab additions
        self._add_parameter_field(hull_layout, "rho_water", "Water Density (kg/m³)", "1025.0")
        # Hydrodynamics tab additions
        self._add_parameter_field(hydro_group_layout, "desired_depth", "Desired Depth (m)", "20")
        self._add_parameter_field(hydro_group_layout, "desired_pitch", "Desired Pitch (deg)", "0")
        
    def _add_parameter_field(self, layout, name, label, default):
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel(label))
        field = QLineEdit(default)
        self.param_fields[name] = field
        hbox.addWidget(field)
        layout.addLayout(hbox)
        # Remove auto-update connection
        # field.textChanged.connect(self.on_param_changed)
        
    def on_apply_changes(self):
        self.save_params()
        self.update_feedback()
        self.update_cross_section()
        self.update_cross_preview()
        
    def save_params(self):
        params = {name: field.text() for name, field in self.param_fields.items()}
        try:
            with open(self.PARAMS_FILE, "w") as f:
                json.dump(params, f)
        except Exception as e:
            print(f"[Save Error] {e}")
            
    def load_params(self):
        if os.path.exists(self.PARAMS_FILE):
            try:
                with open(self.PARAMS_FILE, "r") as f:
                    params = json.load(f)
                for name, value in params.items():
                    if name in self.param_fields:
                        self.param_fields[name].setText(value)
            except Exception as e:
                print(f"[Load Error] {e}")
                
    def get_parameters(self):
        """Collect parameters from GUI fields"""
        params = {}
        for name, field in self.param_fields.items():
            try:
                # Handle different parameter types
                if name.endswith('_position'):
                    # Parse vector input (e.g., "0.8, 0, 0")
                    params[name] = np.array([float(x) for x in field.text().split(',')])
                elif name == 'I_dry_base':
                    # Parse inertia matrix (simplified diagonal)
                    diag = [float(x) for x in field.text().split(',')]
                    params[name] = np.diag(diag)
                else:
                    # Regular scalar parameter
                    params[name] = float(field.text())
            except ValueError:
                print(f"Invalid input for {name}")
                return None
        return params
    
    def update_feedback(self):
        # Try to parse all needed fields
        try:
            rho_water = 1025.0
            nose_length = float(self.param_fields['nose_length'].text())
            nose_radius = float(self.param_fields['nose_radius'].text())
            cyl_length = float(self.param_fields['cyl_length'].text())
            hull_radius = float(self.param_fields['hull_radius'].text())
            tail_length = float(self.param_fields['tail_length'].text())
            tail_radius = float(self.param_fields['tail_radius'].text())
            hull_thickness = float(self.param_fields['hull_thickness'].text())
            hull_density = float(self.param_fields['hull_density'].text())
            ballast_radius = float(self.param_fields['ballast_radius'].text())
            ballast_length = float(self.param_fields['ballast_length'].text())
            tank_thickness = float(self.param_fields['tank_thickness'].text())
            tank_density = float(self.param_fields['tank_density'].text())
            fixed_mass = float(self.param_fields['fixed_mass'].text())
            actuator_mass = float(self.param_fields['actuator_mass'].text())
            piston_mass = float(self.param_fields['piston_mass'].text())
            
            # Volumes
            V_hull = (1/3) * np.pi * nose_radius**2 * nose_length \
                + np.pi * hull_radius**2 * cyl_length \
                + (1/3) * np.pi * tail_radius**2 * tail_length
            V_ballast = np.pi * ballast_radius**2 * ballast_length
            ballast_mass_50 = rho_water * V_ballast * 0.5
            
            # Dry mass
            m_hull = (
                np.pi * nose_radius * np.sqrt(nose_radius**2 + nose_length**2)
                + 2 * np.pi * hull_radius * cyl_length
                + np.pi * tail_radius * np.sqrt(tail_radius**2 + tail_length**2)
            ) * hull_thickness * hull_density
            m_tank = 2 * np.pi * ballast_radius * ballast_length * tank_thickness * tank_density
            dry_mass = m_hull + m_tank + fixed_mass + actuator_mass + piston_mass
            total_mass_50 = dry_mass + ballast_mass_50
            buoyant_mass = rho_water * V_hull
            extra_mass = buoyant_mass - total_mass_50
            if extra_mass > 0:
                status = f"Add {extra_mass:.2f} kg for neutral buoyancy at 50% ballast."
            else:
                status = f"Glider is heavy by {-extra_mass:.2f} kg at 50% ballast."
            if abs(extra_mass) < 0.5:
                status += " (Nearly neutral)"
            elif extra_mass > 0:
                status += " (Positively buoyant)"
            else:
                status += " (Negatively buoyant)"
            self.feedback_label.setText(
                f"Hull Vol: {V_hull:.3f} m³ | Ballast Vol: {V_ballast:.3f} m³\n"
                f"Dry Mass: {dry_mass:.2f} kg | Ballast@50%: {ballast_mass_50:.2f} kg\n"
                f"Buoyant Mass: {buoyant_mass:.2f} kg | Total@50%: {total_mass_50:.2f} kg\n"
                f"{status}"
            )
        except Exception as e:
            self.feedback_label.setText(f"[Feedback Error] {e}")
        
    def update_cross_section(self):
        try:
            # Get parameters
            nose_length = float(self.param_fields['nose_length'].text())
            nose_radius = float(self.param_fields['nose_radius'].text())
            cyl_length = float(self.param_fields['cyl_length'].text())
            hull_radius = float(self.param_fields['hull_radius'].text())
            tail_length = float(self.param_fields['tail_length'].text())
            tail_radius = float(self.param_fields['tail_radius'].text())
            ballast_radius = float(self.param_fields['ballast_radius'].text())
            ballast_length = float(self.param_fields['ballast_length'].text())
            # Use ballast_position and piston_position from design tab
            ballast_pos = [float(x) for x in self.param_fields['ballast_position'].text().split(',')]
            moving_mass_pos = [float(x) for x in self.param_fields['piston_position'].text().split(',')]
            total_length = nose_length + cyl_length + tail_length
            ballast_x = ballast_pos[0]
            moving_mass_x = moving_mass_pos[0]
            self.cross_fig.clear()
            ax = self.cross_fig.add_subplot(111)
            # Draw torpedo-like hull: ellipse for nose, rectangle for body, ellipse for tail
            # Main body (cylinder)
            cyl_x = nose_length
            body = patches.Rectangle((cyl_x, -hull_radius), cyl_length, 2*hull_radius, color='#bbb', alpha=0.9, lw=2)
            ax.add_patch(body)
            # Nose (ellipse, left)
            nose = patches.Ellipse((nose_length/2, 0), nose_length, 2*nose_radius, color='#888', alpha=0.8, lw=2)
            ax.add_patch(nose)
            # Tail (ellipse, right)
            tail_x = nose_length + cyl_length + tail_length/2
            tail = patches.Ellipse((tail_x, 0), tail_length, 2*tail_radius, color='#888', alpha=0.8, lw=2)
            ax.add_patch(tail)
            # Ballast tank (red)
            ballast = patches.Rectangle((ballast_x, -ballast_radius), ballast_length, 2*ballast_radius, color='red', alpha=0.5, lw=2, label='Ballast Tank')
            ax.add_patch(ballast)
            # Moving mass (blue)
            moving_mass = patches.Circle((moving_mass_x, 0), 0.04, color='blue', alpha=0.8, label='Moving Mass')
            ax.add_patch(moving_mass)
            # CG/CB markers (optional, for clarity)
            cg_x = nose_length + cyl_length * 0.5
            cb_x = cg_x + 0.05 * total_length
            ax.plot([cg_x], [0], marker='o', color='green', markersize=10, label='CG')
            ax.plot([cb_x], [0], marker='s', color='cyan', markersize=10, label='CB')
            # Wings (optional, as a line)
            wing_span = float(self.param_fields.get('wing_span', QLineEdit('1.0')).text())
            # Use wing_position from design tab if available
            if 'wing_position' in self.param_fields:
                wing_pos = [float(x) for x in self.param_fields['wing_position'].text().split(',')]
                wing_x = wing_pos[0]
            else:
                wing_x = nose_length + cyl_length * 0.7
            ax.plot([wing_x, wing_x], [-wing_span/2, wing_span/2], color='purple', lw=4, label='Wings')
            # Labels and legend
            ax.set_xlim(-0.1, total_length + 0.2)
            ax.set_ylim(-max(nose_radius, hull_radius, tail_radius, ballast_radius, 0.2)*1.5, max(nose_radius, hull_radius, tail_radius, ballast_radius, 0.2)*1.5)
            ax.set_aspect('equal')
            ax.set_title('Glider Side View (Cross Section)')
            ax.axis('off')
            ax.legend(loc='upper right', fontsize=9)
            self.cross_fig.tight_layout()
            self.cross_canvas.draw()
        except Exception as e:
            self.cross_fig.clear()
            ax = self.cross_fig.add_subplot(111)
            ax.text(0.5, 0.5, f"[Cross Section Error]\n{e}", ha='center', va='center')
            ax.axis('off')
            self.cross_canvas.draw()
    
    def run_simulation(self):
        params = self.get_parameters()
        if params is None:
            return
        params['wing_area'] = params['wing_span'] * params['wing_chord']
        # Get simulation settings from GUI
        dt = self.dt_spin.value()
        t_end = self.tend_spin.value()
        solver = self.solver_combo.currentText()
        init_depth = self.init_depth_spin.value()
        init_pitch = self.init_pitch_spin.value()
        control_choice = self.control_combo.currentText()
        # Use user-settable desired depth/pitch if present
        desired_depth = params.get('desired_depth', 20)
        desired_pitch = params.get('desired_pitch', 0)
        # Select control system
        from glider_controls import depth_pitch_control, trajectory_following_control
        if control_choice == "Depth/Pitch Control":
            control_func = lambda t, s: depth_pitch_control(t, s, desired_depth, desired_pitch)
        elif control_choice == "Trajectory Following Control":
            waypoints = [np.array([0,0,desired_depth]), np.array([10,0,init_depth])]
            control_func = lambda t, s: trajectory_following_control(t, s, waypoints)
        else:
            control_func = lambda t, s: depth_pitch_control(t, s, desired_depth, desired_pitch)
        # Run simulation
        solution = run_simulation(
            params,
            control_func=control_func,
            t_end=t_end,
            dt=dt,
            init_depth=init_depth,
            init_pitch=init_pitch,
            solver=solver
        )
        self.update_plots(solution)
        
    def update_plots(self, solution):
        self.sim_fig.clear()
        # Depth vs time plot
        ax1 = self.sim_fig.add_subplot(211)
        ax1.plot(solution.t, solution.y[2, :])
        ax1.set_ylabel("Depth (m)")
        ax1.invert_yaxis()
        ax1.grid(True)
        # Pitch angle plot
        ax2 = self.sim_fig.add_subplot(212)
        pitch_angles = []
        for i in range(len(solution.t)):
            quat = solution.y[3:7, i]
            pitch = R.from_quat(quat).as_euler('zyx')[1]
            pitch_angles.append(np.degrees(pitch))
        ax2.plot(solution.t, pitch_angles)
        ax2.set_ylabel("Pitch (deg)")
        ax2.set_xlabel("Time (s)")
        ax2.grid(True)
        self.sim_fig.tight_layout()
        self.sim_canvas.draw()

    def update_cross_preview(self):
        try:
            # Use same logic as update_cross_section but smaller and no labels
            nose_length = float(self.param_fields['nose_length'].text())
            nose_radius = float(self.param_fields['nose_radius'].text())
            cyl_length = float(self.param_fields['cyl_length'].text())
            hull_radius = float(self.param_fields['hull_radius'].text())
            tail_length = float(self.param_fields['tail_length'].text())
            tail_radius = float(self.param_fields['tail_radius'].text())
            ballast_radius = float(self.param_fields['ballast_radius'].text())
            ballast_length = float(self.param_fields['ballast_length'].text())
            ballast_pos = [float(x) for x in self.param_fields['ballast_position'].text().split(',')]
            moving_mass_pos = [float(x) for x in self.param_fields['piston_position'].text().split(',')]
            total_length = nose_length + cyl_length + tail_length
            ballast_x = ballast_pos[0]
            moving_mass_x = moving_mass_pos[0]
            self.cross_preview_fig.clear()
            ax = self.cross_preview_fig.add_subplot(111)
            cyl_x = nose_length
            body = patches.Rectangle((cyl_x, -hull_radius), cyl_length, 2*hull_radius, color='#bbb', alpha=0.9, lw=1)
            ax.add_patch(body)
            nose = patches.Ellipse((nose_length/2, 0), nose_length, 2*nose_radius, color='#888', alpha=0.8, lw=1)
            ax.add_patch(nose)
            tail_x = nose_length + cyl_length + tail_length/2
            tail = patches.Ellipse((tail_x, 0), tail_length, 2*tail_radius, color='#888', alpha=0.8, lw=1)
            ax.add_patch(tail)
            ballast = patches.Rectangle((ballast_x, -ballast_radius), ballast_length, 2*ballast_radius, color='red', alpha=0.4, lw=1)
            ax.add_patch(ballast)
            moving_mass = patches.Circle((moving_mass_x, 0), 0.03, color='blue', alpha=0.7)
            ax.add_patch(moving_mass)
            wing_span = float(self.param_fields.get('wing_span', QLineEdit('1.0')).text())
            if 'wing_position' in self.param_fields:
                wing_pos = [float(x) for x in self.param_fields['wing_position'].text().split(',')]
                wing_x = wing_pos[0]
            else:
                wing_x = nose_length + cyl_length * 0.7
            ax.plot([wing_x, wing_x], [-wing_span/2, wing_span/2], color='purple', lw=2)
            ax.set_xlim(-0.1, total_length + 0.2)
            ax.set_ylim(-max(nose_radius, hull_radius, tail_radius, ballast_radius, 0.2)*1.2, max(nose_radius, hull_radius, tail_radius, ballast_radius, 0.2)*1.2)
            ax.set_aspect('equal')
            ax.axis('off')
            # Add legend
            handles = []
            handles.append(patches.Patch(color='#bbb', label='Hull'))
            handles.append(patches.Patch(color='#888', label='Nose/Tail'))
            handles.append(patches.Patch(color='red', alpha=0.4, label='Ballast Tank'))
            handles.append(patches.Patch(color='blue', alpha=0.7, label='Moving Mass'))
            handles.append(patches.Patch(color='purple', label='Wings'))
            ax.legend(handles=handles, loc='upper right', fontsize=8, frameon=True)
            self.cross_preview_fig.tight_layout()
            self.cross_preview_canvas.draw()
        except Exception:
            self.cross_preview_fig.clear()
            ax = self.cross_preview_fig.add_subplot(111)
            ax.axis('off')
            self.cross_preview_canvas.draw()
        # Add summary metrics for quick reference
        try:
            nose_length = float(self.param_fields['nose_length'].text())
            cyl_length = float(self.param_fields['cyl_length'].text())
            tail_length = float(self.param_fields['tail_length'].text())
            total_length = nose_length + cyl_length + tail_length
            hull_radius = float(self.param_fields['hull_radius'].text())
            wing_span = float(self.param_fields.get('wing_span', QLineEdit('1.0')).text())
            ballast_length = float(self.param_fields['ballast_length'].text())
            piston_travel = float(self.param_fields['piston_travel'].text())
            summary = (
                f"Total Length: {total_length:.2f} m\n"
                f"Hull Diameter: {2*hull_radius:.2f} m\n"
                f"Wing Span: {wing_span:.2f} m\n"
                f"Ballast Length: {ballast_length:.2f} m\n"
                f"Piston Travel: {piston_travel:.2f} m"
            )
            self.summary_label.setText(summary)
        except Exception:
            self.summary_label.setText("")
    
    def metric_to_imperial(self):
        try:
            val = float(self.metric_input.text())
            # Try to guess if it's a length or mass (simple heuristic)
            if val < 10:  # likely meters
                feet = val * 3.28084
                self.imperial_input.blockSignals(True)
                self.imperial_input.setText(f"{feet:.3f}")
                self.imperial_input.blockSignals(False)
            else:  # likely kilograms
                pounds = val * 2.20462
                self.imperial_input.blockSignals(True)
                self.imperial_input.setText(f"{pounds:.3f}")
                self.imperial_input.blockSignals(False)
        except Exception:
            self.imperial_input.blockSignals(True)
            self.imperial_input.setText("")
            self.imperial_input.blockSignals(False)
    def imperial_to_metric(self):
        try:
            val = float(self.imperial_input.text())
            # Try to guess if it's a length or mass (simple heuristic)
            if val < 33:  # likely feet
                meters = val / 3.28084
                self.metric_input.blockSignals(True)
                self.metric_input.setText(f"{meters:.3f}")
                self.metric_input.blockSignals(False)
            else:  # likely pounds
                kg = val / 2.20462
                self.metric_input.blockSignals(True)
                self.metric_input.setText(f"{kg:.3f}")
                self.metric_input.blockSignals(False)
        except Exception:
            self.metric_input.blockSignals(True)
            self.metric_input.setText("")
            self.metric_input.blockSignals(False)