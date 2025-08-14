import sys
import time
import math
from math import pi
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                            QTabWidget, QGroupBox, QSlider, QComboBox, QSpinBox, QDoubleSpinBox,
                            QProgressBar, QMessageBox, QGridLayout, QTableWidget, QTableWidgetItem)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from glider_simulation import run_simulation
import json
import os
import matplotlib.patches as patches
from scipy.spatial.transform import Rotation as R
from unit_converter import UnitConverterWidget

# Optional pandas import for data export
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

class SimulationWorker(QThread):
    """Worker thread for running simulations without freezing the GUI"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    
    def __init__(self, params, control_func, t_end, dt, init_depth, init_pitch, solver):
        super().__init__()
        self.params = params
        self.control_func = control_func
        self.t_end = t_end
        self.dt = dt
        self.init_depth = init_depth
        self.init_pitch = init_pitch
        self.solver = solver
        
    def run(self):
        try:
            # Run simulation with progress updates
            solution = run_simulation(
                self.params,
                control_func=self.control_func,
                t_end=self.t_end,
                dt=self.dt,
                init_depth=self.init_depth,
                init_pitch=self.init_pitch,
                solver=self.solver,
                progress_callback=self.progress.emit
            )
            self.finished.emit(solution)
        except Exception as e:
            self.error.emit(str(e))

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
        self.control_combo.addItems(["Depth/Pitch Control", "Trajectory Following Control", "Simple Depth Control"])
        sim_ctrl_layout.addWidget(self.control_combo)
        sim_layout.addLayout(sim_ctrl_layout)
        
        # Progress bar and status
        progress_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_label = QLabel("Ready to run simulation")
        self.time_label = QLabel("")
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(self.cancel_simulation)
        progress_layout.addWidget(self.status_label)
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.time_label)
        progress_layout.addWidget(self.cancel_btn)
        sim_layout.addLayout(progress_layout)
        
        # Physics diagnostics display
        physics_layout = QHBoxLayout()
        physics_layout.addWidget(QLabel("Physics Diagnostics:"))
        self.physics_label = QLabel("No simulation data")
        self.physics_label.setStyleSheet("background-color: #f0f0f0; padding: 5px; border: 1px solid #ccc;")
        self.physics_label.setWordWrap(True)
        physics_layout.addWidget(self.physics_label)
        
        # Control system summary display
        control_layout = QHBoxLayout()
        control_layout.addWidget(QLabel("Control Summary:"))
        self.control_label = QLabel("No control data")
        self.control_label.setStyleSheet("background-color: #e8f4f8; padding: 5px; border: 1px solid #87ceeb;")
        self.control_label.setWordWrap(True)
        control_layout.addWidget(self.control_label)
        
        sim_layout.addLayout(physics_layout)
        sim_layout.addLayout(control_layout)
        self.sim_fig = Figure(figsize=(8, 6))
        self.sim_canvas = FigureCanvas(self.sim_fig)
        sim_layout.addWidget(self.sim_canvas)
        
        # Add plot type selector for different diagnostic views
        plot_layout = QHBoxLayout()
        plot_layout.addWidget(QLabel("Plot Type:"))
        self.plot_combo = QComboBox()
        self.plot_combo.addItems([
            "Basic (Depth & Pitch)", 
            "3D Trajectory", 
            "Velocity Analysis", 
            "Control & Forces", 
            "Energy Analysis",
            "Control Analysis",
            "All Diagnostics"
        ])
        self.plot_combo.currentTextChanged.connect(self.update_plot_type)
        plot_layout.addWidget(self.plot_combo)
        sim_layout.addLayout(plot_layout)
        
        # Simulation control buttons
        button_layout = QHBoxLayout()
        self.btn_run = QPushButton("Run Simulation")
        self.btn_run.clicked.connect(self.run_simulation)
        button_layout.addWidget(self.btn_run)
        
        self.btn_export = QPushButton("Export Data")
        self.btn_export.clicked.connect(self.export_simulation_data)
        self.btn_export.setEnabled(False)
        button_layout.addWidget(self.btn_export)
        
        sim_layout.addLayout(button_layout)
        self.tabs.addTab(sim_tab, "Simulation")
        # --- Converter Tab ---
        converter_tab = UnitConverterWidget()
        self.tabs.addTab(converter_tab, "Unit Converter")

        # --- Depth Table Tab ---
        depth_table_tab = QWidget()
        depth_table_layout = QHBoxLayout(depth_table_tab)  # Changed to horizontal layout

        # Left panel for inputs
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(800)  # Increased width for input panel

        # Input section with side-by-side layout
        input_section = QGroupBox("Input Parameters")
        input_grid = QGridLayout()

        # Row 1: Disk positions
        disk_pos_label = QLabel("Disk Positions (m):")
        disk_pos_label.setToolTip("Comma-separated list of disk positions along the hull length.\nExample: 0.4, 1.0\nLeave empty to auto-generate based on number of disks.")
        input_grid.addWidget(disk_pos_label, 0, 0)
        self.table_disk_positions = QLineEdit()
        self.table_disk_positions.setPlaceholderText("0.4, 1.0 (comma separated)")
        self.table_disk_positions.setToolTip("Enter disk positions manually, or leave empty to auto-generate")
        input_grid.addWidget(self.table_disk_positions, 0, 1)

        n_disks_label = QLabel("Number of Internal Disks:")
        n_disks_label.setToolTip("Number of internal support disks.\nUsed when disk positions are not specified manually.\nEnd disks at 0 and L are always included.")
        input_grid.addWidget(n_disks_label, 0, 2)
        self.table_n_disks = QSpinBox()
        self.table_n_disks.setRange(0, 20)
        self.table_n_disks.setValue(2)
        self.table_n_disks.setToolTip("Used when disk positions are not specified manually")
        input_grid.addWidget(self.table_n_disks, 0, 3)

        # Row 2: Material properties
        E_label = QLabel("Young's Modulus (E):")
        E_label.setToolTip("Material stiffness.\nTypical values:\nAluminum: 69 GPa\nSteel: 200 GPa")
        input_grid.addWidget(E_label, 1, 0)
        self.table_E_input = QDoubleSpinBox()
        self.table_E_input.setDecimals(0)
        self.table_E_input.setRange(1e9, 1000e9)
        self.table_E_input.setValue(69e9)
        self.table_E_input.setSuffix(" Pa")
        self.table_E_input.setToolTip("Material stiffness (Pa)")
        input_grid.addWidget(self.table_E_input, 1, 1)

        nu_label = QLabel("Poisson's Ratio (ν):")
        nu_label.setToolTip("Material property for lateral strain.\nTypical values:\nAluminum: 0.33\nSteel: 0.3")
        input_grid.addWidget(nu_label, 1, 2)
        self.table_nu_input = QDoubleSpinBox()
        self.table_nu_input.setDecimals(3)
        self.table_nu_input.setRange(0.1, 0.5)
        self.table_nu_input.setValue(0.33)
        self.table_nu_input.setToolTip("Ratio of lateral to axial strain")
        input_grid.addWidget(self.table_nu_input, 1, 3)

        # Row 3: Material properties continued
        sigma_y_label = QLabel("Yield Strength (σy):")
        sigma_y_label.setToolTip("Material yield strength.\nTypical values:\nAluminum: 215 MPa\nSteel: 250-500 MPa")
        input_grid.addWidget(sigma_y_label, 2, 0)
        self.table_sigma_y_input = QDoubleSpinBox()
        self.table_sigma_y_input.setDecimals(0)
        self.table_sigma_y_input.setRange(50e6, 2000e6)
        self.table_sigma_y_input.setValue(215e6)
        self.table_sigma_y_input.setSuffix(" Pa")
        self.table_sigma_y_input.setToolTip("Material yield strength (Pa)")
        input_grid.addWidget(self.table_sigma_y_input, 2, 1)

        # Row 4: Geometry
        R_label = QLabel("Hull Radius (R):")
        R_label.setToolTip("External radius of the hull cylinder")
        input_grid.addWidget(R_label, 3, 0)
        self.table_R_input = QDoubleSpinBox()
        self.table_R_input.setDecimals(3)
        self.table_R_input.setRange(0.01, 10.0)
        self.table_R_input.setValue(0.15)
        self.table_R_input.setSuffix(" m")
        self.table_R_input.setToolTip("External radius of the hull")
        input_grid.addWidget(self.table_R_input, 3, 1)

        t_label = QLabel("Hull Thickness (t):")
        t_label.setToolTip("Thickness of the hull wall")
        input_grid.addWidget(t_label, 3, 2)
        self.table_t_input = QDoubleSpinBox()
        self.table_t_input.setDecimals(4)
        self.table_t_input.setRange(0.001, 0.1)
        self.table_t_input.setValue(0.005)
        self.table_t_input.setSuffix(" m")
        self.table_t_input.setToolTip("Thickness of the hull wall")
        input_grid.addWidget(self.table_t_input, 3, 3)

        # Row 5: Environment
        rho_label = QLabel("Water Density (ρ):")
        rho_label.setToolTip("Density of water.\nTypical values:\nFresh water: 1000 kg/m³\nSea water: 1025 kg/m³")
        input_grid.addWidget(rho_label, 4, 0)
        self.table_rho_input = QDoubleSpinBox()
        self.table_rho_input.setDecimals(1)
        self.table_rho_input.setRange(800, 1200)
        self.table_rho_input.setValue(1025.0)
        self.table_rho_input.setSuffix(" kg/m³")
        self.table_rho_input.setToolTip("Water density affects pressure at depth")
        input_grid.addWidget(self.table_rho_input, 4, 1)

        g_label = QLabel("Gravity (g):")
        g_label.setToolTip("Gravitational acceleration.\nStandard value: 9.81 m/s²")
        input_grid.addWidget(g_label, 4, 2)
        self.table_g_input = QDoubleSpinBox()
        self.table_g_input.setDecimals(2)
        self.table_g_input.setRange(9.0, 10.0)
        self.table_g_input.setValue(9.81)
        self.table_g_input.setSuffix(" m/s²")
        self.table_g_input.setToolTip("Gravitational acceleration")
        input_grid.addWidget(self.table_g_input, 4, 3)

        # Row 6: Desired depth
        desired_depth_label = QLabel("Desired Depth:")
        desired_depth_label.setToolTip("Target operating depth for safety factor calculation")
        input_grid.addWidget(desired_depth_label, 5, 0)
        self.table_desired_depth = QDoubleSpinBox()
        self.table_desired_depth.setDecimals(1)
        self.table_desired_depth.setRange(0.1, 10000.0)
        self.table_desired_depth.setValue(100.0)
        self.table_desired_depth.setSuffix(" m")
        self.table_desired_depth.setToolTip("Target operating depth")
        input_grid.addWidget(self.table_desired_depth, 5, 1)

        input_section.setLayout(input_grid)
        left_layout.addWidget(input_section)

        # Safety factors section
        safety_group = QGroupBox("Safety Factors")
        safety_grid = QGridLayout()

        # Row 1
        eta_label = QLabel("End Effect (η_end):")
        eta_label.setToolTip("End effect multiplier for overall buckling.\nTypical value: 1.15\nHigher values are more conservative.")
        safety_grid.addWidget(eta_label, 0, 0)
        self.table_eta_end = QDoubleSpinBox()
        self.table_eta_end.setDecimals(2)
        self.table_eta_end.setRange(0.5, 2.0)
        self.table_eta_end.setValue(1.15)
        self.table_eta_end.setToolTip("Accounts for end effects in overall buckling")
        safety_grid.addWidget(self.table_eta_end, 0, 1)

        kdf_overall_label = QLabel("Overall KDF:")
        kdf_overall_label.setToolTip("Knockdown factor for overall buckling.\nTypical value: 0.65\nAccounts for imperfections and uncertainties.")
        safety_grid.addWidget(kdf_overall_label, 0, 2)
        self.table_kdf_overall = QDoubleSpinBox()
        self.table_kdf_overall.setDecimals(2)
        self.table_kdf_overall.setRange(0.1, 1.0)
        self.table_kdf_overall.setValue(0.65)
        self.table_kdf_overall.setToolTip("Knockdown factor for overall buckling")
        safety_grid.addWidget(self.table_kdf_overall, 0, 3)

        # Row 2
        kdf_if_label = QLabel("Interframe KDF:")
        kdf_if_label.setToolTip("Knockdown factor for interframe buckling.\nTypical value: 0.75\nAccounts for imperfections between frames.")
        safety_grid.addWidget(kdf_if_label, 1, 0)
        self.table_kdf_interframe = QDoubleSpinBox()
        self.table_kdf_interframe.setDecimals(2)
        self.table_kdf_interframe.setRange(0.1, 1.0)
        self.table_kdf_interframe.setValue(0.75)
        self.table_kdf_interframe.setToolTip("Knockdown factor for interframe buckling")
        safety_grid.addWidget(self.table_kdf_interframe, 1, 1)

        phi_label = QLabel("Yield Factor (φ):")
        phi_label.setToolTip("Yield strength reduction factor.\nTypical value: 0.80\nAccounts for material variability.")
        safety_grid.addWidget(phi_label, 1, 2)
        self.table_phi_yield = QDoubleSpinBox()
        self.table_phi_yield.setDecimals(2)
        self.table_phi_yield.setRange(0.1, 1.0)
        self.table_phi_yield.setValue(0.80)
        self.table_phi_yield.setToolTip("Reduction factor for yield strength")
        safety_grid.addWidget(self.table_phi_yield, 1, 3)

        # Row 3
        gamma_label = QLabel("Load Factor (γ):")
        gamma_label.setToolTip("Global load factor.\nTypical value: 1.25\nIncreases loads for safety margin.")
        safety_grid.addWidget(gamma_label, 2, 0)
        self.table_gamma_global = QDoubleSpinBox()
        self.table_gamma_global.setDecimals(2)
        self.table_gamma_global.setRange(0.5, 3.0)
        self.table_gamma_global.setValue(1.25)
        self.table_gamma_global.setToolTip("Global factor applied to all loads")
        safety_grid.addWidget(self.table_gamma_global, 2, 1)

        n_waves_label = QLabel("Waves (n):")
        n_waves_label.setToolTip("Number of circumferential waves.\nAuto: Find optimal n for each bay\nManual: Use specified value")
        safety_grid.addWidget(n_waves_label, 2, 2)
        self.table_n_waves = QComboBox()
        self.table_n_waves.addItems(["Auto"] + [str(i) for i in range(2, 21)])
        self.table_n_waves.setCurrentText("Auto")
        self.table_n_waves.setToolTip("Number of waves in buckling mode shape")
        safety_grid.addWidget(self.table_n_waves, 2, 3)

        safety_group.setLayout(safety_grid)
        left_layout.addWidget(safety_group)

        # Coefficients section
        coeff_group = QGroupBox("Engineering Coefficients")
        coeff_grid = QGridLayout()

        K_if_label = QLabel("Interframe (K_if):")
        K_if_label.setToolTip("Interframe buckling coefficient.\nTypical value: 1.25\nAdjusts interframe buckling pressure.")
        coeff_grid.addWidget(K_if_label, 0, 0)
        self.table_K_if = QDoubleSpinBox()
        self.table_K_if.setDecimals(2)
        self.table_K_if.setRange(0.5, 3.0)
        self.table_K_if.setValue(1.25)
        self.table_K_if.setToolTip("Coefficient for interframe buckling calculation")
        coeff_grid.addWidget(self.table_K_if, 0, 1)

        C_label = QLabel("Overall (C):")
        C_label.setToolTip("Overall buckling coefficient.\nTypical value: 2.0\nAdjusts overall buckling pressure.")
        coeff_grid.addWidget(C_label, 0, 2)
        self.table_C_overall = QDoubleSpinBox()
        self.table_C_overall.setDecimals(1)
        self.table_C_overall.setRange(0.5, 5.0)
        self.table_C_overall.setValue(2.0)
        self.table_C_overall.setToolTip("Coefficient for overall buckling calculation")
        coeff_grid.addWidget(self.table_C_overall, 0, 3)

        coeff_group.setLayout(coeff_grid)
        left_layout.addWidget(coeff_group)

        # Control buttons
        button_layout = QHBoxLayout()
        self.calculate_table_btn = QPushButton("Calculate Depths")
        self.calculate_table_btn.clicked.connect(self.update_depth_table)
        button_layout.addWidget(self.calculate_table_btn)

        self.sync_table_btn = QPushButton("Sync with Design")
        self.sync_table_btn.clicked.connect(self.sync_table_inputs)
        button_layout.addWidget(self.sync_table_btn)

        left_layout.addLayout(button_layout)

        # Add left panel to main layout
        depth_table_layout.addWidget(left_panel)

        # Right panel for results with scroll area
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Create scroll area for results
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Create widget to hold scrollable content
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # Results summary
        self.depth_summary = QLabel("Click Calculate to see results")
        self.depth_summary.setStyleSheet("background-color: #f0f8ff; padding: 10px; border: 1px solid #87ceeb;")
        self.depth_summary.setWordWrap(True)
        scroll_layout.addWidget(self.depth_summary)

        # Results table
        from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView
        self.depth_table = QTableWidget()
        self.depth_table.setColumnCount(5)
        self.depth_table.setHorizontalHeaderLabels([
            "Bay Number", 
            "Start Position (m)", 
            "End Position (m)", 
            "Length (m)", 
            "Safe Depth (m)"
        ])
        header = self.depth_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        scroll_layout.addWidget(self.depth_table)

        # Set scroll area content
        scroll_area.setWidget(scroll_content)
        right_layout.addWidget(scroll_area)

        # Add right panel to main layout
        depth_table_layout.addWidget(right_panel)

        self.tabs.addTab(depth_table_tab, "Depth Table")

        # Load params, feedback, and initial cross section
        self.load_params()
        self.update_feedback()
        self.update_cross_section()
        self.update_cross_preview()
        
        # Initialize simulation worker
        self.simulation_worker = None
        
    def closeEvent(self, event):
        """Clean up threads when closing the application"""
        if self.simulation_worker and self.simulation_worker.isRunning():
            self.simulation_worker.terminate()
            self.simulation_worker.wait()
        event.accept()
        
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
        # Prevent multiple simulations from running simultaneously
        if self.simulation_worker and self.simulation_worker.isRunning():
            QMessageBox.information(self, "Simulation Running", "Please wait for the current simulation to complete.")
            return
            
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
        
        # Store control parameters for analysis
        self.control_params = {
            'desired_depth': desired_depth,
            'desired_pitch': desired_pitch,
            'control_system': control_choice,
            'init_depth': init_depth,
            'init_pitch': init_pitch
        }
        
        # Select control system
        from glider_controls import depth_pitch_control, trajectory_following_control, simple_depth_control
        if control_choice == "Depth/Pitch Control":
            control_func = lambda t, s: depth_pitch_control(t, s, desired_depth, desired_pitch)
        elif control_choice == "Trajectory Following Control":
            waypoints = [np.array([0,0,desired_depth]), np.array([10,0,init_depth])]
            control_func = lambda t, s: trajectory_following_control(t, s, waypoints)
        elif control_choice == "Simple Depth Control":
            control_func = lambda t, s: simple_depth_control(t, s, desired_depth)
        else:
            control_func = lambda t, s: depth_pitch_control(t, s, desired_depth, desired_pitch)
        
        # Setup progress bar and controls
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.status_label.setText("Initializing simulation...")
        self.cancel_btn.setVisible(True)
        self.time_label.setText("")
        
        # Record start time for timing estimates
        self.sim_start_time = time.time()
        
        # Create and start worker thread
        self.simulation_worker = SimulationWorker(
            params, control_func, t_end, dt, init_depth, init_pitch, solver
        )
        self.simulation_worker.progress.connect(self.update_progress)
        self.simulation_worker.finished.connect(self.simulation_completed)
        self.simulation_worker.error.connect(self.simulation_error)
        self.simulation_worker.start()
        
        # Disable buttons during simulation
        self.btn_run.setEnabled(False)
        self.btn_export.setEnabled(False)
        
    def update_progress(self, value):
        """Update progress bar during simulation"""
        self.progress_bar.setValue(value)
        if value < 100:
            # Calculate time estimates
            elapsed_time = time.time() - self.sim_start_time
            if value > 10:  # Only show estimates after initialization
                estimated_total = elapsed_time * 100 / value
                remaining_time = estimated_total - elapsed_time
                if remaining_time > 0:
                    self.time_label.setText(f"Est: {remaining_time:.1f}s remaining")
                    self.status_label.setText(f"Running simulation... {value}%")
                else:
                    self.time_label.setText("")
                    self.status_label.setText(f"Running simulation... {value}%")
            else:
                self.status_label.setText(f"Initializing... {value}%")
        else:
            self.status_label.setText("Simulation completed!")
            self.time_label.setText("")
    
    def update_physics_diagnostics(self, solution):
        """Update physics diagnostics display with key parameters"""
        try:
            # Check if solution is valid
            if solution is None or not hasattr(solution, 't') or not hasattr(solution, 'y'):
                self.physics_label.setText("No valid simulation data available")
                return
                
            # Extract key physics parameters
            current_time = solution.t[-1]
            current_depth = solution.y[2, -1]
            current_x = solution.y[0, -1]
            current_y = solution.y[1, -1]
            
            # Calculate velocities
            vx = np.gradient(solution.y[0, :], solution.t)[-1]
            vy = np.gradient(solution.y[1, :], solution.t)[-1]
            vz = np.gradient(solution.y[2, :], solution.t)[-1]
            speed = np.sqrt(vx**2 + vy**2 + vz**2)
            
            # Calculate accelerations
            if len(solution.t) > 2:
                # Use finite difference for acceleration
                dt = solution.t[1] - solution.t[0]
                ax = (vx - np.gradient(solution.y[0, :], solution.t)[-2]) / dt if len(solution.t) > 2 else 0
                ay = (vy - np.gradient(solution.y[1, :], solution.t)[-2]) / dt if len(solution.t) > 2 else 0
                az = (vz - np.gradient(solution.y[2, :], solution.t)[-2]) / dt if len(solution.t) > 2 else 0
            else:
                ax = ay = az = 0
            
            # Get current orientation
            current_quat = solution.y[3:7, -1]
            current_rot = R.from_quat(current_quat)
            euler_angles = current_rot.as_euler('zyx', degrees=True)
            pitch = euler_angles[1]
            yaw = euler_angles[0]
            roll = euler_angles[2]
            
            # Calculate forces (F = ma)
            mass = 50.0  # Approximate mass
            Fx = mass * ax
            Fy = mass * ay
            Fz = mass * az
            
            # Energy analysis
            g = 9.81
            PE = mass * g * current_depth
            KE = 0.5 * mass * speed**2
            TE = PE + KE
            
            # Format diagnostics string
            diagnostics = (
                f"Time: {current_time:.2f}s | "
                f"Position: ({current_x:.2f}, {current_y:.2f}, {current_depth:.2f})m\n"
                f"Velocity: ({vx:.3f}, {vy:.3f}, {vz:.3f}) m/s | Speed: {speed:.3f} m/s\n"
                f"Acceleration: ({ax:.3f}, {ay:.3f}, {az:.3f}) m/s²\n"
                f"Orientation: Pitch={pitch:.1f}° Yaw={yaw:.1f}° Roll={roll:.1f}°\n"
                f"Forces: ({Fx:.1f}, {Fy:.1f}, {Fz:.1f}) N\n"
                f"Energy: PE={PE:.1f}J KE={KE:.1f}J Total={TE:.1f}J"
            )
            
            self.physics_label.setText(diagnostics)
            
        except Exception as e:
            self.physics_label.setText(f"Diagnostics Error: {str(e)}")
            print(f"Physics diagnostics error details: {e}")
            import traceback
            traceback.print_exc()

    def update_control_summary(self, solution):
        """Update control system summary display"""
        try:
            if not hasattr(self, 'control_params') or solution is None:
                self.control_label.setText("No control data available")
                return
            
            # Extract control performance metrics
            desired_depth = self.control_params['desired_depth']
            actual_depth = solution.y[2, :]
            depth_error = actual_depth - desired_depth
            
            # Calculate performance metrics
            max_error = np.max(np.abs(depth_error))
            final_error = np.abs(depth_error[-1])
            settling_time = None
            
            # Find settling time (when error stays within 5% of target)
            tolerance = 0.05 * abs(desired_depth)
            for i, error in enumerate(depth_error):
                if abs(error) <= tolerance:
                    # Check if it stays within tolerance
                    if all(abs(depth_error[j]) <= tolerance for j in range(i, len(depth_error))):
                        settling_time = solution.t[i]
                        break
            
            # Ballast and piston analysis
            ballast_mass = solution.y[13, :]
            piston_position = solution.y[14, :]
            piston_velocity = solution.y[15, :]
            
            ballast_change = ballast_mass[-1] - ballast_mass[0]
            piston_travel_used = piston_position[-1] - piston_position[0]
            max_piston_velocity = np.max(np.abs(piston_velocity))
            
            # Format control summary
            control_summary = (
                f"Control System: {self.control_params['control_system']}\n"
                f"Target Depth: {desired_depth:.1f}m | Final Depth: {actual_depth[-1]:.1f}m\n"
                f"Max Error: {max_error:.2f}m | Final Error: {final_error:.2f}m\n"
                f"Settling Time: {settling_time:.1f}s" if settling_time else "Settling Time: Not reached"
            )
            
            actuator_summary = (
                f"\nActuator Performance:\n"
                f"Ballast Change: {ballast_change:.3f}kg | Piston Travel: {piston_travel_used:.3f}m\n"
                f"Max Piston Velocity: {max_piston_velocity:.3f}m/s"
            )
            
            self.control_label.setText(control_summary + actuator_summary)
            
        except Exception as e:
            self.control_label.setText(f"Control Summary Error: {str(e)}")
            print(f"Control summary error details: {e}")
            import traceback
            traceback.print_exc()
    
    def export_simulation_data(self):
        """Export simulation data to CSV for further analysis"""
        if not hasattr(self, 'last_solution') or self.last_solution is None:
            QMessageBox.warning(self, "No Data", "No simulation data to export. Run a simulation first.")
            return
        
        try:
            from PyQt5.QtWidgets import QFileDialog
            filename, _ = QFileDialog.getSaveFileName(
                self, "Export Simulation Data", "", "CSV Files (*.csv)"
            )
            
            if filename:
                solution = self.last_solution
                
                # Calculate derived quantities
                vx = np.gradient(solution.y[0, :], solution.t)
                vy = np.gradient(solution.y[1, :], solution.t)
                vz = np.gradient(solution.y[2, :], solution.t)
                speed = np.sqrt(vx**2 + vy**2 + vz**2)
                
                ax = np.gradient(vx, solution.t)
                ay = np.gradient(vy, solution.t)
                az = np.gradient(vz, solution.t)
                
                # Get Euler angles
                pitch_angles, yaw_angles, roll_angles = [], [], []
                for i in range(len(solution.t)):
                    quat = solution.y[3:7, i]
                    rot = R.from_quat(quat)
                    euler = rot.as_euler('zyx', degrees=True)
                    yaw_angles.append(euler[0])
                    pitch_angles.append(euler[1])
                    roll_angles.append(euler[2])
                
                # Calculate forces and energy
                mass = 50.0
                g = 9.81
                Fx = mass * ax
                Fy = mass * ay
                Fz = mass * az
                PE = mass * g * solution.y[2, :]
                KE = 0.5 * mass * speed**2
                TE = PE + KE
                
                # Create data dictionary
                data = {
                    'Time (s)': solution.t,
                    'X (m)': solution.y[0, :],
                    'Y (m)': solution.y[1, :],
                    'Z (m)': solution.y[2, :],
                    'Vx (m/s)': vx,
                    'Vy (m/s)': vy,
                    'Vz (m/s)': vz,
                    'Speed (m/s)': speed,
                    'Ax (m/s²)': ax,
                    'Ay (m/s²)': ay,
                    'Az (m/s²)': az,
                    'Pitch (deg)': pitch_angles,
                    'Yaw (deg)': yaw_angles,
                    'Roll (deg)': roll_angles,
                    'Fx (N)': Fx,
                    'Fy (N)': Fy,
                    'Fz (N)': Fz,
                    'Potential Energy (J)': PE,
                    'Kinetic Energy (J)': KE,
                    'Total Energy (J)': TE
                }
                
                # Export to CSV
                if PANDAS_AVAILABLE:
                    df = pd.DataFrame(data)
                    df.to_csv(filename, index=False)
                else:
                    # Fallback to basic CSV export
                    import csv
                    with open(filename, 'w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=data.keys())
                        writer.writeheader()
                        for i in range(len(next(iter(data.values())))):
                            row = {k: v[i] for k, v in data.items()}
                            writer.writerow(row)
                
                QMessageBox.information(self, "Export Successful", 
                                      f"Simulation data exported to:\n{filename}")
                
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export data:\n{str(e)}")
            
    def cancel_simulation(self):
        """Cancel the running simulation"""
        if self.simulation_worker and self.simulation_worker.isRunning():
            self.simulation_worker.terminate()
            self.simulation_worker.wait()
            self.simulation_worker = None
            self.progress_bar.setVisible(False)
            self.cancel_btn.setVisible(False)
            self.time_label.setText("")
            self.status_label.setText("Simulation cancelled")
            self.btn_run.setEnabled(True)
            self.btn_export.setEnabled(False)
        
    def simulation_completed(self, solution):
        """Handle simulation completion"""
        self.progress_bar.setVisible(False)
        self.cancel_btn.setVisible(False)
        self.time_label.setText("")
        self.status_label.setText("Simulation completed! Updating plots...")
        self.btn_run.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.update_plots(solution)
        self.update_physics_diagnostics(solution)
        self.update_control_summary(solution)
        
    def simulation_error(self, error_msg):
        """Handle simulation errors"""
        self.progress_bar.setVisible(False)
        self.cancel_btn.setVisible(False)
        self.time_label.setText("")
        self.status_label.setText(f"Simulation failed: {error_msg}")
        self.btn_run.setEnabled(True)
        self.btn_export.setEnabled(False)
        QMessageBox.critical(self, "Simulation Error", f"An error occurred during simulation:\n{error_msg}")
        
    def update_plot_type(self):
        """Update plots when plot type selection changes"""
        if hasattr(self, 'last_solution') and self.last_solution is not None:
            try:
                self.update_plots(self.last_solution)
            except Exception as e:
                print(f"Error updating plot type: {e}")
                import traceback
                traceback.print_exc()
        
    def update_plots(self, solution):
        """Update plots based on selected plot type"""
        try:
            # Check if solution is valid
            if solution is None or not hasattr(solution, 't') or not hasattr(solution, 'y'):
                print("Invalid solution object in update_plots")
                return
                
            self.last_solution = solution  # Store for plot type changes
            plot_type = self.plot_combo.currentText()
            
            if plot_type == "Basic (Depth & Pitch)":
                self.plot_basic(solution)
            elif plot_type == "3D Trajectory":
                self.plot_3d_trajectory(solution)
            elif plot_type == "Velocity Analysis":
                self.plot_velocity_analysis(solution)
            elif plot_type == "Control & Forces":
                self.plot_control_forces(solution)
            elif plot_type == "Energy Analysis":
                self.plot_energy_analysis(solution)
            elif plot_type == "Control Analysis":
                self.plot_control_analysis(solution)
            elif plot_type == "All Diagnostics":
                self.plot_all_diagnostics(solution)
            else:
                self.plot_basic(solution)
        except Exception as e:
            print(f"Error in update_plots: {e}")
            import traceback
            traceback.print_exc()
    
    def plot_basic(self, solution):
        """Basic depth and pitch plots"""
        self.sim_fig.clear()
        # Depth vs time plot
        ax1 = self.sim_fig.add_subplot(211)
        ax1.plot(solution.t, solution.y[2, :])
        ax1.set_ylabel("Depth (m)")
        ax1.invert_yaxis()
        ax1.grid(True)
        ax1.set_title("Depth vs Time")
        
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
        ax2.set_title("Pitch Angle vs Time")
        
        self.sim_fig.tight_layout()
        self.sim_canvas.draw()
    
    def plot_3d_trajectory(self, solution):
        """3D trajectory plot"""
        self.sim_fig.clear()
        ax = self.sim_fig.add_subplot(111, projection='3d')
        
        # Extract position data
        x = solution.y[0, :]
        y = solution.y[1, :]
        z = solution.y[2, :]
        
        # Plot 3D trajectory
        ax.plot(x, y, z, 'b-', linewidth=2, label='Trajectory')
        ax.scatter(x[0], y[0], z[0], c='g', s=100, label='Start')
        ax.scatter(x[-1], y[-1], z[-1], c='r', s=100, label='End')
        
        # Add arrows for orientation at key points
        for i in range(0, len(solution.t), max(1, len(solution.t)//10)):
            quat = solution.y[3:7, i]
            rot = R.from_quat(quat)
            # Get forward direction (assuming x is forward)
            forward = rot.apply([1, 0, 0])
            ax.quiver(x[i], y[i], z[i], forward[0], forward[1], forward[2], 
                     length=0.5, color='red', alpha=0.7)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Depth (m)')
        ax.set_title('3D Trajectory')
        ax.legend()
        ax.invert_zaxis()  # Depth increases downward
        
        self.sim_fig.tight_layout()
        self.sim_canvas.draw()
    
    def plot_velocity_analysis(self, solution):
        """Velocity component analysis"""
        self.sim_fig.clear()
        
        # Calculate velocities from position derivatives
        dt = np.diff(solution.t)
        vx = np.gradient(solution.y[0, :], solution.t)
        vy = np.gradient(solution.y[1, :], solution.t)
        vz = np.gradient(solution.y[2, :], solution.t)
        speed = np.sqrt(vx**2 + vy**2 + vz**2)
        
        # Velocity components
        ax1 = self.sim_fig.add_subplot(221)
        ax1.plot(solution.t, vx, 'r-', label='Vx')
        ax1.plot(solution.t, vy, 'g-', label='Vy')
        ax1.plot(solution.t, vz, 'b-', label='Vz')
        ax1.set_ylabel('Velocity (m/s)')
        ax1.grid(True)
        ax1.legend()
        ax1.set_title('Velocity Components')
        
        # Speed magnitude
        ax2 = self.sim_fig.add_subplot(222)
        ax2.plot(solution.t, speed, 'k-', linewidth=2)
        ax2.set_ylabel('Speed (m/s)')
        ax2.grid(True)
        ax2.set_title('Speed Magnitude')
        
        # Angular velocities (from quaternions)
        ax3 = self.sim_fig.add_subplot(223)
        omega_x, omega_y, omega_z = [], [], []
        for i in range(len(solution.t)):
            quat = solution.y[3:7, i]
            # Approximate angular velocity from quaternion differences
            if i > 0:
                dq = quat - solution.y[3:7, i-1]
                dt_ang = solution.t[i] - solution.t[i-1]
                if dt_ang > 0:
                    omega = 2 * dq / dt_ang
                    omega_x.append(omega[0])
                    omega_y.append(omega[1])
                    omega_z.append(omega[2])
                else:
                    omega_x.append(0)
                    omega_y.append(0)
                    omega_z.append(0)
            else:
                omega_x.append(0)
                omega_y.append(0)
                omega_z.append(0)
        
        ax3.plot(solution.t, omega_x, 'r-', label='ωx')
        ax3.plot(solution.t, omega_y, 'g-', label='ωy')
        ax3.plot(solution.t, omega_z, 'b-', label='ωz')
        ax3.set_ylabel('Angular Velocity (rad/s)')
        ax3.set_xlabel('Time (s)')
        ax3.grid(True)
        ax3.legend()
        ax3.set_title('Angular Velocities')
        
        # Velocity phase space
        ax4 = self.sim_fig.add_subplot(224)
        ax4.scatter(vx, vz, c=solution.t, cmap='viridis', alpha=0.7)
        ax4.set_xlabel('Vx (m/s)')
        ax4.set_ylabel('Vz (m/s)')
        ax4.grid(True)
        ax4.set_title('Velocity Phase Space (Vx vs Vz)')
        
        self.sim_fig.tight_layout()
        self.sim_canvas.draw()
    
    def plot_control_forces(self, solution):
        """Control inputs and forces analysis"""
        self.sim_fig.clear()
        
        # Extract control inputs (approximate from state changes)
        dt = np.diff(solution.t)
        ax = np.gradient(solution.y[0, :], solution.t)
        ay = np.gradient(solution.y[1, :], solution.t)
        az = np.gradient(solution.y[2, :], solution.t)
        
        # Control inputs (approximate)
        ax1 = self.sim_fig.add_subplot(221)
        ax1.plot(solution.t, ax, 'r-', label='Ax')
        ax1.plot(solution.t, ay, 'g-', label='Ay')
        ax1.plot(solution.t, az, 'b-', label='Az')
        ax1.set_ylabel('Acceleration (m/s²)')
        ax1.grid(True)
        ax1.legend()
        ax1.set_title('Acceleration Components')
        
        # Force analysis (F = ma, approximate)
        mass = 50.0  # Approximate glider mass
        Fx = mass * ax
        Fy = mass * ay
        Fz = mass * az
        
        ax2 = self.sim_fig.add_subplot(222)
        ax2.plot(solution.t, Fx, 'r-', label='Fx')
        ax2.plot(solution.t, Fy, 'g-', label='Fy')
        ax2.plot(solution.t, Fz, 'b-', label='Fz')
        ax2.set_ylabel('Force (N)')
        ax2.grid(True)
        ax2.legend()
        ax2.set_title('Force Components')
        
        # Control effort (integral of force)
        control_effort_x = np.cumsum(np.abs(Fx)) * np.mean(np.diff(solution.t))
        control_effort_z = np.cumsum(np.abs(Fz)) * np.mean(np.diff(solution.t))
        
        ax3 = self.sim_fig.add_subplot(223)
        ax3.plot(solution.t, control_effort_x, 'r-', label='X Control Effort')
        ax3.plot(solution.t, control_effort_z, 'b-', label='Z Control Effort')
        ax3.set_ylabel('Control Effort (N·s)')
        ax3.set_xlabel('Time (s)')
        ax3.grid(True)
        ax3.legend()
        ax3.set_title('Cumulative Control Effort')
        
        # Force phase space
        ax4 = self.sim_fig.add_subplot(224)
        ax4.scatter(Fx, Fz, c=solution.t, cmap='plasma', alpha=0.7)
        ax4.set_xlabel('Fx (N)')
        ax4.set_ylabel('Fz (N)')
        ax4.grid(True)
        ax4.set_title('Force Phase Space (Fx vs Fz)')
        
        self.sim_fig.tight_layout()
        self.sim_canvas.draw()
    
    def plot_energy_analysis(self, solution):
        """Energy analysis plots"""
        self.sim_fig.clear()
        
        # Calculate energies
        g = 9.81  # gravity
        mass = 50.0  # approximate mass
        
        # Potential energy (PE = mgh, where h is depth)
        depth = solution.y[2, :]
        PE = mass * g * depth
        
        # Kinetic energy (KE = 0.5 * m * v²)
        vx = np.gradient(solution.y[0, :], solution.t)
        vy = np.gradient(solution.y[1, :], solution.t)
        vz = np.gradient(solution.y[2, :], solution.t)
        speed = np.sqrt(vx**2 + vy**2 + vz**2)
        KE = 0.5 * mass * speed**2
        
        # Total energy
        TE = PE + KE
        
        # Energy plots
        ax1 = self.sim_fig.add_subplot(221)
        ax1.plot(solution.t, PE, 'b-', label='Potential Energy')
        ax1.plot(solution.t, KE, 'r-', label='Kinetic Energy')
        ax1.plot(solution.t, TE, 'k-', label='Total Energy')
        ax1.set_ylabel('Energy (J)')
        ax1.grid(True)
        ax1.legend()
        ax1.set_title('Energy vs Time')
        
        # Energy conservation (should be relatively constant)
        ax2 = self.sim_fig.add_subplot(222)
        energy_change = np.diff(TE)
        ax2.plot(solution.t[1:], energy_change, 'g-')
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_ylabel('Energy Change (J)')
        ax2.grid(True)
        ax2.set_title('Energy Conservation (ΔE)')
        
        # Power (rate of energy change)
        ax3 = self.sim_fig.add_subplot(223)
        dt = np.diff(solution.t)
        power = energy_change / dt
        ax3.plot(solution.t[1:], power, 'm-')
        ax3.set_ylabel('Power (W)')
        ax3.set_xlabel('Time (s)')
        ax3.grid(True)
        ax3.set_title('Power vs Time')
        
        # Energy efficiency (work done vs energy input)
        ax4 = self.sim_fig.add_subplot(224)
        # Calculate work done by control forces
        Fx = mass * np.gradient(vx, solution.t)
        Fz = mass * np.gradient(vz, solution.t)
        work_x = np.cumsum(Fx * vx) * np.mean(np.diff(solution.t))
        work_z = np.cumsum(Fz * vz) * np.mean(np.diff(solution.t))
        total_work = work_x + work_z
        
        efficiency = np.abs(total_work) / (TE + 1e-6)  # Avoid division by zero
        ax4.plot(solution.t, efficiency, 'c-')
        ax4.set_ylabel('Efficiency')
        ax4.set_xlabel('Time (s)')
        ax4.grid(True)
        ax4.set_title('Control Efficiency')
        
        self.sim_fig.tight_layout()
        self.sim_canvas.draw()
    
    def plot_control_analysis(self, solution):
        """Control inputs and ballast fill analysis"""
        self.sim_fig.clear()
        
        # Extract state variables
        time = solution.t
        ballast_mass = solution.y[13, :]  # Ballast mass over time
        piston_position = solution.y[14, :]  # Piston position over time
        piston_velocity = solution.y[15, :]  # Piston velocity over time
        
        # Calculate ballast fill percentage
        try:
            ballast_radius = float(self.param_fields['ballast_radius'].text())
            ballast_length = float(self.param_fields['ballast_length'].text())
            rho_water = float(self.param_fields['rho_water'].text())
            
            # Maximum ballast volume and mass
            max_ballast_volume = np.pi * ballast_radius**2 * ballast_length
            max_ballast_mass = rho_water * max_ballast_volume
            
            # Fill percentage over time
            fill_percentage = (ballast_mass / max_ballast_mass) * 100
        except:
            # Fallback if parameters not available
            fill_percentage = np.zeros_like(time)
        
        # Plot 1: Ballast Fill Percentage vs Time
        ax1 = self.sim_fig.add_subplot(221)
        ax1.plot(time, fill_percentage, 'b-', linewidth=2, label='Actual Fill %')
        ax1.axhline(y=50, color='r', linestyle='--', alpha=0.7, label='50% (Neutral)')
        ax1.set_ylabel('Ballast Fill (%)')
        ax1.set_xlabel('Time (s)')
        ax1.grid(True)
        ax1.legend()
        ax1.set_title('Ballast Fill Percentage vs Time')
        
        # Plot 2: Moving Mass (Piston) Position vs Time
        ax2 = self.sim_fig.add_subplot(222)
        ax2.plot(time, piston_position, 'g-', linewidth=2, label='Piston Position')
        
        # Show piston travel limits if available
        try:
            piston_travel = float(self.param_fields['piston_travel'].text())
            if hasattr(self, 'piston_set_pos') and self.piston_set_pos is not None:
                min_pos = self.piston_set_pos
                max_pos = self.piston_set_pos + piston_travel
                ax2.axhline(y=min_pos, color='r', linestyle='--', alpha=0.7, label='Min Position')
                ax2.axhline(y=max_pos, color='r', linestyle='--', alpha=0.7, label='Max Position')
        except:
            pass
            
        ax2.set_ylabel('Piston Position (m)')
        ax2.set_xlabel('Time (s)')
        ax2.grid(True)
        ax2.legend()
        ax2.set_title('Moving Mass Position vs Time')
        
        # Plot 3: Piston Velocity vs Time
        ax3 = self.sim_fig.add_subplot(223)
        ax3.plot(time, piston_velocity, 'm-', linewidth=2, label='Piston Velocity')
        ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax3.set_ylabel('Piston Velocity (m/s)')
        ax3.set_xlabel('Time (s)')
        ax3.grid(True)
        ax3.legend()
        ax3.set_title('Moving Mass Velocity vs Time')
        
        # Plot 4: Control System Response Analysis
        ax4 = self.sim_fig.add_subplot(224)
        
        # Show control targets and performance
        if hasattr(self, 'control_params'):
            try:
                desired_depth = self.control_params['desired_depth']
                actual_depth = solution.y[2, :]  # Z position (depth)
                depth_error = actual_depth - desired_depth
                
                # Plot depth error
                ax4.plot(time, depth_error, 'c-', linewidth=2, label='Depth Error')
                ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3, label='Target (0)')
                
                # Show control system info
                control_system = self.control_params['control_system']
                ax4.set_title(f'{control_system}\nTarget Depth: {desired_depth}m')
                ax4.set_ylabel('Depth Error (m)')
                ax4.set_xlabel('Time (s)')
                ax4.grid(True)
                
                # Add control performance metrics
                max_error = np.max(np.abs(depth_error))
                final_error = np.abs(depth_error[-1])
                ax4.text(0.02, 0.98, f'Max Error: {max_error:.2f}m\nFinal Error: {final_error:.2f}m', 
                        transform=ax4.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
            except Exception as e:
                # Fallback plot
                ax4.plot(time, ballast_mass, 'c-', linewidth=2, label='Ballast Mass')
                ax4.set_ylabel('Ballast Mass (kg)')
                ax4.set_xlabel('Time (s)')
                ax4.grid(True)
                ax4.set_title('Ballast Mass vs Time')
        else:
            # Fallback plot
            ax4.plot(time, ballast_mass, 'c-', linewidth=2, label='Ballast Mass')
            ax4.set_ylabel('Ballast Mass (kg)')
            ax4.set_xlabel('Time (s)')
            ax4.grid(True)
            ax4.set_title('Ballast Mass vs Time')
        
        ax4.legend()
        
        self.sim_fig.tight_layout()
        self.sim_canvas.draw()
    
    def plot_all_diagnostics(self, solution):
        """Show all diagnostic plots in a comprehensive view"""
        self.sim_fig.clear()
        
        # Create a 3x3 grid of plots
        # Row 1: Basic plots
        ax1 = self.sim_fig.add_subplot(3, 3, 1)
        ax1.plot(solution.t, solution.y[2, :])
        ax1.set_ylabel("Depth (m)")
        ax1.invert_yaxis()
        ax1.grid(True)
        ax1.set_title("Depth")
        
        ax2 = self.sim_fig.add_subplot(3, 3, 2)
        pitch_angles = []
        for i in range(len(solution.t)):
            quat = solution.y[3:7, i]
            pitch = R.from_quat(quat).as_euler('zyx')[1]
            pitch_angles.append(np.degrees(pitch))
        ax2.plot(solution.t, pitch_angles)
        ax2.set_ylabel("Pitch (deg)")
        ax2.grid(True)
        ax2.set_title("Pitch")
        
        ax3 = self.sim_fig.add_subplot(3, 3, 3)
        vx = np.gradient(solution.y[0, :], solution.t)
        vy = np.gradient(solution.y[1, :], solution.t)
        vz = np.gradient(solution.y[2, :], solution.t)
        speed = np.sqrt(vx**2 + vy**2 + vz**2)
        ax3.plot(solution.t, speed)
        ax3.set_ylabel("Speed (m/s)")
        ax3.grid(True)
        ax3.set_title("Speed")
        
        # Row 2: Velocity analysis
        ax4 = self.sim_fig.add_subplot(3, 3, 4)
        ax4.plot(solution.t, vx, 'r-', label='Vx')
        ax4.plot(solution.t, vy, 'g-', label='Vy')
        ax4.plot(solution.t, vz, 'b-', label='Vz')
        ax4.set_ylabel('Velocity (m/s)')
        ax4.grid(True)
        ax4.legend(fontsize=8)
        ax4.set_title("Velocity Components")
        
        ax5 = self.sim_fig.add_subplot(3, 3, 5)
        ax5.scatter(vx, vz, c=solution.t, cmap='viridis', alpha=0.7, s=10)
        ax5.set_xlabel('Vx (m/s)')
        ax5.set_ylabel('Vz (m/s)')
        ax5.grid(True)
        ax5.set_title("Vx vs Vz")
        
        ax6 = self.sim_fig.add_subplot(3, 3, 6)
        # 3D trajectory projection (X vs Z)
        ax6.plot(solution.y[0, :], solution.y[2, :], 'b-')
        ax6.set_xlabel('X (m)')
        ax6.set_ylabel('Depth (m)')
        ax6.invert_yaxis()
        ax6.grid(True)
        ax6.set_title("X-Z Trajectory")
        
        # Row 3: Control and energy
        ax7 = self.sim_fig.add_subplot(3, 3, 7)
        mass = 50.0
        ax = np.gradient(vx, solution.t)
        az = np.gradient(vz, solution.t)
        Fx = mass * ax
        Fz = mass * az
        ax7.plot(solution.t, Fx, 'r-', label='Fx')
        ax7.plot(solution.t, Fz, 'b-', label='Fz')
        ax7.set_ylabel('Force (N)')
        ax7.set_xlabel('Time (s)')
        ax7.grid(True)
        ax7.legend(fontsize=8)
        ax7.set_title("Control Forces")
        
        ax8 = self.sim_fig.add_subplot(3, 3, 8)
        g = 9.81
        depth = solution.y[2, :]
        PE = mass * g * depth
        KE = 0.5 * mass * speed**2
        TE = PE + KE
        ax8.plot(solution.t, PE, 'b-', label='PE')
        ax8.plot(solution.t, KE, 'r-', label='KE')
        ax8.plot(solution.t, TE, 'k-', label='Total')
        ax8.set_ylabel('Energy (J)')
        ax8.set_xlabel('Time (s)')
        ax8.grid(True)
        ax8.legend(fontsize=8)
        ax8.set_title("Energy")
        
        ax9 = self.sim_fig.add_subplot(3, 3, 9)
        # Control effort
        control_effort = np.cumsum(np.abs(Fx) + np.abs(Fz)) * np.mean(np.diff(solution.t))
        ax9.plot(solution.t, control_effort, 'g-')
        ax9.set_ylabel('Control Effort (N·s)')
        ax9.set_xlabel('Time (s)')
        ax9.grid(True)
        ax9.set_title("Control Effort")
        
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

    def update_depth_table(self):
        """Update the depth calculation table"""
        try:
            # Get input values
            E = self.table_E_input.value()
            nu = self.table_nu_input.value()
            R = self.table_R_input.value()
            t = self.table_t_input.value()
            rho = self.table_rho_input.value()
            g = self.table_g_input.value()
            L = float(self.param_fields['cyl_length'].text())  # Get hull length from design tab

            # Get disk positions
            disk_text = self.table_disk_positions.text().strip()
            if disk_text:
                try:
                    disk_positions = [float(x.strip()) for x in disk_text.split(',')]
                except ValueError:
                    # Fall back to number of internal disks
                    n_internal = self.table_n_disks.value()
                    if n_internal > 0:
                        step = L / (n_internal + 1)
                        disk_positions = [step * (i + 1) for i in range(n_internal)]
                    else:
                        disk_positions = []
            else:
                n_internal = self.table_n_disks.value()
                if n_internal > 0:
                    step = L / (n_internal + 1)
                    disk_positions = [step * (i + 1) for i in range(n_internal)]
                else:
                    disk_positions = []

            # Add end disks at 0 and L
            disk_positions = [0] + sorted(disk_positions) + [L]

            # Calculate bay properties
            self.depth_table.setRowCount(len(disk_positions) - 1)
            for i in range(len(disk_positions) - 1):
                start_pos = disk_positions[i]
                end_pos = disk_positions[i + 1]
                bay_length = end_pos - start_pos

                # Calculate safe depth for this bay
                from glider_depth import (Material, Geometry, Env, Factors, Coefficients, 
                                       evaluate_explicit)
                
                # Create objects for calculation
                sigma_y = self.table_sigma_y_input.value()
                material = Material(E=E, nu=nu, sigma_y=sigma_y)
                geometry = Geometry(R=R, t=t, L=L)
                env = Env(rho=rho, g=g)
                factors = Factors(
                    eta_end=self.table_eta_end.value(),
                    kdf_overall=self.table_kdf_overall.value(),
                    kdf_interframe=self.table_kdf_interframe.value(),
                    phi_yield=self.table_phi_yield.value(),
                    gamma_global=self.table_gamma_global.value()
                )
                coeffs = Coefficients(
                    K_if=self.table_K_if.value(),
                    C_overall=self.table_C_overall.value()
                )
                
                # Get n value
                n_waves_text = self.table_n_waves.currentText()
                n_waves = None if n_waves_text == "Auto" else int(n_waves_text)

                # Calculate depths
                results = evaluate_explicit(material, geometry, disk_positions, env, factors, coeffs, n_waves)
                safe_depth = results['h_safe_min']

                # Update summary
                n_info = (
                    f"Optimal n = {results['optimal_n']}"
                    if n_waves is None else
                    f"Using n = {results['n_used']} (manual)"
                )

                # Calculate bay details for debugging
                De = 2 * R
                bay_details = []
                for bay_idx, bay_length in enumerate(results['bay_lengths']):
                    ratio = (2 * (n_waves or 4) * bay_length) / (math.pi * De)
                    ratio_sq = ratio ** 2
                    
                    # Calculate pressures for this bay
                    p_membrane = (2 * E / ((n_waves or 4)**2 - 1) * (1 + ratio_sq) ** 2) * (t/De)
                    p_bending = (2 * E / (3 * (1 - nu**2))) * ((n_waves or 4)**2 - 1 - 
                               ((2 * (n_waves or 4)**2 - 1 - nu) / (1 - ratio_sq)) * 
                               (t/De) ** 3)
                    p_total = p_membrane + p_bending
                    
                    # Convert to different units
                    p_membrane_mpa = p_membrane / 1e6
                    p_membrane_gpa = p_membrane / 1e9
                    p_membrane_atm = p_membrane / 101325  # 1 atm = 101325 Pa
                    
                    p_bending_mpa = p_bending / 1e6
                    p_bending_gpa = p_bending / 1e9
                    p_bending_atm = p_bending / 101325
                    
                    p_total_mpa = p_total / 1e6
                    p_total_gpa = p_total / 1e9
                    p_total_atm = p_total / 101325
                    
                    bay_details.append(
                        f"Bay {bay_idx+1}:<br/>"
                        f"• Length: {bay_length:.3f} m<br/>"
                        f"• (2nL/πDe)² = {ratio_sq:.3f}<br/>"
                        f"• Membrane pressure: {p_membrane_mpa:.1f} MPa / {p_membrane_gpa:.3f} GPa / {p_membrane_atm:.1f} atm<br/>"
                        f"• Bending pressure: {p_bending_mpa:.1f} MPa / {p_bending_gpa:.3f} GPa / {p_bending_atm:.1f} atm<br/>"
                        f"• Total pressure: {p_total_mpa:.1f} MPa / {p_total_gpa:.3f} GPa / {p_total_atm:.1f} atm"
                    )

                # Convert depths to different units
                h_interframe_ft = results['h_safe_interframe'] * 3.28084  # m to ft
                h_overall_ft = results['h_safe_overall'] * 3.28084
                h_yield_ft = results['h_safe_yielding'] * 3.28084
                h_min_ft = results['h_safe_min'] * 3.28084
                
                # Calculate safety factor
                desired_depth = self.table_desired_depth.value()
                safety_factor = results['h_safe_min'] / desired_depth if desired_depth > 0 else float('inf')
                
                # Safety factor color coding
                if safety_factor >= 2.0:
                    safety_color = "green"
                    safety_status = "Excellent"
                elif safety_factor >= 1.5:
                    safety_color = "orange"
                    safety_status = "Good"
                elif safety_factor >= 1.0:
                    safety_color = "red"
                    safety_status = "Marginal"
                else:
                    safety_color = "darkred"
                    safety_status = "Unsafe"
                
                summary = f"""
                <h3>Depth Analysis Results</h3>
                <p><b>Safe Depths:</b></p>
                <ul>
                    <li>Interframe (Bay) Buckling: {results['h_safe_interframe']:.1f} m / {h_interframe_ft:.1f} ft</li>
                    <li>Overall (Global) Buckling: {results['h_safe_overall']:.1f} m / {h_overall_ft:.1f} ft</li>
                    <li>Yielding: {results['h_safe_yielding']:.1f} m / {h_yield_ft:.1f} ft</li>
                </ul>
                <p><b>Controlling Mode:</b> {results['mode_controlling']}</p>
                <p><b>Minimum Safe Depth:</b> <span style='color: red; font-weight: bold;'>{results['h_safe_min']:.1f} m / {h_min_ft:.1f} ft</span></p>
                <p><b>Effective Diameter (De):</b> {2*R:.3f} m / {(2*R)*3.28084:.3f} ft</p>
                <p><b>Circumferential Waves:</b> {n_info}</p>
                <p><b>Safety Analysis:</b></p>
                <ul>
                    <li>Desired Operating Depth: {desired_depth:.1f} m / {desired_depth*3.28084:.1f} ft</li>
                    <li>Safety Factor: <span style='color: {safety_color}; font-weight: bold;'>{safety_factor:.2f} ({safety_status})</span></li>
                </ul>
                <p><b>Bay Analysis:</b></p>
                <ul>
                    {"".join(f"<li>{detail}</li>" for detail in bay_details)}
                </ul>
                <p><i>Note: High values of (2nL/πDe)² may give unrealistic results</i></p>
                <p><i>Safety Factor: ≥2.0 (Excellent), ≥1.5 (Good), ≥1.0 (Marginal), <1.0 (Unsafe)</i></p>
                """
                self.depth_summary.setText(summary)

                # Update table for this bay
                bay_idx = i  # Current bay index
                self.depth_table.setItem(bay_idx, 0, QTableWidgetItem(str(bay_idx+1)))
                self.depth_table.setItem(bay_idx, 1, QTableWidgetItem(f"{start_pos:.3f}"))
                self.depth_table.setItem(bay_idx, 2, QTableWidgetItem(f"{end_pos:.3f}"))
                self.depth_table.setItem(bay_idx, 3, QTableWidgetItem(f"{bay_length:.3f}"))
                self.depth_table.setItem(bay_idx, 4, QTableWidgetItem(f"{safe_depth:.1f}"))

        except Exception as e:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Calculation Error", str(e))

    def sync_table_inputs(self):
        """Sync inputs from the design tab"""
        try:
            # Sync disk positions from design tab
            if 'piston_position' in self.param_fields and 'ballast_position' in self.param_fields:
                piston_pos = [float(x.strip()) for x in self.param_fields['piston_position'].text().split(',')][0]
                ballast_pos = [float(x.strip()) for x in self.param_fields['ballast_position'].text().split(',')][0]
                self.table_disk_positions.setText(f"{piston_pos}, {ballast_pos}")

            # Sync other parameters
            if 'hull_radius' in self.param_fields:
                self.table_R_input.setValue(float(self.param_fields['hull_radius'].text()))
            if 'hull_thickness' in self.param_fields:
                self.table_t_input.setValue(float(self.param_fields['hull_thickness'].text()))
            if 'rho_water' in self.param_fields:
                self.table_rho_input.setValue(float(self.param_fields['rho_water'].text()))
            # Note: Young's modulus and yield strength are typically material properties that may not be in the design tab

            # Update the table
            self.update_depth_table()

        except Exception as e:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Sync Error", str(e))