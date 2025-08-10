from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QComboBox, QGroupBox, QFormLayout, QSpacerItem, QSizePolicy
from PyQt5.QtCore import Qt

class UnitConverterWidget(QWidget):
    def __init__(self):
        super().__init__()
        # Main layout: center two columns horizontally and vertically
        main_layout = QHBoxLayout(self)
        main_layout.addStretch(1)
        center_widget = QWidget()
        center_layout = QHBoxLayout(center_widget)
        center_layout.setAlignment(Qt.AlignCenter)
        # Left and right columns
        left_col = QVBoxLayout()
        right_col = QVBoxLayout()
        # --- Length ---
        length_group = QGroupBox("Length Converter")
        length_layout = QFormLayout()
        self.length_input = QLineEdit()
        self.length_unit = QComboBox()
        self.length_unit.addItems(["meters (m)", "feet (ft)", "inches (in)", "centimeters (cm)", "millimeters (mm)"])
        self.length_output = QLineEdit()
        self.length_output.setReadOnly(True)
        self.length_output_unit = QComboBox()
        self.length_output_unit.addItems(["meters (m)", "feet (ft)", "inches (in)", "centimeters (cm)", "millimeters (mm)"])
        length_layout.addRow(QLabel("Value:"), self.length_input)
        length_layout.addRow(QLabel("From:"), self.length_unit)
        length_layout.addRow(QLabel("To:"), self.length_output_unit)
        length_layout.addRow(QLabel("Result:"), self.length_output)
        length_group.setLayout(length_layout)
        left_col.addWidget(length_group)
        # --- Mass ---
        mass_group = QGroupBox("Mass Converter")
        mass_layout = QFormLayout()
        self.mass_input = QLineEdit()
        self.mass_unit = QComboBox()
        self.mass_unit.addItems(["kilograms (kg)", "grams (g)", "pounds (lb)", "ounces (oz)"])
        self.mass_output = QLineEdit()
        self.mass_output.setReadOnly(True)
        self.mass_output_unit = QComboBox()
        self.mass_output_unit.addItems(["kilograms (kg)", "grams (g)", "pounds (lb)", "ounces (oz)"])
        mass_layout.addRow(QLabel("Value:"), self.mass_input)
        mass_layout.addRow(QLabel("From:"), self.mass_unit)
        mass_layout.addRow(QLabel("To:"), self.mass_output_unit)
        mass_layout.addRow(QLabel("Result:"), self.mass_output)
        mass_group.setLayout(mass_layout)
        left_col.addWidget(mass_group)
        # --- Volume ---
        volume_group = QGroupBox("Volume Converter")
        volume_layout = QFormLayout()
        self.volume_input = QLineEdit()
        self.volume_unit = QComboBox()
        self.volume_unit.addItems(["liters (L)", "milliliters (mL)", "cubic meters (m³)", "cubic feet (ft³)", "gallons (gal)"])
        self.volume_output = QLineEdit()
        self.volume_output.setReadOnly(True)
        self.volume_output_unit = QComboBox()
        self.volume_output_unit.addItems(["liters (L)", "milliliters (mL)", "cubic meters (m³)", "cubic feet (ft³)", "gallons (gal)"])
        volume_layout.addRow(QLabel("Value:"), self.volume_input)
        volume_layout.addRow(QLabel("From:"), self.volume_unit)
        volume_layout.addRow(QLabel("To:"), self.volume_output_unit)
        volume_layout.addRow(QLabel("Result:"), self.volume_output)
        volume_group.setLayout(volume_layout)
        left_col.addWidget(volume_group)
        # --- Time ---
        time_group = QGroupBox("Time Converter")
        time_layout = QFormLayout()
        self.time_input = QLineEdit()
        self.time_unit = QComboBox()
        self.time_unit.addItems(["seconds (s)", "minutes (min)", "hours (h)"])
        self.time_output = QLineEdit()
        self.time_output.setReadOnly(True)
        self.time_output_unit = QComboBox()
        self.time_output_unit.addItems(["seconds (s)", "minutes (min)", "hours (h)"])
        time_layout.addRow(QLabel("Value:"), self.time_input)
        time_layout.addRow(QLabel("From:"), self.time_unit)
        time_layout.addRow(QLabel("To:"), self.time_output_unit)
        time_layout.addRow(QLabel("Result:"), self.time_output)
        time_group.setLayout(time_layout)
        right_col.addWidget(time_group)
        # --- Force ---
        force_group = QGroupBox("Force Converter")
        force_layout = QFormLayout()
        self.force_input = QLineEdit()
        self.force_unit = QComboBox()
        self.force_unit.addItems(["newtons (N)", "pounds-force (lbf)"])
        self.force_output = QLineEdit()
        self.force_output.setReadOnly(True)
        self.force_output_unit = QComboBox()
        self.force_output_unit.addItems(["newtons (N)", "pounds-force (lbf)"])
        force_layout.addRow(QLabel("Value:"), self.force_input)
        force_layout.addRow(QLabel("From:"), self.force_unit)
        force_layout.addRow(QLabel("To:"), self.force_output_unit)
        force_layout.addRow(QLabel("Result:"), self.force_output)
        force_group.setLayout(force_layout)
        right_col.addWidget(force_group)
        # --- Pressure ---
        pressure_group = QGroupBox("Pressure Converter")
        pressure_layout = QFormLayout()
        self.pressure_input = QLineEdit()
        self.pressure_unit = QComboBox()
        self.pressure_unit.addItems(["pascals (Pa)", "bar (bar)", "psi (psi)", "atmospheres (atm)"])
        self.pressure_output = QLineEdit()
        self.pressure_output.setReadOnly(True)
        self.pressure_output_unit = QComboBox()
        self.pressure_output_unit.addItems(["pascals (Pa)", "bar (bar)", "psi (psi)", "atmospheres (atm)"])
        pressure_layout.addRow(QLabel("Value:"), self.pressure_input)
        pressure_layout.addRow(QLabel("From:"), self.pressure_unit)
        pressure_layout.addRow(QLabel("To:"), self.pressure_output_unit)
        pressure_layout.addRow(QLabel("Result:"), self.pressure_output)
        pressure_group.setLayout(pressure_layout)
        right_col.addWidget(pressure_group)
        # --- Energy ---
        energy_group = QGroupBox("Energy Converter")
        energy_layout = QFormLayout()
        self.energy_input = QLineEdit()
        self.energy_unit = QComboBox()
        self.energy_unit.addItems(["joules (J)", "calories (cal)", "watt-hours (Wh)"])
        self.energy_output = QLineEdit()
        self.energy_output.setReadOnly(True)
        self.energy_output_unit = QComboBox()
        self.energy_output_unit.addItems(["joules (J)", "calories (cal)", "watt-hours (Wh)"])
        energy_layout.addRow(QLabel("Value:"), self.energy_input)
        energy_layout.addRow(QLabel("From:"), self.energy_unit)
        energy_layout.addRow(QLabel("To:"), self.energy_output_unit)
        energy_layout.addRow(QLabel("Result:"), self.energy_output)
        energy_group.setLayout(energy_layout)
        right_col.addWidget(energy_group)
        # --- Density ---
        density_group = QGroupBox("Density Converter")
        density_layout = QFormLayout()
        self.density_input = QLineEdit()
        self.density_unit = QComboBox()
        self.density_unit.addItems(["kg/m³", "g/cm³", "lb/ft³"])
        self.density_output = QLineEdit()
        self.density_output.setReadOnly(True)
        self.density_output_unit = QComboBox()
        self.density_output_unit.addItems(["kg/m³", "g/cm³", "lb/ft³"])
        density_layout.addRow(QLabel("Value:"), self.density_input)
        density_layout.addRow(QLabel("From:"), self.density_unit)
        density_layout.addRow(QLabel("To:"), self.density_output_unit)
        density_layout.addRow(QLabel("Result:"), self.density_output)
        density_group.setLayout(density_layout)
        right_col.addWidget(density_group)
        # Add stretch to columns for vertical centering
        left_col.addStretch(1)
        right_col.addStretch(1)
        # Add columns to center layout
        center_layout.addLayout(left_col)
        center_layout.addSpacing(40)  # Space between columns
        center_layout.addLayout(right_col)
        # Add center widget to main layout
        main_layout.addWidget(center_widget)
        main_layout.addStretch(1)
        # --- Conversion logic and signal connections (unchanged) ---
        self.length_input.textChanged.connect(self.convert_length)
        self.length_unit.currentIndexChanged.connect(self.convert_length)
        self.length_output_unit.currentIndexChanged.connect(self.convert_length)
        self.mass_input.textChanged.connect(self.convert_mass)
        self.mass_unit.currentIndexChanged.connect(self.convert_mass)
        self.mass_output_unit.currentIndexChanged.connect(self.convert_mass)
        self.volume_input.textChanged.connect(self.convert_volume)
        self.volume_unit.currentIndexChanged.connect(self.convert_volume)
        self.volume_output_unit.currentIndexChanged.connect(self.convert_volume)
        self.time_input.textChanged.connect(self.convert_time)
        self.time_unit.currentIndexChanged.connect(self.convert_time)
        self.time_output_unit.currentIndexChanged.connect(self.convert_time)
        self.force_input.textChanged.connect(self.convert_force)
        self.force_unit.currentIndexChanged.connect(self.convert_force)
        self.force_output_unit.currentIndexChanged.connect(self.convert_force)
        self.pressure_input.textChanged.connect(self.convert_pressure)
        self.pressure_unit.currentIndexChanged.connect(self.convert_pressure)
        self.pressure_output_unit.currentIndexChanged.connect(self.convert_pressure)
        self.energy_input.textChanged.connect(self.convert_energy)
        self.energy_unit.currentIndexChanged.connect(self.convert_energy)
        self.energy_output_unit.currentIndexChanged.connect(self.convert_energy)
        self.density_input.textChanged.connect(self.convert_density)
        self.density_unit.currentIndexChanged.connect(self.convert_density)
        self.density_output_unit.currentIndexChanged.connect(self.convert_density)
    # --- Conversion logic ---
    def convert_length(self):
        try:
            value = float(self.length_input.text())
            from_unit = self.length_unit.currentText().split()[0]
            to_unit = self.length_output_unit.currentText().split()[0]
            meters = self._to_meters(value, from_unit)
            result = self._from_meters(meters, to_unit)
            self.length_output.setText(f"{result:.6g}")
        except Exception:
            self.length_output.setText("")
    def _to_meters(self, value, unit):
        if unit == "meters": return value
        if unit == "feet": return value * 0.3048
        if unit == "inches": return value * 0.0254
        if unit == "centimeters": return value * 0.01
        if unit == "millimeters": return value * 0.001
        return value
    def _from_meters(self, value, unit):
        if unit == "meters": return value
        if unit == "feet": return value / 0.3048
        if unit == "inches": return value / 0.0254
        if unit == "centimeters": return value / 0.01
        if unit == "millimeters": return value / 0.001
        return value
    def convert_mass(self):
        try:
            value = float(self.mass_input.text())
            from_unit = self.mass_unit.currentText().split()[0]
            to_unit = self.mass_output_unit.currentText().split()[0]
            kg = self._to_kg(value, from_unit)
            result = self._from_kg(kg, to_unit)
            self.mass_output.setText(f"{result:.6g}")
        except Exception:
            self.mass_output.setText("")
    def _to_kg(self, value, unit):
        if unit == "kilograms": return value
        if unit == "grams": return value * 0.001
        if unit == "pounds": return value * 0.45359237
        if unit == "ounces": return value * 0.0283495
        return value
    def _from_kg(self, value, unit):
        if unit == "kilograms": return value
        if unit == "grams": return value / 0.001
        if unit == "pounds": return value / 0.45359237
        if unit == "ounces": return value / 0.0283495
        return value
    def convert_volume(self):
        try:
            value = float(self.volume_input.text())
            from_unit = self.volume_unit.currentText().split()[0]
            to_unit = self.volume_output_unit.currentText().split()[0]
            liters = self._to_liters(value, from_unit)
            result = self._from_liters(liters, to_unit)
            self.volume_output.setText(f"{result:.6g}")
        except Exception:
            self.volume_output.setText("")
    def _to_liters(self, value, unit):
        if unit == "liters": return value
        if unit == "milliliters": return value * 0.001
        if unit == "cubic": return value  # will handle below
        if unit == "gallons": return value * 3.78541
        if unit == "feet": return value * 28.3168
        if unit == "meters": return value * 1000
        # Handle cubic meters/feet
        if unit == "meters": return value * 1000
        if unit == "feet": return value * 28.3168
        if unit == "m³": return value * 1000
        if unit == "ft³": return value * 28.3168
        return value
    def _from_liters(self, value, unit):
        if unit == "liters": return value
        if unit == "milliliters": return value / 0.001
        if unit == "gallons": return value / 3.78541
        if unit == "cubic": return value  # will handle below
        if unit == "meters": return value / 1000
        if unit == "feet": return value / 28.3168
        if unit == "m³": return value / 1000
        if unit == "ft³": return value / 28.3168
        return value
    def convert_time(self):
        try:
            value = float(self.time_input.text())
            from_unit = self.time_unit.currentText().split()[0]
            to_unit = self.time_output_unit.currentText().split()[0]
            seconds = self._to_seconds(value, from_unit)
            result = self._from_seconds(seconds, to_unit)
            self.time_output.setText(f"{result:.6g}")
        except Exception:
            self.time_output.setText("")
    def _to_seconds(self, value, unit):
        if unit == "seconds": return value
        if unit == "minutes": return value * 60
        if unit == "hours": return value * 3600
        return value
    def _from_seconds(self, value, unit):
        if unit == "seconds": return value
        if unit == "minutes": return value / 60
        if unit == "hours": return value / 3600
        return value
    def convert_force(self):
        try:
            value = float(self.force_input.text())
            from_unit = self.force_unit.currentText().split()[0]
            to_unit = self.force_output_unit.currentText().split()[0]
            newtons = self._to_newtons(value, from_unit)
            result = self._from_newtons(newtons, to_unit)
            self.force_output.setText(f"{result:.6g}")
        except Exception:
            self.force_output.setText("")
    def _to_newtons(self, value, unit):
        if unit == "newtons": return value
        if unit == "pounds-force": return value * 4.4482216
        return value
    def _from_newtons(self, value, unit):
        if unit == "newtons": return value
        if unit == "pounds-force": return value / 4.4482216
        return value
    def convert_pressure(self):
        try:
            value = float(self.pressure_input.text())
            from_unit = self.pressure_unit.currentText().split()[0]
            to_unit = self.pressure_output_unit.currentText().split()[0]
            pascals = self._to_pascals(value, from_unit)
            result = self._from_pascals(pascals, to_unit)
            self.pressure_output.setText(f"{result:.6g}")
        except Exception:
            self.pressure_output.setText("")
    def _to_pascals(self, value, unit):
        if unit == "pascals": return value
        if unit == "bar": return value * 1e5
        if unit == "psi": return value * 6894.757
        if unit == "atmospheres": return value * 101325
        return value
    def _from_pascals(self, value, unit):
        if unit == "pascals": return value
        if unit == "bar": return value / 1e5
        if unit == "psi": return value / 6894.757
        if unit == "atmospheres": return value / 101325
        return value
    def convert_energy(self):
        try:
            value = float(self.energy_input.text())
            from_unit = self.energy_unit.currentText().split()[0]
            to_unit = self.energy_output_unit.currentText().split()[0]
            joules = self._to_joules(value, from_unit)
            result = self._from_joules(joules, to_unit)
            self.energy_output.setText(f"{result:.6g}")
        except Exception:
            self.energy_output.setText("")
    def _to_joules(self, value, unit):
        if unit == "joules": return value
        if unit == "calories": return value * 4.184
        if unit == "watt-hours": return value * 3600
        return value
    def _from_joules(self, value, unit):
        if unit == "joules": return value
        if unit == "calories": return value / 4.184
        if unit == "watt-hours": return value / 3600
        return value
    def convert_density(self):
        try:
            value = float(self.density_input.text())
            from_unit = self.density_unit.currentText().split()[0]
            to_unit = self.density_output_unit.currentText().split()[0]
            kgm3 = self._to_kgm3(value, from_unit)
            result = self._from_kgm3(kgm3, to_unit)
            self.density_output.setText(f"{result:.6g}")
        except Exception:
            self.density_output.setText("")
    def _to_kgm3(self, value, unit):
        if unit == "kg/m³": return value
        if unit == "g/cm³": return value * 1000
        if unit == "lb/ft³": return value * 16.0185
        return value
    def _from_kgm3(self, value, unit):
        if unit == "kg/m³": return value
        if unit == "g/cm³": return value / 1000
        if unit == "lb/ft³": return value / 16.0185
        return value
