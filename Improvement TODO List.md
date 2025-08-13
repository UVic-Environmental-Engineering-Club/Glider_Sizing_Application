# Physics Improvement TODO List

## Current State Analysis

**6x6 Added-Mass Matrix**: ✅ **Already Implemented**
- The 6x6 added-mass matrix is fully implemented with both user-supplied and auto-calculated options
- Users can provide a full 6x6 matrix via `params['added_mass_matrix']`
- If not provided, it automatically constructs a diagonal matrix from scalar parameters
- The GUI currently doesn't have inputs for the 6x6 matrix, but the backend supports it

**Drag Coefficients**: ❌ **Currently Hardcoded**
- `Cd_x`, `Cd_y`, `Cd_z` are hardcoded as 0.8, 1.0, 1.0 respectively
- No table-based lookup system exists

**Import/Export**: ⚠️ **Partially Implemented**
- Basic parameter save/load exists for simulation data
- No comprehensive settings import/export for physics parameters

---

## 1. Drag Coefficient Tables 🚨 **HIGH PRIORITY**

### Research and Data Collection
- [ ] Research standard marine vehicle drag coefficient databases
- [ ] Collect experimental data for different glider hull shapes
- [ ] Identify key parameters affecting drag (Reynolds number, angle of attack, surface roughness)
- [ ] Research existing marine vehicle coefficient databases (ITTC, etc.)

### Implementation
- [ ] Create drag coefficient lookup tables
  - [ ] Angle-of-attack dependent drag coefficients
  - [ ] Tables for different glider configurations (nose shape, surface roughness, etc.)
- [ ] Replace hardcoded Cd values in `hydrodynamic_forces()` method
- [ ] Add interpolation between table values
- [ ] Implement Reynolds number calculation from velocity and characteristic length

### GUI Integration
- [ ] Add fields for Cd_x, Cd_y, Cd_z in Hydrodynamics tab
- [ ] Add option to select from predefined coefficient sets
- [ ] Add custom coefficient input capability
- [ ] Add coefficient validation and warnings

---

## 2. 6x6 Added-Mass Matrix GUI 🟡 **MEDIUM PRIORITY**

### Matrix Input Interface
- [ ] Create matrix input widget (6x6 grid of input fields)
- [ ] Add validation for matrix symmetry and positive definiteness
- [ ] Add option to import matrix from CSV/Excel files
- [ ] Add option to export current matrix
- [ ] Add matrix preview/visualization

### Matrix Visualization
- [ ] Show matrix heatmap in GUI
- [ ] Highlight coupling terms (off-diagonal elements)
- [ ] Show matrix condition number and warnings
- [ ] Add matrix properties display (eigenvalues, condition number)

### Preset Matrices
- [ ] Common glider shapes (torpedo, streamlined, etc.)
- [ ] CFD-derived matrices from literature
- [ ] User-defined matrix templates
- [ ] Matrix comparison tools

---

## 3. Enhanced Import/Export System 🟡 **MEDIUM PRIORITY**

### Comprehensive Settings Export
- [ ] Export all physics parameters to JSON
- [ ] Export 6x6 matrices to CSV/Excel
- [ ] Export parameter sets as named configurations
- [ ] Add configuration versioning
- [ ] Add configuration metadata (description, date, author)

### Import Capabilities
- [ ] Import parameter sets from JSON
- [ ] Import matrices from CSV/Excel
- [ ] Import from other simulation formats (if applicable)
- [ ] Validate imported data
- [ ] Handle import errors gracefully

### Configuration Management
- [ ] Save multiple named configurations
- [ ] Quick-switch between configurations
- [ ] Configuration comparison tools
- [ ] Configuration backup and restore
- [ ] Configuration search and filtering

---

## 4. Advanced Hydrodynamics �� **LOW PRIORITY**

### Lift Coefficient Tables
- [ ] Angle-of-attack dependent Cl tables
- [ ] Reynolds number effects on lift
- [ ] 3D effects (aspect ratio, planform shape)
- [ ] Control surface effectiveness

### Moment Coefficient Improvements
- [ ] More accurate CM_alpha relationships
- [ ] Control surface effectiveness
- [ ] Stability derivatives
- [ ] Dynamic stability analysis

### Viscous Effects
- [ ] Boundary layer modeling
- [ ] Surface roughness effects
- [ ] Transition modeling
- [ ] Turbulence effects

---

## 5. Validation and Testing �� **LOW PRIORITY**

### CFD Validation
- [ ] Compare model predictions with CFD results
- [ ] Validate added-mass matrices from CFD
- [ ] Benchmark against experimental data
- [ ] Create validation test suite

### Parameter Sensitivity Analysis
- [ ] Identify most critical parameters
- [ ] Create parameter uncertainty bounds
- [ ] Monte Carlo analysis tools
- [ ] Parameter optimization tools

---

## Implementation Priority Order

1. **Drag Coefficient Tables** - Biggest impact on simulation accuracy
2. **6x6 Matrix GUI Inputs** - Leverage existing backend implementation
3. **Comprehensive Import/Export** - Improve user experience
4. **Advanced Hydrodynamics** - Enhance model fidelity
5. **Validation and Testing** - Ensure accuracy and reliability

---

## Technical Notes

### Current Implementation Status
- 6x6 added-mass system: Complete in backend
- Drag coefficient system: Needs complete overhaul
- Import/export system: Basic functionality exists
- GUI physics inputs: Limited coverage

### Dependencies
- **pandas**: For table management and data handling
- **numpy**: For matrix operations and interpolation
- **scipy**: For advanced mathematical functions
- **matplotlib**: For matrix visualization

### File Locations
- **Physics backend**: `glider_physics.py`
- **GUI interface**: `glider_gui.py`
- **Simulation runner**: `glider_simulation.py`
- **Parameter storage**: `glider_gui_params.json`

---

## Progress Tracking

- [ ] **Phase 1**: Drag coefficient tables (Target: Week 1-2)
- [ ] **Phase 2**: 6x6 matrix GUI (Target: Week 3-4)
- [ ] **Phase 3**: Import/export system (Target: Week 5-6)
- [ ] **Phase 4**: Advanced hydrodynamics (Target: Week 7-8)
- [ ] **Phase 5**: Validation and testing (Target: Week 9-10)

---

## Notes and Ideas

- Look into existing marine vehicle simulation standards
- Investigate open-source CFD tools for validation
- Consider adding uncertainty quantification to all parameters
- Plan for future extensions (multi-body dynamics, fluid-structure interaction)

---

*Last Updated: [Current Date]*
*Status: Planning Phase*
*Next Milestone: Drag Coefficient Tables Implementation*