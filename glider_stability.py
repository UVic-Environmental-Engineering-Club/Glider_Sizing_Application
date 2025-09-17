"""
Meta Center Analysis and Stability Calculations for Underwater Glider
This module provides comprehensive stability analysis using the existing physics calculations
from glider_physics.py, including meta center height, righting arms, and stability criteria.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, List
from dataclasses import dataclass
import math
from glider_physics import UnderwaterGlider
from datetime import datetime


@dataclass
class StabilityResult:
    """Container for stability analysis results"""
    meta_center_height: float  # GM - distance from CG to meta center
    center_of_gravity: Tuple[float, float, float]  # (x, y, z) in meters
    center_of_buoyancy: Tuple[float, float, float]  # (x, y, z) in meters
    meta_center: Tuple[float, float, float]  # (x, y, z) in meters
    is_stable: bool  # True if GM > 0
    stability_margin: float  # GM as percentage of beam
    righting_arm_max: float  # Maximum righting arm
    heel_angle_max: float  # Angle at which righting arm becomes negative
    ballast_stability_curve: List[Tuple[float, float]]  # (fill_ratio, GM) pairs


class GliderStability:
    """Class for calculating underwater glider stability parameters using existing physics"""
    
    def __init__(self, glider_params: Dict[str, Any]):
        # Create a glider instance to use its physics calculations
        self.glider = UnderwaterGlider(glider_params)
        self.params = glider_params
        
        # Extract key parameters for stability calculations
        self.hull_radius = float(glider_params.get('hull_radius', 0.08))
        self.hull_thickness = float(glider_params.get('hull_thickness', 0.002))
        self.rho_water = float(glider_params.get('rho_water', 1025.0))
        self.g = float(glider_params.get('g', 9.81))
        
    def _update_glider_state(self, ballast_fill: float = None):
        """Update the glider's internal state for calculations"""
        if ballast_fill is not None:
            # Update ballast fill fraction
            self.glider.fill_fraction = ballast_fill
            
        # Recalculate mass properties and buoyancy
        self.glider._calculate_mass_properties()
        self.glider._estimate_hull_volume_and_cb()
        self.glider._calculate_inertia()
        
    def get_center_of_gravity(self) -> Tuple[float, float, float]:
        """Get center of gravity from physics calculations"""
        self._update_glider_state()
        cg = self.glider.cg
        return (float(cg[0]), float(cg[1]), float(cg[2]))
    
    def get_center_of_buoyancy(self) -> Tuple[float, float, float]:
        """Get center of buoyancy from physics calculations"""
        self._update_glider_state()
        cb = self.glider.cb
        return (float(cb[0]), float(cb[1]), float(cb[2]))
    
    def get_mass(self) -> float:
        """Get total mass from physics calculations"""
        self._update_glider_state()
        return self.glider.mass
    
    def get_volume(self) -> float:
        """Get hull volume from physics calculations"""
        self._update_glider_state()
        return self.glider.V_hull

    def get_inertia(self) -> float:
        """Get second moment of inertia of the waterplane"""
        self._update_glider_state()
        r_o = self.hull_radius
        r_i = self.hull_radius - self.hull_thickness
        # Second moment of inertia of a thin-walled circular tube
        I = np.pi * (r_o**4 - r_i**4) / 4
        return I
    
    def calculate_meta_center(self) -> Tuple[float, float, float]:
        """Calculate meta center position using waterplane second moment of inertia"""
        cb = self.get_center_of_buoyancy()
        
        hull_inertia = self.get_inertia()
        
        # Meta center is above CB by I/V where I is moment of inertia
        volume = self.get_volume()
        meta_height = hull_inertia / volume if volume > 0 else 0
        
        meta_center = (cb[0], cb[1], cb[2] + meta_height)
        
        return meta_center
    
    def calculate_meta_center_height(self) -> float:
        """Calculate meta center height (GM) - distance from CG to meta center"""
        cg = self.get_center_of_gravity()
        meta_center = self.calculate_meta_center()
        
        # GM is the vertical distance from CG to meta center
        gm = meta_center[2] - cg[2]
        
        return gm
    
    def calculate_righting_arm(self, heel_angle: float) -> float:
        """Calculate righting arm for a given heel angle"""
        cg = self.get_center_of_gravity()
        cb = self.get_center_of_buoyancy()
        
        # Convert angle to radians
        angle_rad = math.radians(heel_angle)
        
        # Calculate horizontal distance between CG and CB
        horizontal_distance = cb[1] - cg[1]
        
        # Righting arm = horizontal_distance * cos(angle) + GM * sin(angle)
        righting_arm = horizontal_distance * math.cos(angle_rad) + self.calculate_meta_center_height() * math.sin(angle_rad)
        
        return righting_arm
    
    def analyze_stability(self) -> StabilityResult:
        """Perform comprehensive stability analysis using physics calculations"""
        gm = self.calculate_meta_center_height()
        cg = self.get_center_of_gravity()
        cb = self.get_center_of_buoyancy()
        meta_center = self.calculate_meta_center()
        
        # Stability criteria: GM > 0 for stability
        is_stable = gm > 0
        
        # Stability margin as percentage of beam (2 * hull_radius)
        beam = 2 * self.hull_radius
        stability_margin = (gm / beam) * 100 if beam > 0 else 0
        
        # Calculate righting arm curve
        angles = np.linspace(0, 90, 19)  # 0 to 90 degrees
        righting_arms = [self.calculate_righting_arm(angle) for angle in angles]
        
        # Find maximum righting arm and angle where it becomes negative
        righting_arm_max = max(righting_arms)
        max_angle_idx = np.argmax(righting_arms)
        heel_angle_max = angles[max_angle_idx]
        
        # Find angle where righting arm becomes negative (if any)
        negative_angles = [angle for angle, ra in zip(angles, righting_arms) if ra < 0]
        heel_angle_max_stable = max(negative_angles) if negative_angles else 90.0
        
        # Ballast stability curve
        ballast_curve = []
        for fill_ratio in np.linspace(0, 1, 11):
            self._update_glider_state(fill_ratio)
            gm_at_fill = self.calculate_meta_center_height()
            ballast_curve.append((fill_ratio, gm_at_fill))
        
        # Restore original ballast fill
        original_fill = float(self.params.get('current_fill', 0.5))
        self._update_glider_state(original_fill)
        
        return StabilityResult(
            meta_center_height=gm,
            center_of_gravity=cg,
            center_of_buoyancy=cb,
            meta_center=meta_center,
            is_stable=is_stable,
            stability_margin=stability_margin,
            righting_arm_max=righting_arm_max,
            heel_angle_max=heel_angle_max_stable,
            ballast_stability_curve=ballast_curve
        )
    
    def get_stability_summary(self) -> str:
        """Get a human-readable stability summary"""
        result = self.analyze_stability()
        
        summary = f"""
        STABILITY ANALYSIS SUMMARY
        ==========================

        Meta Center Height (GM): {result.meta_center_height:.4f} m
        Stability Status: {'STABLE' if result.is_stable else 'UNSTABLE'}
        Stability Margin: {result.stability_margin:.1f}% of beam

        Center of Gravity: ({result.center_of_gravity[0]:.3f}, {result.center_of_gravity[1]:.3f}, {result.center_of_gravity[2]:.3f}) m
        Center of Buoyancy: ({result.center_of_buoyancy[0]:.3f}, {result.center_of_buoyancy[1]:.3f}, {result.center_of_buoyancy[2]:.3f}) m
        Meta Center: ({result.meta_center[0]:.3f}, {result.meta_center[1]:.3f}, {result.meta_center[2]:.3f}) m

        Righting Arm Maximum: {result.righting_arm_max:.4f} m
        Maximum Stable Heel Angle: {result.heel_angle_max:.1f}°

        Ballast Tank Effect: {'Increases' if result.ballast_stability_curve[-1][1] > result.ballast_stability_curve[0][1] else 'Decreases'} stability when filled

        Physics Data:
        - Total Mass: {self.get_mass():.3f} kg
        - Hull Volume: {self.get_volume():.6f} m³
        - Water Density: {self.rho_water:.1f} kg/m³
        """
        
        return summary.strip()
    
    def plot_stability_diagram(self, ax=None):
        """Plot stability diagram showing CG, CB, and meta center in Y-Z plane (transverse vs vertical)"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(16, 8)) 
        
        result = self.analyze_stability()
        
        # Get geometry from physics for reference
        hull_radius = self.glider.hull_radius
        
        # Create main plot area 
        main_ax = ax
        
        # Plot glider outline in Y-Z plane (transverse cross-section)
        # Show the circular cross-section at the CG location
        cg_x = result.center_of_gravity[0]
        
        # Create a circle representing the hull cross-section at CG
        theta = np.linspace(0, 2*np.pi, 100)
        hull_y = hull_radius * np.cos(theta)
        hull_z = hull_radius * np.sin(theta)
        
        # Plot hull cross-section
        main_ax.plot(hull_y, hull_z, 'k-', linewidth=2, label='Hull Cross-Section')
        
        # Plot CG, CB, and Meta Center in Y-Z plane
        cg_y, cg_z = result.center_of_gravity[1], result.center_of_gravity[2]
        cb_y, cb_z = result.center_of_buoyancy[1], result.center_of_buoyancy[2]
        meta_y, meta_z = result.meta_center[1], result.meta_center[2]
        
        # Plot the key points with larger markers
        main_ax.plot(cg_y, cg_z, 'ro', markersize=8, label=f'CG ({cg_y:.3f}, {cg_z:.3f})')
        main_ax.plot(cb_y, cb_z, 'bo', markersize=8, label=f'CB ({cb_y:.3f}, {cb_z:.3f})')
        main_ax.plot(meta_y, meta_z, 'go', markersize=8, label=f'Meta Center ({meta_y:.3f}, {meta_z:.3f})')
        
        # Draw GM line (vertical distance from CG to meta center)
        main_ax.plot([cg_y, meta_y], [cg_z, meta_z], 'g--', linewidth=3, label=f'GM = {result.meta_center_height:.4f} m')
        
        # Draw horizontal line from CG to CB for reference
        main_ax.plot([cg_y, cb_y], [cg_z, cb_z], 'b:', linewidth=2, alpha=0.7, label='CG to CB')
        
        # Add stability indicator
        stability_color = 'green' if result.is_stable else 'red'
        stability_text = 'STABLE' if result.is_stable else 'UNSTABLE'
        ax.text(-0.75, 1, f'STABILITY: {stability_text}', 
                transform=ax.transAxes, ha='left', va='top', 
                fontsize=16, fontweight='bold', color=stability_color,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # Add GM value
        ax.text(-0.75, 0.90, f'GM = {result.meta_center_height:.4f} m', 
                transform=ax.transAxes, ha='left', va='top', 
                fontsize=14, fontweight='bold', color='green',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        # Add "Changes Applied" indicator
        ax.text(-0.75, 0.8, 'CHANGES APPLIED', 
                transform=ax.transAxes, ha='left', va='top', 
                fontsize=12, fontweight='bold', color='darkgreen',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))
        
        # Add timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        ax.text(-0.75, 0.70, f'Updated: {timestamp}', 
                transform=ax.transAxes, ha='left', va='top', 
                fontsize=10, color='darkblue',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Set proper axis labels and title
        ax.set_xlabel('Transverse Position (Y) [m]', fontsize=12)
        ax.set_ylabel('Vertical Position (Z) [m]', fontsize=12)
        ax.set_title('Glider Stability Diagram (Y-Z Plane)', fontsize=14, fontweight='bold')
        
        # Set axis limits to show the important features clearly
        # Center the view around the hull and expand to show CG, CB, meta center
        margin = max(hull_radius, abs(result.meta_center_height)) * 1.25
        ax.set_xlim(-margin, margin)
        ax.set_ylim(-margin, margin)
        
        # Add grid and legend
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=10)
        
        # Make axes equal aspect ratio for proper circular representation
        ax.set_aspect('equal')
        
        # Add reference lines for zero positions
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
        
        return ax
    
    def plot_righting_arm_curve(self, ax=None):
        """Plot righting arm curve with improved scaling and information"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        result = self.analyze_stability()
        
        # Generate righting arm curve with more points for smooth plotting
        angles = np.linspace(0, 90, 181)  # 0 to 90 degrees, 0.5 degree increments
        righting_arms = [self.calculate_righting_arm(angle) for angle in angles]
        
        # Plot the main righting arm curve
        ax.plot(angles, righting_arms, 'b-', linewidth=3, label='Righting Arm')
        
        # Add reference lines
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.7, linewidth=2, label='Neutral Stability')
        
        # Find and mark the maximum righting arm
        max_ra_idx = np.argmax(righting_arms)
        max_angle = angles[max_ra_idx]
        max_ra = righting_arms[max_ra_idx]
        
        # Mark the maximum point
        ax.plot(max_angle, max_ra, 'ro', markersize=10, label=f'Max RA: {max_ra:.4f} m @ {max_angle:.1f}°')
        
        # Mark the maximum stable angle
        ax.axvline(x=result.heel_angle_max, color='r', linestyle='--', alpha=0.8, linewidth=2,
                   label=f'Max Stable Angle: {result.heel_angle_max:.1f}°')
        
        # Add stability zones
        ax.fill_between(angles, 0, righting_arms, where=np.array(righting_arms) > 0, 
                       alpha=0.2, color='green', label='Stable Zone')
        ax.fill_between(angles, righting_arms, 0, where=np.array(righting_arms) < 0, 
                       alpha=0.2, color='red', label='Unstable Zone')
        
        # Add GM value prominently
        ax.text(0.02, 0.98, f'GM = {result.meta_center_height:.4f} m', 
                transform=ax.transAxes, ha='left', va='top', 
                fontsize=14, fontweight='bold', color='green',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        # Add stability status
        stability_color = 'green' if result.is_stable else 'red'
        stability_text = 'STABLE' if result.is_stable else 'UNSTABLE'
        ax.text(0.02, 0.92, f'Status: {stability_text}', 
                transform=ax.transAxes, ha='left', va='top', 
                fontsize=12, fontweight='bold', color=stability_color,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # Set axis labels and title
        ax.set_xlabel('Heel Angle [degrees]', fontsize=12)
        ax.set_ylabel('Righting Arm [m]', fontsize=12)
        ax.set_title('Righting Arm Curve (Roll Stability)', fontsize=14, fontweight='bold')
        
        # Set axis limits for better visibility with improved scaling
        ra_range = max(righting_arms) - min(righting_arms)
        margin = ra_range * 0.15 if ra_range > 0 else 0.01  
        ax.set_ylim(min(righting_arms) - margin, max(righting_arms) + margin)
        ax.set_xlim(0, 90)
        
        # Add grid and legend
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)
        
        # Add angle markers for common angles
        for angle in [15, 30, 45, 60, 75]:
            ax.axvline(x=angle, color='gray', linestyle=':', alpha=0.5, linewidth=1)
            ax.text(angle, ax.get_ylim()[1] * 0.95, f'{angle}°', 
                   ha='center', va='top', fontsize=8, color='gray')
        
        return ax
    
    def plot_ballast_stability_curve(self, ax=None):
        """Plot how ballast tank filling affects stability with improved scaling"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        result = self.analyze_stability()
        
        fill_ratios, gm_values = zip(*result.ballast_stability_curve)
        
        # Plot the main curve
        ax.plot(fill_ratios, gm_values, 'b-o', linewidth=3, markersize=8, label='GM vs Ballast Fill')
        
        # Add reference lines
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.8, linewidth=2, label='Neutral Stability')
        ax.axhline(y=result.meta_center_height, color='g', linestyle='--', alpha=0.8, linewidth=2, 
                   label=f'Current GM: {result.meta_center_height:.4f} m')
        
        # Add stability zones
        ax.fill_between(fill_ratios, 0, gm_values, where=np.array(gm_values) > 0, 
                       alpha=0.2, color='green', label='Stable Zone')
        ax.fill_between(fill_ratios, gm_values, 0, where=np.array(gm_values) < 0, 
                       alpha=0.2, color='red', label='Unstable Zone')
        
        # Mark the current ballast fill level
        current_fill = float(self.params.get('current_fill', 0.5))
        current_gm = result.meta_center_height
        ax.plot(current_fill, current_gm, 'ro', markersize=12, 
                label=f'Current: {current_fill:.2f} fill, GM={current_gm:.4f} m')
        
        # Add GM value prominently
        ax.text(0.02, 0.98, f'Current GM = {result.meta_center_height:.4f} m', 
                transform=ax.transAxes, ha='left', va='top', 
                fontsize=14, fontweight='bold', color='green',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        # Add stability status
        stability_color = 'green' if result.is_stable else 'red'
        stability_text = 'STABLE' if result.is_stable else 'UNSTABLE'
        ax.text(0.02, 0.92, f'Status: {stability_text}', 
                transform=ax.transAxes, ha='left', va='top', 
                fontsize=12, fontweight='bold', color=stability_color,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # Calculate and display ballast effect
        gm_empty = gm_values[0]
        gm_full = gm_values[-1]
        ballast_effect = gm_full - gm_empty
        effect_text = f"Ballast Effect: {ballast_effect:+.4f} m"
        ax.text(0.02, 0.86, effect_text, 
                transform=ax.transAxes, ha='left', va='top', 
                fontsize=11, fontweight='bold', 
                color='blue' if ballast_effect > 0 else 'red',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # Set axis labels and title
        ax.set_xlabel('Ballast Tank Fill Ratio', fontsize=12)
        ax.set_ylabel('Meta Center Height (GM) [m]', fontsize=12)
        ax.set_title('Ballast Tank Effect on Stability', fontsize=14, fontweight='bold')
        
        # Set axis limits for better visibility with improved scaling
        gm_range = max(gm_values) - min(gm_values)
        margin = gm_range * 0.15 if gm_range > 0 else 0.01  # Increased from 0.1 to 0.15
        ax.set_ylim(min(gm_values) - margin, max(gm_values) + margin)
        ax.set_xlim(0, 1)
        
        # Add grid and legend
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)
        
        # Add fill ratio markers
        for ratio in [0.25, 0.5, 0.75]:
            ax.axvline(x=ratio, color='gray', linestyle=':', alpha=0.5, linewidth=1)
            ax.text(ratio, ax.get_ylim()[1] * 0.95, f'{ratio:.0%}', 
                   ha='center', va='top', fontsize=8, color='gray')
        
        # Add scale indicator
        scale_text = f"Scale: 1 unit = {gm_range/2:.4f} m (enhanced for visibility)"
        ax.text(0.98, 0.02, scale_text, 
                transform=ax.transAxes, ha='right', va='bottom', 
                fontsize=9, color='gray', style='italic',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        return ax


def analyze_glider_stability(glider_params: Dict[str, Any]) -> StabilityResult:
    """Convenience function to analyze glider stability"""
    analyzer = GliderStability(glider_params)
    return analyzer.analyze_stability()
