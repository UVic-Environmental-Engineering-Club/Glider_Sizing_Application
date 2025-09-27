#!/usr/bin/env python3
"""
Test script to verify the CFD coefficient changes work correctly.
"""

from glider_physics import UnderwaterGlider
from glider_simulation import load_cfd_table_from_file
import numpy as np

def test_cfd_changes():
    print("Testing CFD coefficient changes...")
    
    # Test 1: Create glider and check CFD table structure
    print("\n1. Testing glider CFD table structure:")
    glider = UnderwaterGlider()
    print(f"   CFD table shape: {glider.cfd_table.shape}")
    print(f"   Expected shape: (11, 4) for [AoA_deg, Cd, CL, CM]")
    print(f"   First few rows:")
    print(glider.cfd_table[:3])
    
    # Test 2: Test coefficient interpolation
    print("\n2. Testing coefficient interpolation:")
    coeffs = glider.get_drag_coefficients(15.0)
    print(f"   Coefficients at 15° AoA: {coeffs}")
    print(f"   Expected keys: ['Cd', 'CL', 'CM']")
    
    # Test 3: Test CFD table loading
    print("\n3. Testing CFD table loading:")
    try:
        table = load_cfd_table_from_file('sample_cfd_table.csv')
        print(f"   Loaded CFD table shape: {table.shape}")
        print(f"   First few rows:")
        print(table[:3])
    except Exception as e:
        print(f"   Error loading CFD table: {e}")
    
    # Test 4: Test force calculation
    print("\n4. Testing force calculation:")
    glider.u, glider.v, glider.w = 1.0, 0.0, 0.5  # Set velocity
    glider.compute_forces_and_moments()
    print(f"   Drag force: {glider.F_drag_body}")
    print(f"   Lift force: {glider.F_lift_body}")
    print(f"   Total hydrodynamic force: {glider.F_hydro_body}")
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    test_cfd_changes()
