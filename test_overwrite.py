#!/usr/bin/env python3
"""
Test script for the overwrite functionality
"""
import os
import sys
from ECOv003_L3T_L4T_JET import L3T_L4T_JET

def test_overwrite_functionality():
    """Test the overwrite functionality with existing files"""
    
    # Use the runconfig file from the previous successful run
    runconfig_filename = os.path.expanduser('~/data/ECOv003_example/L3T_L4T_JET_working/ECOv003_L3T_JET_35698_014_11SPS_07131111T000000_01_runconfig.xml')
    
    if not os.path.exists(runconfig_filename):
        print(f"Runconfig file not found: {runconfig_filename}")
        print("Please run the example script first to generate the runconfig file.")
        return 1
    
    print("=" * 60)
    print("Testing WITHOUT overwrite (should skip if files exist):")
    print("=" * 60)
    
    try:
        result1 = L3T_L4T_JET(runconfig_filename=runconfig_filename, strip_console=True, overwrite=False)
        print(f"Exit code without overwrite: {result1}")
    except Exception as e:
        print(f"Error without overwrite: {e}")
        return 1
    
    print("\n" + "=" * 60)
    print("Testing WITH overwrite (should process even if files exist):")
    print("=" * 60)
    
    try:
        result2 = L3T_L4T_JET(runconfig_filename=runconfig_filename, strip_console=True, overwrite=True)
        print(f"Exit code with overwrite: {result2}")
    except Exception as e:
        print(f"Error with overwrite: {e}")
        return 1
    
    print("\nTest completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(test_overwrite_functionality())