"""
Debug Script for monte_carlo_sensitivity linregress AttributeError

ISSUE SUMMARY:
--------------
sensitivity_analysis() with use_joint_run=True fails with AttributeError when
using real ECOv002 Cal-Val data with process_JET_table. The error occurs at
line 276 in _sensitivity_analysis_joint when scipy.stats.linregress receives
arrays with insufficient variation.

ERROR STACK TRACE (from JET Sensitivity notebook):
--------------------------------------------------
File .../monte_carlo_sensitivity/sensitivity_analysis.py:64, in sensitivity_analysis
    return _sensitivity_analysis_joint(...)

File .../monte_carlo_sensitivity/sensitivity_analysis.py:276, in _sensitivity_analysis_joint
    r2 = scipy.stats.linregress(
        variable_perturbation_df.input_perturbation_std,
        variable_perturbation_df.output_perturbation_std
    )[2] ** 2

File .../scipy/stats/_stats_py.py:10524, in linregress
    ssxm, ssxym, _, ssym = np.cov(x, y, bias=1).flat

File .../numpy/lib/function_base.py:2724, in cov
    avg, w_sum = average(X, axis=1, weights=w, returned=True)

File .../numpy/lib/function_base.py:557, in average
    if scl.shape != avg_as_array.shape:
AttributeError: 'float' object has no attribute 'shape'

ROOT CAUSE:
-----------
When using the ACTUAL ECOv002 Cal-Val data with certain input variables,
the perturbations result in arrays that cause np.cov() to fail internally.

REPRODUCTION:
-------------
This script uses the REAL ECOv002 Cal-Val data and process_JET_table function
to reproduce the exact error from the notebook.

SUCCESS CRITERIA:
-----------------
The issue is FIXED when this script runs without errors and prints:
"✓ SUCCESS: The linregress issue is FIXED!"
"""

import sys
import os

# Add current directory to path to find ECOv003_L3T_L4T_JET
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings
import numpy as np
import pandas as pd
import scipy.stats
from typing import Callable

# Try to import ECOv003 components for real data
HAVE_ECOV003 = False
try:
    # Import specific modules directly to avoid full package dependencies
    import imp
    import importlib.util
    
    # Load process_JET_table
    spec = importlib.util.spec_from_file_location(
        "process_JET_table_module",
        os.path.join(os.path.dirname(__file__), "ECOv003_L3T_L4T_JET", "process_JET_table.py")
    )
    process_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(process_module)
    process_JET_table = process_module.process_JET_table
    
    # Load load_ECOv002_calval_JET_inputs
    spec2 = importlib.util.spec_from_file_location(
        "calval_inputs_module",
        os.path.join(os.path.dirname(__file__), "ECOv003_L3T_L4T_JET", "ECOv002_calval_JET_inputs.py")
    )
    inputs_module = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(inputs_module)
    load_ECOv002_calval_JET_inputs = inputs_module.load_ECOv002_calval_JET_inputs
    
    HAVE_ECOV003 = True
except Exception as e:
    HAVE_ECOV003 = False
    import_error = str(e)


def test_with_real_data():
    """
    Use REAL ECOv002 data and process_JET_table to reproduce the error.
    """
    print("\n" + "="*70)
    print("TEST: Real ECOv002 Data with process_JET_table")
    print("="*70)
    
    if not HAVE_ECOV003:
        print("\n⚠ SKIPPED: ECOv003_L3T_L4T_JET module not available")
        print("  Run this from the ECOv003-L3T-L4T-JET directory.")
        return None
    
    try:
        from monte_carlo_sensitivity import sensitivity_analysis, divide_absolute_by_unperturbed
        
        # Load and filter real data (same as notebook)
        print("\n→ Loading ECOv002 Cal-Val data...")
        input_df = load_ECOv002_calval_JET_inputs()
        input_df = input_df[input_df.ST_C <= 50]
        input_df = input_df[input_df.NDVI.apply(lambda NDVI: NDVI > 0.05)]
        print(f"✓ Loaded {len(input_df)} rows after filtering")
        
        # These are the EXACT variables from the failing notebook cell
        input_variables = [
            "ST_C", "NDVI", "albedo", "Ta_C", "RH",
            "AOT", "COT", "vapor_gccm", "ozone_cm",
            "elevation_m", "canopy_height_meters"
        ]
        
        output_variables = [
            "Rn_Wm2", "ET_daylight_kg", "GPP_inst_g_m2_s"
        ]
        
        print(f"\n→ Running sensitivity_analysis (use_joint_run=True)...")
        print(f"   Input vars: {len(input_variables)}")
        print(f"   Output vars: {len(output_variables)}")
        print("   (This should trigger AttributeError if unfixed)")
        
        # This is the EXACT call from the notebook
        perturbation_df, sensitivity_metrics_df = sensitivity_analysis(
            input_df=input_df,
            input_variables=input_variables,
            output_variables=output_variables,
            forward_process=process_JET_table,
            normalization_function=divide_absolute_by_unperturbed,
            use_joint_run=True  # CRITICAL - triggers the bug
        )
        
        print(f"\n✓ SUCCESS: Completed without errors!")
        print(f"   Perturbation shape: {perturbation_df.shape}")
        print(f"   Metrics shape: {sensitivity_metrics_df.shape}")
        return True
        
    except AttributeError as e:
        if "'float' object has no attribute 'shape'" in str(e):
            print(f"\n✗ EXPECTED ERROR (unfixed): AttributeError")
            print(f"   {str(e)}")
            print("\n   This IS the bug from the JET Sensitivity notebook!")
            return False
        else:
            raise
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    print("\n" + "#"*70)
    print("# DEBUG: JET Sensitivity linregress AttributeError")
    print("#"*70)
    
    print("\nPURPOSE: Reproduce AttributeError from JET Sensitivity notebook")
    print("-" * 70)
    
    if HAVE_ECOV003:
        print("✓ ECOv003 package detected - using REAL data")
    else:
        print("⚠ ECOv003 package not found")
        print("  Run from: /Users/halverso/Projects/ECOv003-L3T-L4T-JET")
        return 1
    
    # Check monte_carlo_sensitivity
    try:
        import monte_carlo_sensitivity
        print(f"✓ monte_carlo_sensitivity: {monte_carlo_sensitivity.__file__}")
    except ImportError:
        print("✗ ERROR: monte_carlo_sensitivity not found!")
        print("  Install: pip install monte-carlo-sensitivity")
        return 1
    
    print(f"  numpy: {np.__version__}")
    print(f"  scipy: {scipy.__version__}")
    print(f"  pandas: {pd.__version__}")
    
    # Run the test
    result = test_with_real_data()
    
    print("\n" + "="*70)
    if result:
        print("✓ SUCCESS: The linregress issue is FIXED!")
        print("="*70)
        print("\nThe JET Sensitivity notebook should now work.")
        return 0
    elif result is False:
        print("✗ FAILURE: The linregress issue EXISTS.")
        print("="*70)
        print("\nRECOMMENDED FIX in monte_carlo_sensitivity/sensitivity_analysis.py:")
        print("-" * 70)
        print("\nAround line 276, replace:")
        print("  r2 = scipy.stats.linregress(")
        print("      variable_perturbation_df.input_perturbation_std,")
        print("      variable_perturbation_df.output_perturbation_std")
        print("  )[2] ** 2")
        print("\nWith safe version:")
        print("  x = np.asarray(variable_perturbation_df.input_perturbation_std)")
        print("  y = np.asarray(variable_perturbation_df.output_perturbation_std)")
        print("  mask = np.isfinite(x) & np.isfinite(y)")
        print("  x, y = x[mask], y[mask]")
        print("  if len(x) < 2 or np.var(x) < 1e-10 or np.var(y) < 1e-10:")
        print("      r2 = np.nan")
        print("  else:")
        print("      try:")
        print("          r2 = scipy.stats.linregress(x, y)[2] ** 2")
        print("      except (AttributeError, ValueError):")
        print("          r2 = np.nan")
        return 1
    else:
        print("⊘ SKIPPED: ECOv003 package not available")
        return 1


if __name__ == "__main__":
    sys.exit(main())
