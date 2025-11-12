from ECOv003_L3T_L4T_JET import L3T_L4T_JET

runconfig_filename = "example_L3T_L4T_JET_runconfig.xml"

# Run with default settings (will skip if output files already exist)
L3T_L4T_JET(runconfig_filename=runconfig_filename)

# To overwrite existing output files, set overwrite=True
# L3T_L4T_JET(runconfig_filename=runconfig_filename, overwrite=True)
