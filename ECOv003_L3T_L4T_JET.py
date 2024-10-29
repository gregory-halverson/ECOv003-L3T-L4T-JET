from sys import argv, exit
import logging
import yaml
import colored_logging as cl

logger = logging.getLogger(__name__)

# FIXME revert to XML instead of YAML for ECOSTRESS Collection 3 run-configs

def ECOv003_L3T_L4T_JET(runconfig_filename = argv[1]) -> int:
    """
    entry point for ECOSTRESS Collection 3 Level 3/4 Evapotranspiration PGE
    """
    with open(runconfig_filename, "r") as file:
        runconfig_dict = yaml.safe_load(file)
    
    logger.info(f"run-config file: {runconfig_filename}")
    logger.info(f"run-config:")
    logger.info(yaml.dump(runconfig_dict))

    exit_code = 0

    return exit_code

if __name__ == "__main__":
    exit(SBGv001_L4T_JET(runconfig_filename=argv[1]))
