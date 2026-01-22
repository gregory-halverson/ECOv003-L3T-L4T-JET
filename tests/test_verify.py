import pytest
from ECOv003_L3T_L4T_JET import verify

def test_verify():
    assert verify(), "Model verification failed: outputs do not match expected results."
