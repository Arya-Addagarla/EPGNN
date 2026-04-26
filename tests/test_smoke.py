import pytest
import os

def test_mock_data_generation():
    # Verify the mock data script runs without throwing an error
    exit_code = os.system("python main.py --mode mock")
    assert exit_code == 0
    
    assert os.path.exists("mock_waveforms.hdf5")
    assert os.path.exists("stead_earthquake.csv")
    assert os.path.exists("stead_noise.csv")
