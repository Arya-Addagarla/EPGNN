import pandas as pd
import numpy as np
import h5py

def create_mock_stead_data(num_earthquakes=50, num_noise=50):
    eq_data = []
    for i in range(num_earthquakes):
        trace_name = f"eq_trace_{i}"
        eq_data.append({
            "trace_name": trace_name,
            "receiver_latitude": np.random.uniform(32.0, 42.0),
            "receiver_longitude": np.random.uniform(-124.0, -114.0),
            "source_magnitude": np.random.uniform(1.0, 7.0),
            "p_arrival_sample": np.random.randint(100, 300)
        })
    df_eq = pd.DataFrame(eq_data)
    df_eq.to_csv("stead_earthquake.csv", index=False)

    noise_data = []
    for i in range(num_noise):
        trace_name = f"noise_trace_{i}"
        noise_data.append({
            "trace_name": trace_name,
            "receiver_latitude": np.random.uniform(32.0, 42.0),
            "receiver_longitude": np.random.uniform(-124.0, -114.0),
            "source_magnitude": np.nan,
            "p_arrival_sample": np.nan
        })
    df_noise = pd.DataFrame(noise_data)
    df_noise.to_csv("stead_noise.csv", index=False)

    h5_filename = "mock_waveforms.hdf5"
    with h5py.File(h5_filename, 'w') as f:
        grp_eq = f.create_group("earthquake")
        for trace_name in df_eq["trace_name"]:
            waveform = np.cumsum(np.random.randn(600, 3), axis=0) * 0.1
            p_arr = df_eq[df_eq["trace_name"] == trace_name]["p_arrival_sample"].values[0]
            waveform[int(p_arr):int(p_arr)+100] += np.random.randn(100, 3) * 5.0
            f.create_dataset(trace_name, data=waveform)
            
        grp_noise = f.create_group("non_earthquake")
        for trace_name in df_noise["trace_name"]:
            waveform = np.random.randn(600, 3) * 0.5
            f.create_dataset(trace_name, data=waveform)

if __name__ == "__main__":
    create_mock_stead_data()
