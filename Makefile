.PHONY: setup test train evaluate clean

setup:
	bash scripts/setup_env.sh

test:
	pytest tests/

smoke:
	bash scripts/run_smoke_test.sh

train:
	bash scripts/run_full_train.sh

clean:
	rm -rf __pycache__ .pytest_cache
	rm -f metadata_clean.csv mock_waveforms.hdf5 stead_earthquake.csv stead_noise.csv
