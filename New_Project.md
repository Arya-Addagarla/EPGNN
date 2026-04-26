# New Project Guidelines

When expanding this repository or forking it for a new project, ensure the following conventions are maintained:

1. **Modular Consistency:** Keep all data processing in `data/`, neural architectures in `models/` or `modules/`, and execution loops in `engine/`.
2. **Reproducibility:** Any new dependencies must be added to `requirements.txt` and `setup_env.sh`.
3. **Testing:** All new data modalities (e.g., GPS, fiber-optic) must include an associated synthetic data generator in `data/mock_data.py` to allow for local smoke testing without requiring the full datasets.
