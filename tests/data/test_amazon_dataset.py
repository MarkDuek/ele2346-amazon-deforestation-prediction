import h5py
import numpy as np

from src.data.amazon_dataset import AmazonDataset
from src.utils.utils import load_config


def test_dataset_returns_expected_sequence(tmp_path):
    config = load_config("config.yaml")
    input_path = tmp_path / "input_patches.h5"
    patches_per_time = 2
    time_steps = 8
    patches = np.zeros((patches_per_time * time_steps, 5, 32, 32))
    patches[12:14, 1] = 1

    with h5py.File(input_path, "w") as h5_file:
        h5_file.create_dataset("patches", data=patches)
        h5_file.create_dataset(
            "time_indices",
            data=np.repeat(np.arange(time_steps), patches_per_time),
        )

    config["data"]["h5_paths"]["input"] = str(input_path)
    config["data"]["time_slice"] = 7
    dataset = AmazonDataset(config)

    try:
        input_tensor, target_tensor = dataset[0]
        assert len(dataset) == patches_per_time
        assert input_tensor.shape == (5, 6, 32, 32)
        assert target_tensor.shape == (1, 1, 32, 32)
        assert target_tensor.sum() == 32 * 32
    finally:
        dataset.input_h5.close()
