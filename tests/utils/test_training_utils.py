from src.utils.training_utils import create_directories


def test_create_directories(tmp_path):
    checkpoint = tmp_path / "checkpoints" / "model.pt"
    create_directories({"save_paths": {"model": str(checkpoint)}})
    assert checkpoint.parent.is_dir()
