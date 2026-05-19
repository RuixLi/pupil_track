from pathlib import Path

import pytest

from pupil_track.config import Config
from pupil_track.pupil import Pupil


class FakeVideoReader:
    def __init__(self, path: str):
        self.path = path
        self.n_frames = 42
        self.fps = 30.0
        self.width = 640
        self.height = 480
        self.closed = False

    def close(self):
        self.closed = True


def test_load_from_config_loads_referenced_video_and_restores_state(tmp_path, monkeypatch):
    video_path = tmp_path / "eye.mp4"
    video_path.write_bytes(b"not a real video; VideoReader is patched")
    config_path = tmp_path / "eye_config.json"
    Config(
        data_path=str(video_path),
        roi={"x": 10, "y": 20, "w": 128, "h": 128},
        input_size=256,
        model_path="model.pth",
    ).save(config_path)
    monkeypatch.setattr("pupil_track.pupil.VideoReader", FakeVideoReader)

    pupil = Pupil(output_dir=tmp_path / "old_output", input_size=128)
    pupil._cached_full_frames = {0: object()}

    loaded_video = pupil.load_from_config(config_path)

    assert loaded_video == video_path
    assert pupil.reader.path == str(video_path)
    assert pupil.output_dir == config_path.parent
    assert pupil.roi == {"x": 10, "y": 20, "w": 128, "h": 128}
    assert pupil.input_size == 256
    assert pupil.model_path == Path("model.pth")
    assert pupil._cached_full_frames is None


def test_load_from_config_resolves_relative_video_path_from_config_directory(tmp_path, monkeypatch):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    video_path = config_dir / "eye.mp4"
    video_path.write_bytes(b"not a real video; VideoReader is patched")
    config_path = config_dir / "eye_config.json"
    Config(data_path="eye.mp4").save(config_path)
    monkeypatch.setattr("pupil_track.pupil.VideoReader", FakeVideoReader)

    pupil = Pupil()

    loaded_video = pupil.load_from_config(config_path)

    assert loaded_video == video_path
    assert pupil.reader.path == str(video_path)
    assert pupil.output_dir == config_dir


def test_load_from_config_rejects_config_without_video_path(tmp_path):
    config_path = tmp_path / "missing_video_path_config.json"
    Config().save(config_path)

    pupil = Pupil()

    with pytest.raises(ValueError, match="data.path"):
        pupil.load_from_config(config_path)


def test_load_from_config_rejects_missing_video_file(tmp_path):
    config_path = tmp_path / "missing_video_config.json"
    Config(data_path=str(tmp_path / "missing.mp4")).save(config_path)

    pupil = Pupil()

    with pytest.raises(FileNotFoundError, match="Video file from config not found"):
        pupil.load_from_config(config_path)
