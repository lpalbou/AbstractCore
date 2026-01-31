from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.basic
def test_extract_video_frames_uses_keyframes_when_available(monkeypatch, tmp_path: Path) -> None:
    from abstractcore.media.utils import video_frames

    # Pretend ffmpeg exists and skip actual subprocess execution.
    monkeypatch.setattr(video_frames, "_which", lambda cmd: "/usr/bin/" + cmd)

    # Provide a fake keyframe list and avoid calling ffmpeg.
    monkeypatch.setattr(video_frames, "probe_keyframe_timestamps_s", lambda p: [0.1, 1.0, 2.0, 3.0, 4.0])
    monkeypatch.setattr(video_frames, "probe_duration_s", lambda p: 10.0)

    calls = []

    def fake_run(cmd, check, stdout, stderr):
        calls.append(cmd)
        # Create the expected output file so the extractor considers it “exists”.
        out_path = Path(cmd[-1])
        out_path.write_bytes(b"x")
        return None

    monkeypatch.setattr(video_frames.subprocess, "run", fake_run)

    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"0")

    frames, timestamps = video_frames.extract_video_frames(
        video_path,
        max_frames=3,
        frame_format="jpg",
        sampling_strategy="keyframes",
        output_dir=tmp_path / "out",
    )

    assert len(frames) == 3
    assert timestamps == [0.1, 2.0, 4.0]
    # Ensure we invoked ffmpeg with -ss at those timestamps (best-effort check).
    ss_args = [cmd[cmd.index("-ss") + 1] for cmd in calls if "-ss" in cmd]
    assert len(ss_args) == 3


@pytest.mark.basic
def test_extract_video_frames_falls_back_to_uniform_when_no_keyframes(monkeypatch, tmp_path: Path) -> None:
    from abstractcore.media.utils import video_frames

    monkeypatch.setattr(video_frames, "_which", lambda cmd: "/usr/bin/" + cmd)
    monkeypatch.setattr(video_frames, "probe_keyframe_timestamps_s", lambda p: [])
    monkeypatch.setattr(video_frames, "probe_duration_s", lambda p: 9.0)

    def fake_run(cmd, check, stdout, stderr):
        out_path = Path(cmd[-1])
        out_path.write_bytes(b"x")
        return None

    monkeypatch.setattr(video_frames.subprocess, "run", fake_run)

    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"0")

    _frames, timestamps = video_frames.extract_video_frames(
        video_path,
        max_frames=3,
        frame_format="jpg",
        sampling_strategy="keyframes",
        output_dir=tmp_path / "out2",
    )

    # Uniform sampling away from endpoints: duration*(i+1)/(n+1) for i=0..2
    assert timestamps == [2.25, 4.5, 6.75]


@pytest.mark.basic
def test_extract_video_frames_includes_scale_filter_when_max_side_set(monkeypatch, tmp_path: Path) -> None:
    from abstractcore.media.utils import video_frames

    monkeypatch.setattr(video_frames, "_which", lambda cmd: "/usr/bin/" + cmd)
    monkeypatch.setattr(video_frames, "probe_duration_s", lambda p: 10.0)

    calls = []

    def fake_run(cmd, check, stdout, stderr):
        calls.append(cmd)
        out_path = Path(cmd[-1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"x")
        return None

    monkeypatch.setattr(video_frames.subprocess, "run", fake_run)

    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"0")

    _frames, _timestamps = video_frames.extract_video_frames(
        video_path,
        max_frames=1,
        frame_format="jpg",
        max_side=512,
        output_dir=tmp_path / "out",
    )

    assert calls, "Expected ffmpeg to be invoked"
    vf_args = [cmd[cmd.index("-vf") + 1] for cmd in calls if "-vf" in cmd]
    assert vf_args, "Expected -vf scale filter when max_side is set"
    assert any("scale=" in v for v in vf_args)
