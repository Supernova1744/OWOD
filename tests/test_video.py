import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
from owod.video import sample_indices, hold_map


def test_sample_indices():
    assert sample_indices(300, 30, 1.0) == list(range(0, 300, 30))
    assert sample_indices(300, 30, 1.0, max_seconds=3) == [0, 30, 60]
    assert sample_indices(10, 30, 0.01) == list(range(10))           # step never below 1 frame
    with pytest.raises(ValueError):
        sample_indices(0, 30, 1.0)


def test_hold_map_holds_until_next_sample():
    m = hold_map(10, [0, 4, 8])
    assert m == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2]
    assert hold_map(5, [2]) == [None, None, 0, 0, 0]


def test_draw_results_skips_parts_and_marks_unknown():
    PIL = pytest.importorskip("PIL")
    from PIL import Image
    from owod.draw import draw_results
    from owod.crops import Box
    from owod.crop_classifier import Verdict
    from owod.pipeline import CropResult

    def res(box, label, unknown, part_of=None):
        return CropResult(0, box, [], None, False, Verdict(label, unknown, "not_in_known" if unknown else None, 0.5, {label: 0.9}), part_of)
    img = Image.new("RGB", (200, 100), (0, 0, 0))
    out = draw_results(img, [res(Box(10, 30, 60, 90), "cat", False), res(Box(100, 30, 150, 90), "unknown", True),
                             res(Box(20, 40, 30, 50), "cat", False, part_of=0)])
    assert out.getpixel((10, 60)) == (60, 220, 60)          # known: green outline
    assert out.getpixel((100, 60)) == (255, 60, 60)         # unknown: red outline
    assert out.getpixel((20, 45)) == (0, 0, 0)              # suppressed part: nothing drawn there
    assert img.getpixel((10, 60)) == (0, 0, 0)              # the input image is not changed


def test_parse_probe_basic_and_missing_frame_count():
    from owod.ffio import parse_probe
    info = {"streams": [{"codec_type": "video", "width": 1920, "height": 1080, "avg_frame_rate": "30000/1001",
                         "nb_frames": "N/A", "duration": "10.0"}], "format": {}}
    p = parse_probe(info)
    assert (p["width"], p["height"]) == (1920, 1080) and p["fps"] == pytest.approx(29.97, abs=0.01) and p["n_frames"] == 300
    info["streams"][0]["nb_frames"] = "250"
    assert parse_probe(info)["n_frames"] == 250


def test_parse_probe_rotation_swaps_size_and_errors():
    from owod.ffio import parse_probe
    s = {"codec_type": "video", "width": 1920, "height": 1080, "avg_frame_rate": "25/1", "nb_frames": "50",
         "side_data_list": [{"rotation": -90}]}
    p = parse_probe({"streams": [s], "format": {}})
    assert (p["width"], p["height"]) == (1080, 1920)
    with pytest.raises(ValueError):
        parse_probe({"streams": [{"codec_type": "audio"}], "format": {}})
    with pytest.raises(ValueError):
        parse_probe({"streams": [{"codec_type": "video", "width": 2, "height": 2, "avg_frame_rate": "0/0", "r_frame_rate": "0/0"}], "format": {}})


def test_ffmpeg_roundtrip_if_available(tmp_path):
    import shutil
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        pytest.skip("ffmpeg not installed here")
    pytest.importorskip("PIL")
    from PIL import Image
    from owod.ffio import VideoWriter, probe, read_frames
    out = str(tmp_path / "t.mp4")
    w = VideoWriter(out, 64, 48, 10)
    for c in (0, 100, 200):
        for _ in range(4):
            w.write(Image.new("RGB", (64, 48), (c, 0, 0)))
    w.close()
    info = probe(out)
    assert (info["width"], info["height"], info["n_frames"]) == (64, 48, 12) and info["fps"] == pytest.approx(10)
    frames = list(read_frames(out, 64, 48))
    assert len(frames) == 12 and frames[0].getpixel((5, 5))[0] < 30 and frames[-1].getpixel((5, 5))[0] > 150
