"""027-028: SigLIP / CLIP as ReID models. Tiny random towers, no download."""

import json

import numpy as np
import pytest

transformers = pytest.importorskip("transformers")
torch = pytest.importorskip("torch")


def _tiny(tmp_path, kind):
    from transformers import (
        CLIPImageProcessor,
        CLIPVisionConfig,
        CLIPVisionModelWithProjection,
        SiglipImageProcessor,
        SiglipVisionConfig,
        SiglipVisionModel,
    )

    common = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                  image_size=224, patch_size=16)
    path = tmp_path / kind
    if kind == "siglip":
        SiglipVisionModel(SiglipVisionConfig(**common)).save_pretrained(path)
        SiglipImageProcessor().save_pretrained(path)
    else:
        CLIPVisionModelWithProjection(CLIPVisionConfig(projection_dim=24, **common)).save_pretrained(path)
        CLIPImageProcessor().save_pretrained(path)
    return path


def _frame():
    rng = np.random.default_rng(0)
    return (rng.random((300, 400, 3)) * 255).astype(np.uint8), np.array([[10, 20, 74, 148], [200, 40, 250, 240.0]])


@pytest.mark.parametrize("kind", ["siglip", "clip"])
def test_reid_uses_256x128_and_its_own_preprocessing(tmp_path, kind):
    from hooptrack.reid.extractors import build_reid

    ext = build_reid({"type": kind, "weights": str(_tiny(tmp_path, kind))})
    assert ext.input_size == (256, 128)
    frame, boxes = _frame()
    x = ext._pixels([frame[20:148, 10:74, ::-1].copy()])
    assert tuple(x.shape[-2:]) == (256, 128)              # not squeezed, not centre-cropped
    e = ext(frame, boxes)
    assert e.shape == (2, ext.dim) and np.allclose(np.linalg.norm(e, axis=1), 1, atol=1e-5)


def test_fine_tuned_folder_sets_its_input_size(tmp_path):
    from hooptrack.reid.extractors import build_reid

    path = _tiny(tmp_path, "siglip")
    (path / "hooptrack_reid.json").write_text(json.dumps({"input_size": [224, 112]}))
    assert build_reid({"type": "siglip", "weights": str(path)}).input_size == (224, 112)


def test_team_label_path_is_unchanged(tmp_path):
    """020 builds SigLIPExtractor(path) directly: the model's own processor, as before 027."""
    from transformers import AutoImageProcessor, SiglipVisionModel

    from hooptrack.reid.siglip import SigLIPExtractor

    path = _tiny(tmp_path, "siglip")
    ext = SigLIPExtractor(str(path))
    assert ext.input_size is None
    frame, boxes = _frame()
    model, proc = SiglipVisionModel.from_pretrained(path).eval(), AutoImageProcessor.from_pretrained(path)
    crops = [np.ascontiguousarray(frame[int(b[1]):int(b[3]), int(b[0]):int(b[2]), ::-1]) for b in boxes]
    with torch.no_grad():
        old = model(pixel_values=proc(images=crops, return_tensors="pt")["pixel_values"]).pooler_output
    old = torch.nn.functional.normalize(old, dim=1).numpy()
    assert np.allclose(ext(frame, boxes), old, atol=1e-5)


def test_siglip_needs_weights():
    from hooptrack.reid.extractors import build_reid

    with pytest.raises(ValueError):
        build_reid({"type": "siglip"})
