"""018: part-based OSNet (needs torch + torchreid; skipped otherwise)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchreid")

from hooptrack.reid.parts import OSNetParts, embed_from_branches, load_checkpoint, save_checkpoint  # noqa: E402


def test_cosine_of_embeddings_is_mean_of_branch_cosines():
    a = [torch.randn(1, 8) for _ in range(5)]
    b = [torch.randn(1, 8) for _ in range(5)]
    cos = float(embed_from_branches(a) @ embed_from_branches(b).T)
    mean = float(np.mean([torch.nn.functional.cosine_similarity(x, y).item() for x, y in zip(a, b, strict=True)]))
    assert abs(cos - mean) < 1e-5
    assert abs(float(embed_from_branches(a).norm()) - 1.0) < 1e-5


def test_shapes_train_and_eval_any_input_size():
    m = OSNetParts(num_classes=7, parts=4, pretrained=False)
    x = torch.randn(4, 3, 384, 192)
    m.train()
    logits, feats = m(x)
    assert len(logits) == len(feats) == 5 and logits[0].shape == (4, 7) and feats[3].shape == (4, 512)
    m.eval()
    with torch.no_grad():
        e = m(torch.randn(2, 3, 256, 128))
    assert e.shape == (2, 2560)


def test_checkpoint_round_trip_and_extractor(tmp_path):
    from hooptrack.reid import build_reid

    m = OSNetParts(num_classes=7, parts=4, pretrained=False).eval()
    path = tmp_path / "parts.pth"
    save_checkpoint(m, path, (384, 192))
    m2, size = load_checkpoint(path)
    x = torch.randn(1, 3, 384, 192)
    with torch.no_grad():
        assert torch.allclose(m(x), m2(x), atol=1e-6)
    assert size == (384, 192)
    ext = build_reid({"type": "osnet_parts", "weights": str(path), "device": "cpu"})
    frame = np.random.randint(0, 255, (720, 1280, 3), np.uint8)
    f = ext(frame, np.array([[100, 100, 160, 300], [400, 200, 450, 380]], float))
    assert f.shape == (2, 2560) and ext.size == (192, 384)
    np.testing.assert_allclose(np.linalg.norm(f, axis=1), 1.0, atol=1e-5)
