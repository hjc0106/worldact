from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from worldact.stage1_decomposition.sam3_segmenter import SAM3Candidate, SAM3CandidateSet, SAM3Segmenter


def test_to_pil_accepts_hwc_and_chw_rgb() -> None:
    hwc = torch.zeros(5, 7, 3)
    hwc[..., 0] = 1.0
    hwc[..., 1] = 0.5

    from_hwc = SAM3Segmenter._to_pil(hwc)
    from_chw = SAM3Segmenter._to_pil(hwc.permute(2, 0, 1))

    assert from_hwc.mode == "RGB"
    assert from_hwc.size == (7, 5)
    assert np.array_equal(np.asarray(from_hwc), np.asarray(from_chw))


def test_to_pil_accepts_singleton_batch_and_grayscale() -> None:
    image = SAM3Segmenter._to_pil(torch.ones(1, 1, 5, 7, dtype=torch.uint8) * 255)

    assert image.mode == "RGB"
    assert image.size == (7, 5)
    assert np.asarray(image).min() == 255


def test_to_pil_rejects_multi_image_batch() -> None:
    with pytest.raises(ValueError, match="expects one image"):
        SAM3Segmenter._to_pil(torch.zeros(2, 3, 5, 7))


def _make_candidate_set() -> SAM3CandidateSet:
    """Mirrors a real failure: a sprawling low-score blob outsizes a small high-score, compact detection."""
    image_size = (32, 32)

    sparse_blob = torch.zeros(image_size)
    sparse_blob[6:26, 6:26:2] = 1.0  # 20x10 pixels lit inside a 20x20 bbox: area=200, compactness=0.5
    blob_score = 0.05

    compact_box = torch.zeros(image_size)
    compact_box[10:16, 10:16] = 1.0  # solid 6x6 square: area=36, compactness=1.0
    compact_score = 0.7

    masks = torch.stack([sparse_blob, compact_box])
    scores = torch.tensor([blob_score, compact_score])
    boxes = SAM3Segmenter._mask_boxes(masks)
    candidates = []
    for idx, (mask, score, box) in enumerate(zip(masks, scores, boxes)):
        area = float((mask > 0.5).float().sum().item())
        bbox_area = float(
            max(0.0, (box[2] - box[0]).item() + 1.0) * max(0.0, (box[3] - box[1]).item() + 1.0)
        )
        quality = SAM3Segmenter._candidate_quality(mask, float(score.item()), box, image_size)
        candidates.append(
            SAM3Candidate(
                candidate_id=idx, mask=mask, score=float(score.item()), box=box,
                quality=quality, area=area, bbox_area=bbox_area,
            )
        )
    qualities = torch.tensor([c.quality for c in candidates])
    assert candidates[0].area > candidates[1].area  # blob has the larger raw pixel area
    assert candidates[1].score > candidates[0].score  # but the compact box is the confident detection
    assert candidates[1].quality > candidates[0].quality  # and it wins on the compactness-aware score
    return SAM3CandidateSet(
        prompt="building", image_size=image_size, masks=masks, scores=scores,
        boxes=boxes, qualities=qualities, candidates=candidates,
    )


def test_choose_candidate_largest_picks_sprawling_low_score_blob() -> None:
    candidates = _make_candidate_set()
    segmenter = SAM3Segmenter(SimpleNamespace(instance_strategy="largest"))

    chosen = segmenter.choose_candidate(candidates)

    assert chosen.candidate_id == 0


def test_choose_candidate_best_score_picks_compact_high_confidence_box() -> None:
    candidates = _make_candidate_set()
    segmenter = SAM3Segmenter(SimpleNamespace(instance_strategy="best_score"))

    chosen = segmenter.choose_candidate(candidates)

    assert chosen.candidate_id == 1


def test_choose_candidate_best_quality_picks_compact_high_confidence_box() -> None:
    candidates = _make_candidate_set()
    segmenter = SAM3Segmenter(SimpleNamespace(instance_strategy="best_quality"))

    chosen = segmenter.choose_candidate(candidates)

    assert chosen.candidate_id == 1
