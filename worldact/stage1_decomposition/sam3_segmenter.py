from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from worldact.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class SAM3Candidate:
    candidate_id: int
    mask: torch.Tensor
    score: float
    box: torch.Tensor
    quality: float
    area: float
    bbox_area: float


@dataclass
class SAM3CandidateSet:
    prompt: str
    image_size: tuple[int, int]
    masks: torch.Tensor
    scores: torch.Tensor
    boxes: torch.Tensor
    qualities: torch.Tensor
    candidates: list[SAM3Candidate]


class SAM3Segmenter:
    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self._processor = None

    def _load_processor(self):
        if self._processor is not None:
            return self._processor
        source_dir = Path(self.cfg.source_dir)
        checkpoint = Path(self.cfg.checkpoint_path)
        if not source_dir.exists():
            raise FileNotFoundError(f"SAM3 source_dir not found: {source_dir}")
        if not checkpoint.exists():
            raise FileNotFoundError(f"SAM3 checkpoint_path not found: {checkpoint}")
        sys.path.insert(0, str(source_dir.resolve()))
        try:
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model
        except ImportError as exc:
            raise ImportError(
                "SAM3 dependencies are not importable. Install the official repo in the worldact env, e.g. "
                "`conda run -n worldact python -m pip install -e third_party/sam3` and install missing optional deps."
            ) from exc
        self._patch_sam3_fused_mlp_dtype()
        model = build_sam3_image_model(
            checkpoint_path=str(checkpoint),
            load_from_HF=False,
            device=self.cfg.device,
            eval_mode=True,
        )
        self._processor = Sam3Processor(
            model,
            device=self.cfg.device,
            confidence_threshold=float(self.cfg.confidence_threshold),
        )
        return self._processor

    def _patch_sam3_fused_mlp_dtype(self) -> None:
        """Keep SAM3's fused fc1 output compatible with following fp32 fc2.

        The official SAM3 image path currently returns bfloat16 from
        sam3.perflib.fused.addmm_act(), while the next nn.Linear layer may
        remain float32 when loading local checkpoints. PyTorch then raises:
        "mat1 and mat2 must have the same dtype". Recasting the fused output to
        the active layer dtype preserves the official fast path while avoiding
        the dtype mismatch.
        """
        try:
            import sam3.model.vitdet as vitdet
        except ImportError:
            return
        if getattr(vitdet.addmm_act, "_worldact_dtype_patch", False):
            return
        original_addmm_act = vitdet.addmm_act

        def patched_addmm_act(activation, linear, mat1):
            out = original_addmm_act(activation, linear, mat1)
            return out.to(dtype=linear.weight.dtype)

        patched_addmm_act._worldact_dtype_patch = True  # type: ignore[attr-defined]
        vitdet.addmm_act = patched_addmm_act

    def segment(self, image: torch.Tensor, text_prompt: str) -> torch.Tensor | None:
        candidates = self.segment_candidates(image, text_prompt)
        if candidates is None or not candidates.candidates:
            return None
        chosen = self.choose_candidate(candidates)
        if chosen is None:
            masks = candidates.masks
            return (masks.sum(dim=0) > 0).to("cpu").float()
        return chosen.mask.to("cpu").float()

    def choose_candidate(self, candidates: SAM3CandidateSet) -> SAM3Candidate | None:
        if not candidates.candidates:
            return None
        strategy = str(self.cfg.instance_strategy)
        if strategy == "union":
            return None
        masks = candidates.masks
        scores = candidates.scores
        if strategy == "best_score" and scores is not None and scores.numel() > 0:
            idx = int(torch.argmax(scores).item())
            return candidates.candidates[idx]
        if strategy == "best_quality" and candidates.qualities is not None and candidates.qualities.numel() > 0:
            idx = int(torch.argmax(candidates.qualities).item())
            return candidates.candidates[idx]
        areas = masks.flatten(1).sum(dim=1)
        idx = int(torch.argmax(areas).item())
        return candidates.candidates[idx]

    def segment_candidates(self, image: torch.Tensor, text_prompt: str) -> SAM3CandidateSet | None:
        processor = self._load_processor()
        pil = self._to_pil(image)
        state = processor.set_image(pil)
        output = processor.set_text_prompt(prompt=text_prompt, state=state)
        masks = output.get("masks")
        scores = output.get("scores")
        boxes = output.get("boxes")
        if masks is None or masks.numel() == 0:
            log.debug("SAM3 found no mask for prompt %r", text_prompt)
            return None
        masks = masks.detach().float()
        if masks.ndim == 4:
            masks = masks[:, 0]
        if scores is None:
            scores = torch.ones(masks.shape[0], device=masks.device, dtype=masks.dtype)
        if boxes is None:
            boxes = self._mask_boxes(masks)
        scores = scores.detach().float()
        boxes = boxes.detach().float()
        candidates: list[SAM3Candidate] = []
        qualities = []
        candidate_masks = []
        candidate_scores = []
        candidate_boxes = []
        for idx in range(masks.shape[0]):
            mask = masks[idx].to("cpu").float()
            score = float(scores[idx].item())
            box = boxes[idx].to("cpu").float()
            for component_mask, component_box in self._candidate_components(mask, box, pil.size[::-1]):
                area = float((component_mask > 0.5).float().sum().item())
                bbox_area = float(
                    max(0.0, (component_box[2] - component_box[0]).item() + 1.0)
                    * max(0.0, (component_box[3] - component_box[1]).item() + 1.0)
                )
                quality = self._candidate_quality(component_mask, score, component_box, pil.size[::-1])
                candidate_id = len(candidates)
                qualities.append(quality)
                candidate_masks.append(component_mask)
                candidate_scores.append(score)
                candidate_boxes.append(component_box)
                candidates.append(
                    SAM3Candidate(
                        candidate_id=candidate_id,
                        mask=component_mask,
                        score=score,
                        box=component_box,
                        quality=quality,
                        area=area,
                        bbox_area=bbox_area,
                    )
                )
        if not candidates:
            return None
        return SAM3CandidateSet(
            prompt=text_prompt,
            image_size=(pil.height, pil.width),
            masks=torch.stack(candidate_masks, dim=0),
            scores=torch.tensor(candidate_scores, dtype=torch.float32),
            boxes=torch.stack(candidate_boxes, dim=0),
            qualities=torch.tensor(qualities, dtype=torch.float32),
            candidates=candidates,
        )

    def _candidate_components(
        self,
        mask: torch.Tensor,
        box: torch.Tensor,
        image_size: tuple[int, int],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        cfg = getattr(self.cfg, "component_filter", None)
        if cfg is None or not bool(getattr(cfg, "enabled", False)):
            return [(mask, box)]

        mask_bool = mask > 0.5
        if not bool(mask_bool.any()):
            return []
        components = self._connected_components(mask_bool)
        image_area = max(1.0, float(image_size[0] * image_size[1]))
        min_area = max(float(getattr(cfg, "min_component_area_px", 1)), float(getattr(cfg, "min_area_ratio", 0.0)) * image_area)
        max_area = float(getattr(cfg, "max_area_ratio", 1.0)) * image_area
        kept: list[tuple[torch.Tensor, torch.Tensor]] = []
        for component in components:
            area = float(component.sum().item())
            if area < min_area or area > max_area:
                continue
            component_box = self._mask_boxes(component.float().unsqueeze(0))[0].cpu()
            kept.append((component.float(), component_box))
        if kept:
            return kept
        if bool(getattr(cfg, "reject_multi_component_union", True)):
            return []
        return [(mask, box)]

    @staticmethod
    def _connected_components(mask: torch.Tensor) -> list[torch.Tensor]:
        height, width = mask.shape
        visited = torch.zeros_like(mask, dtype=torch.bool)
        components: list[torch.Tensor] = []
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for y0, x0 in mask.nonzero(as_tuple=False).tolist():
            if bool(visited[y0, x0]):
                continue
            stack = [(int(y0), int(x0))]
            visited[y0, x0] = True
            ys: list[int] = []
            xs: list[int] = []
            while stack:
                y, x = stack.pop()
                ys.append(y)
                xs.append(x)
                for dy, dx in neighbors:
                    ny, nx = y + dy, x + dx
                    if ny < 0 or nx < 0 or ny >= height or nx >= width:
                        continue
                    if bool(visited[ny, nx]) or not bool(mask[ny, nx]):
                        continue
                    visited[ny, nx] = True
                    stack.append((ny, nx))
            component = torch.zeros_like(mask, dtype=torch.bool)
            component[torch.tensor(ys, dtype=torch.long), torch.tensor(xs, dtype=torch.long)] = True
            components.append(component)
        return components

    def _select_instance(self, masks: torch.Tensor, scores: torch.Tensor | None) -> int:
        strategy = str(self.cfg.instance_strategy)
        if strategy == "best_score" and scores is not None and scores.numel() > 0:
            return int(torch.argmax(scores).item())
        areas = masks.flatten(1).sum(dim=1)
        return int(torch.argmax(areas).item())

    @staticmethod
    def _mask_boxes(masks: torch.Tensor) -> torch.Tensor:
        boxes = []
        for mask in masks:
            ys, xs = (mask > 0.5).nonzero(as_tuple=True)
            if ys.numel() == 0:
                boxes.append(torch.zeros(4, dtype=torch.float32, device=mask.device))
                continue
            boxes.append(
                torch.tensor(
                    [xs.min().item(), ys.min().item(), xs.max().item(), ys.max().item()],
                    dtype=torch.float32,
                    device=mask.device,
                )
            )
        return torch.stack(boxes, dim=0) if boxes else torch.zeros(0, 4, dtype=torch.float32, device=masks.device)

    @staticmethod
    def _candidate_quality(mask: torch.Tensor, score: float, box: torch.Tensor, image_size: tuple[int, int]) -> float:
        height, width = image_size
        area = float((mask > 0.5).float().sum().item())
        area_ratio = area / max(1.0, float(height * width))
        bbox_w = max(0.0, float(box[2].item() - box[0].item() + 1.0))
        bbox_h = max(0.0, float(box[3].item() - box[1].item() + 1.0))
        bbox_area = max(1.0, bbox_w * bbox_h)
        compactness = min(1.0, area / bbox_area)
        size_prior = max(0.1, min(1.0, 0.25 + area_ratio ** 0.5))
        edge_penalty = 1.0 if area > 0 else 0.0
        return float(score) * size_prior * compactness * edge_penalty

    @staticmethod
    def _to_pil(image: torch.Tensor) -> Image.Image:
        """Convert one tensor image to RGB PIL, accepting HWC or CHW layouts."""
        tensor = image.detach().cpu()
        if tensor.ndim == 4:
            if tensor.shape[0] != 1:
                raise ValueError(f"SAM3 expects one image, got batch shape {tuple(tensor.shape)}")
            tensor = tensor[0]

        if tensor.ndim == 3:
            channel_sizes = (1, 3, 4)
            if tensor.shape[-1] in channel_sizes:
                # Preserve the HWC layout used by WorldAct rendered views. In
                # the rare ambiguous case where both ends look like channels,
                # prefer HWC for backward compatibility.
                pass
            elif tensor.shape[0] in channel_sizes:
                tensor = tensor.permute(1, 2, 0)
            else:
                raise ValueError(
                    "SAM3 image must have 1, 3, or 4 channels in HWC/CHW layout, "
                    f"got shape {tuple(tensor.shape)}"
                )
            if tensor.shape[-1] == 1:
                tensor = tensor[..., 0]
        elif tensor.ndim != 2:
            raise ValueError(f"SAM3 image must be HW, HWC, CHW, or a batch of one; got {tuple(tensor.shape)}")

        if torch.is_floating_point(tensor):
            pixels = (tensor.float().clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
        elif tensor.dtype == torch.bool:
            pixels = tensor.to(torch.uint8) * 255
        elif tensor.dtype == torch.uint8:
            pixels = tensor
        else:
            pixels = tensor.clamp(0, 255).to(torch.uint8)

        return Image.fromarray(pixels.contiguous().numpy()).convert("RGB")
