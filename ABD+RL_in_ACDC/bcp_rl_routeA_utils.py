"""BCP Route-A utilities (K=4 mask actions + reward helpers).

Action space (K=4)
- 0: Random-Rect
- 1: Entropy-Rect (teacher uncertainty)
- 2: Disagree-Rect (teacher vs student probability disagreement)
- 3: Edge-Rect (teacher foreground edge strength)

Mask convention
- M==1 => keep "base" image/label (left operand in mixing)
- M==0 => paste from "patch" image/label (right operand in mixing)

Example (outward):
  net_input_unl = uimg_a * M + img_a * (1 - M)
  target_unl    = plab_a * M + lab_a * (1 - M)

All computations are per-iteration and do not require unlabeled ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class MaskActionConfig:
     beta: float = 2.0 / 3.0          # rectangle size ratio
     guided_beta: float = 0.50       # smaller rect ratio for guided actions (1/2/3); reduces structure breakage
     topk_percent: float = 0.02      # sample center from top-k% pixels (avoid unstable argmax peaks), e.g. 0.02 => top 2%
     per_sample: bool = True          # True => generate different mask per sample; False => one mask for batch
     eps: float = 1e-6


def _sample_center_topk(score_map: torch.Tensor, topk_percent: float = 0.02) -> Tuple[int, int]:
    """Sample a center from the TOP-k% pixels of a score map [H,W].

    Why: argmax is extremely unstable in semi-supervised setting (noisy pseudo labels).
    Sampling within the top-k% keeps the "hard region" intent but avoids locking onto a single noisy spike.
    """
    H, W = score_map.shape
    HW = H * W
    k = max(1, int(round(float(topk_percent) * HW)))
    flat = score_map.reshape(-1)
    # If the map is degenerate, fall back to a uniform random center.
    if not torch.isfinite(flat).all() or float(flat.max().item()) == float(flat.min().item()):
        ridx = int(torch.randint(0, HW, (1,), device=score_map.device).item())
        return ridx // W, ridx % W
    # Take top-k indices then randomly pick one of them.
    topk = torch.topk(flat, k=k, largest=True).indices
    pick = int(torch.randint(0, k, (1,), device=score_map.device).item())
    ridx = int(topk[pick].item())
    return ridx // W, ridx % W







# -------------------------
# Basic maps from logits
# -------------------------

def softmax_probs(logits: torch.Tensor) -> torch.Tensor:
    """Convert logits [B,C,H,W] to probs [B,C,H,W]."""
    return F.softmax(logits, dim=1)


def entropy_map_from_probs(probs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Pixel-wise entropy map. Output shape [B,H,W]."""
    p = torch.clamp(probs, eps, 1.0)
    ent = -(p * torch.log(p)).sum(dim=1)  # [B,H,W]
    return ent


def fg_prob_from_probs(probs: torch.Tensor, bg_channel: int = 0) -> torch.Tensor:
    """Foreground probability p_fg = 1 - p_bg. Output shape [B,1,H,W]."""
    p_bg = probs[:, bg_channel:bg_channel+1]
    return 1.0 - p_bg


def disagree_map_teacher_student(p_teacher: torch.Tensor, p_student: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Teacher-student disagreement map via KL(pT || pS). Output [B,H,W]."""
    pT = torch.clamp(p_teacher, eps, 1.0)
    pS = torch.clamp(p_student, eps, 1.0)
    kl = (pT * (torch.log(pT) - torch.log(pS))).sum(dim=1)  # [B,H,W]
    return kl


# -------------------------
# Sobel gradients (edge proxy)
# -------------------------

def _sobel_kernels(device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    kx = torch.tensor([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], device=device, dtype=dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]], device=device, dtype=dtype).view(1, 1, 3, 3)
    return kx, ky


def sobel_grad_mag(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute Sobel gradient magnitude for x of shape [B,1,H,W]. Returns [B,1,H,W]."""
    assert x.dim() == 4 and x.size(1) == 1, f"Expected [B,1,H,W], got {tuple(x.shape)}"
    kx, ky = _sobel_kernels(x.device, x.dtype)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    g = torch.sqrt(gx * gx + gy * gy + eps)
    return g


def edge_strength_map_from_teacher_logits(teacher_logits: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Edge strength map based on teacher foreground probability. Output [B,H,W]."""
    probs = softmax_probs(teacher_logits)
    p_fg = fg_prob_from_probs(probs)  # [B,1,H,W]
    g = sobel_grad_mag(p_fg, eps=eps)  # [B,1,H,W]
    return g[:, 0]  # [B,H,W]


# -------------------------
# Mask generation
# -------------------------

def _rect_from_center(H: int, W: int, cx: int, cy: int, rh: int, rw: int) -> Tuple[int, int, int, int]:
    """Return (x0,x1,y0,y1) clipped to image bounds."""
    x0 = int(cx - rh // 2)
    y0 = int(cy - rw // 2)
    x0 = max(0, min(x0, H - rh))
    y0 = max(0, min(y0, W - rw))
    x1 = x0 + rh
    y1 = y0 + rw
    return x0, x1, y0, y1


def make_rect_mask(
    B: int,
    H: int,
    W: int,
    cx: int,
    cy: int,
    rh: int,
    rw: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    per_sample: bool = False,
) -> torch.Tensor:
    """Create binary mask M in {0,1} with a zero rectangle patch.

    Returns:
        M: [B,H,W] float tensor with ones everywhere except patch region set to 0.
    """
    x0, x1, y0, y1 = _rect_from_center(H, W, cx, cy, rh, rw)

    if per_sample:
        M = torch.ones((B, H, W), device=device, dtype=dtype)
        # Same rectangle for all samples here; caller can vary cx/cy per sample if desired.
        M[:, x0:x1, y0:y1] = 0.0
        return M

    # Single mask broadcasted across batch
    M1 = torch.ones((H, W), device=device, dtype=dtype)
    M1[x0:x1, y0:y1] = 0.0
    M = M1.unsqueeze(0).repeat(B, 1, 1)
    return M


def _argmax_center(score_map: torch.Tensor) -> Tuple[int, int]:
    """Argmax center from score map [H,W]"""
    H, W = score_map.shape
    flat_idx = int(torch.argmax(score_map).item())
    cx = flat_idx // W
    cy = flat_idx % W
    return cx, cy


def generate_mask_by_action(
    action: int,
    *,
    teacher_logits: torch.Tensor,
    student_logits: Optional[torch.Tensor],
    cfg: MaskActionConfig,
) -> torch.Tensor:
    """Generate BCP-consistent mask M given an action.

    Args:
        action: int in {0,1,2,3}
        teacher_logits: logits [B,C,H,W] for the *same image* used to derive maps.
        student_logits: logits [B,C,H,W] (optional, required for action=2)
        cfg: MaskActionConfig

    Returns:
        M: [B,H,W] float tensor in {0,1}
    """
    assert action in (0, 1, 2, 3), f"action must be 0..3, got {action}"
    B, C, H, W = teacher_logits.shape
    # Use smaller rectangles for guided actions to avoid over-destroying global anatomy.
    beta_use = cfg.beta if action == 0 else getattr(cfg, 'guided_beta', cfg.beta)
    rh = max(1, int(round(H * beta_use)))
    rw = max(1, int(round(W * beta_use)))

    if action == 0:
        # Random-Rect
        cx = np.random.randint(rh // 2, H - rh // 2 + 1) if H > rh else H // 2
        cy = np.random.randint(rw // 2, W - rw // 2 + 1) if W > rw else W // 2
        return make_rect_mask(B, H, W, cx, cy, rh, rw, teacher_logits.device, per_sample=cfg.per_sample)

    probsT = softmax_probs(teacher_logits)

    if action == 1:
        # Entropy-Rect (teacher uncertainty)
        ent = entropy_map_from_probs(probsT, eps=cfg.eps)  # [B,H,W]
        score = ent.mean(dim=0) if not cfg.per_sample else ent  # [H,W] or [B,H,W]
        if cfg.per_sample:
            # per-sample: different center per sample
            Ms = []
            for b in range(B):
                cx, cy = _sample_center_topk(score[b], cfg.topk_percent)
                Ms.append(make_rect_mask(1, H, W, cx, cy, rh, rw, teacher_logits.device, per_sample=False)[0])
            return torch.stack(Ms, dim=0)
        cx, cy = _sample_center_topk(score, cfg.topk_percent)
        return make_rect_mask(B, H, W, cx, cy, rh, rw, teacher_logits.device, per_sample=False)

    if action == 2:
        # Disagree-Rect (teacher vs student)
        if student_logits is None:
            raise ValueError("student_logits is required for action=2 (disagreement)")
        probsS = softmax_probs(student_logits)
        dis = disagree_map_teacher_student(probsT, probsS, eps=cfg.eps)  # [B,H,W]
        score = dis.mean(dim=0) if not cfg.per_sample else dis
        if cfg.per_sample:
            Ms = []
            for b in range(B):
                cx, cy = _sample_center_topk(score[b], cfg.topk_percent)
                Ms.append(make_rect_mask(1, H, W, cx, cy, rh, rw, teacher_logits.device, per_sample=False)[0])
            return torch.stack(Ms, dim=0)
        cx, cy = _sample_center_topk(score[b], cfg.topk_percent)
        return make_rect_mask(B, H, W, cx, cy, rh, rw, teacher_logits.device, per_sample=False)

    # action == 3
    if action == 3:
        edge = edge_strength_map_from_teacher_logits(teacher_logits, eps=cfg.eps)  # [B,H,W]
        score = edge.mean(dim=0) if not cfg.per_sample else edge
        if cfg.per_sample:
            Ms = []
            for b in range(B):
                cx, cy = _sample_center_topk(score[b], cfg.topk_percent)
                Ms.append(make_rect_mask(1, H, W, cx, cy, rh, rw, teacher_logits.device, per_sample=False)[0])
            return torch.stack(Ms, dim=0)
        cx, cy = _sample_center_topk(score, cfg.topk_percent)
        return make_rect_mask(B, H, W, cx, cy, rh, rw, teacher_logits.device, per_sample=False)


# -------------------------
# Reward helpers
# -------------------------

def bcp_mixed_targets(
    M: torch.Tensor,
    *,
    plab_a: torch.Tensor,
    lab_a: torch.Tensor,
    lab_b: torch.Tensor,
    plab_b: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create BCP-consistent mixed targets for outward and inward branches.
    两张图用 mask 拼成了一张新的混合图，所以标签也要用同样的 mask 方式，把伪标签和真实标签拼到一起，生成混合后的标签图。

    Outward (unlabeled background + labeled patch):
        net_input_unl = uimg_a * M + img_a * (1-M)
        target_unl    = plab_a * M + lab_a * (1-M)

    Inward (labeled background + unlabeled patch):
        net_input_l = img_b * M + uimg_b * (1-M)
        target_l    = lab_b * M + plab_b * (1-M)

    Args:
        M: [B,H,W] float/bool/long (broadcastable)
        plab_a: [B,H,W] pseudo labels for uimg_a
        lab_a:  [B,H,W] GT labels for img_a
        lab_b:  [B,H,W] GT labels for img_b
        plab_b: [B,H,W] pseudo labels for uimg_b

    Returns:
        target_unl: [B,H,W]
        target_l:   [B,H,W]
    """
    if M.dtype != torch.float32 and M.dtype != torch.float16 and M.dtype != torch.float64:
        M = M.float()
    target_unl = plab_a * M + lab_a * (1.0 - M)
    target_l = lab_b * M + plab_b * (1.0 - M)
    return target_unl.long(), target_l.long()


def edge_proxy_penalty(
    student_logits: torch.Tensor,
    target_seg: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Boundary proxy penalty: L1 difference between Sobel gradient magnitudes.

    Args:
        student_logits: [B,C,H,W]
        target_seg: [B,H,W] int labels (0=bg, >0=fg)

    Returns:
        scalar tensor
    """
    probsS = softmax_probs(student_logits)
    p_fg = fg_prob_from_probs(probsS)  # [B,1,H,W]
    y_fg = (target_seg != 0).float().unsqueeze(1)  # [B,1,H,W]

    g_pred = sobel_grad_mag(p_fg, eps=eps)
    g_tgt = sobel_grad_mag(y_fg, eps=eps)

    return torch.mean(torch.abs(g_pred - g_tgt))
