# =========================
# PATCH for 2D/3D compat
# File: bcp_rl_routeA_utils.py
# =========================

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class MaskActionConfig:
    beta: float = 2.0 / 3.0
    guided_beta: float = 0.50
    # --- optional 3D controls (safe defaults: follow beta / guided_beta) ---
    beta_z: Optional[float] = None
    guided_beta_z: Optional[float] = None

    topk_percent: float = 0.02
    per_sample: bool = True
    eps: float = 1e-6


def _sample_center_topk(score_map: torch.Tensor, topk_percent: float = 0.02) -> Tuple[int, int]:
    """2D: score_map [H,W] -> (cx,cy)"""
    H, W = score_map.shape
    HW = H * W
    k = max(1, int(round(float(topk_percent) * HW)))
    flat = score_map.reshape(-1)

    if (not torch.isfinite(flat).all()) or (float(flat.max().item()) == float(flat.min().item())):
        ridx = int(torch.randint(0, HW, (1,), device=score_map.device).item())
        return ridx // W, ridx % W

    topk = torch.topk(flat, k=k, largest=True).indices
    pick = int(torch.randint(0, k, (1,), device=score_map.device).item())
    ridx = int(topk[pick].item())
    return ridx // W, ridx % W


def _sample_center_topk_3d(score_map: torch.Tensor, topk_percent: float = 0.02) -> Tuple[int, int, int]:
    """3D: score_map [D,H,W] -> (cz,cx,cy)"""
    D, H, W = score_map.shape
    DHW = D * H * W
    k = max(1, int(round(float(topk_percent) * DHW)))
    flat = score_map.reshape(-1)

    if (not torch.isfinite(flat).all()) or (float(flat.max().item()) == float(flat.min().item())):
        ridx = int(torch.randint(0, DHW, (1,), device=score_map.device).item())
        cz = ridx // (H * W)
        rem = ridx % (H * W)
        cx = rem // W
        cy = rem % W
        return cz, cx, cy

    topk = torch.topk(flat, k=k, largest=True).indices
    pick = int(torch.randint(0, k, (1,), device=score_map.device).item())
    ridx = int(topk[pick].item())
    cz = ridx // (H * W)
    rem = ridx % (H * W)
    cx = rem // W
    cy = rem % W
    return cz, cx, cy


def softmax_probs(logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits, dim=1)


def entropy_map_from_probs(probs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """2D: probs [B,C,H,W] -> [B,H,W]
       3D: probs [B,C,D,H,W] -> [B,D,H,W]
    """
    p = torch.clamp(probs, eps, 1.0)
    ent = -(p * torch.log(p)).sum(dim=1)
    return ent


def fg_prob_from_probs(probs: torch.Tensor, bg_channel: int = 0) -> torch.Tensor:
    """2D: [B,C,H,W] -> [B,1,H,W]
       3D: [B,C,D,H,W] -> [B,1,D,H,W]
    """
    p_bg = probs[:, bg_channel:bg_channel + 1]
    return 1.0 - p_bg


def disagree_map_teacher_student(p_teacher: torch.Tensor, p_student: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """2D: -> [B,H,W], 3D: -> [B,D,H,W]"""
    pT = torch.clamp(p_teacher, eps, 1.0)
    pS = torch.clamp(p_student, eps, 1.0)
    kl = (pT * (torch.log(pT) - torch.log(pS))).sum(dim=1)
    return kl


def _sobel_kernels(device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    kx = torch.tensor([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], device=device, dtype=dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]], device=device, dtype=dtype).view(1, 1, 3, 3)
    return kx, ky


def grad_mag(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Gradient magnitude for:
       - 2D: x [B,1,H,W] (Sobel)
       - 3D: x [B,1,D,H,W] (finite differences)
       returns same shape as x
    """
    if x.dim() == 4:
        assert x.size(1) == 1, f"Expected [B,1,H,W], got {tuple(x.shape)}"
        kx, ky = _sobel_kernels(x.device, x.dtype)
        gx = F.conv2d(x, kx, padding=1)
        gy = F.conv2d(x, ky, padding=1)
        return torch.sqrt(gx * gx + gy * gy + eps)

    if x.dim() == 5:
        assert x.size(1) == 1, f"Expected [B,1,D,H,W], got {tuple(x.shape)}"
        # forward differences, then pad to keep shape
        dz = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
        dy = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
        dx = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]

        dz = F.pad(dz, (0, 0, 0, 0, 0, 1))  # pad D-right
        dy = F.pad(dy, (0, 0, 0, 1, 0, 0))  # pad H-right
        dx = F.pad(dx, (0, 1, 0, 0, 0, 0))  # pad W-right

        return torch.sqrt(dx * dx + dy * dy + dz * dz + eps)

    raise ValueError(f"grad_mag expects 4D or 5D tensor, got dim={x.dim()}")


def edge_strength_map_from_teacher_logits(teacher_logits: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """2D: logits [B,C,H,W] -> [B,H,W]
       3D: logits [B,C,D,H,W] -> [B,D,H,W]
    """
    probs = softmax_probs(teacher_logits)
    p_fg = fg_prob_from_probs(probs)          # [B,1,...]
    g = grad_mag(p_fg, eps=eps)               # [B,1,...]
    return g[:, 0]                            # [B,...]


def _rect_from_center(H: int, W: int, cx: int, cy: int, rh: int, rw: int) -> Tuple[int, int, int, int]:
    x0 = int(cx - rh // 2)
    y0 = int(cy - rw // 2)
    x0 = max(0, min(x0, H - rh))
    y0 = max(0, min(y0, W - rw))
    x1 = x0 + rh
    y1 = y0 + rw
    return x0, x1, y0, y1


def make_rect_mask(
    B: int, H: int, W: int, cx: int, cy: int, rh: int, rw: int,
    device: torch.device, dtype: torch.dtype = torch.float32, per_sample: bool = False
) -> torch.Tensor:
    x0, x1, y0, y1 = _rect_from_center(H, W, cx, cy, rh, rw)
    M1 = torch.ones((H, W), device=device, dtype=dtype)
    M1[x0:x1, y0:y1] = 0.0
    return M1.unsqueeze(0).repeat(B, 1, 1)


def _box_from_center(D: int, H: int, W: int, cz: int, cx: int, cy: int, rd: int, rh: int, rw: int):
    z0 = int(cz - rd // 2)
    x0 = int(cx - rh // 2)
    y0 = int(cy - rw // 2)

    z0 = max(0, min(z0, D - rd))
    x0 = max(0, min(x0, H - rh))
    y0 = max(0, min(y0, W - rw))

    z1 = z0 + rd
    x1 = x0 + rh
    y1 = y0 + rw
    return z0, z1, x0, x1, y0, y1


def make_box_mask(
    B: int, D: int, H: int, W: int, cz: int, cx: int, cy: int, rd: int, rh: int, rw: int,
    device: torch.device, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    z0, z1, x0, x1, y0, y1 = _box_from_center(D, H, W, cz, cx, cy, rd, rh, rw)
    M1 = torch.ones((D, H, W), device=device, dtype=dtype)
    M1[z0:z1, x0:x1, y0:y1] = 0.0
    return M1.unsqueeze(0).repeat(B, 1, 1, 1)


def generate_mask_by_action(
    action: int, *, teacher_logits: torch.Tensor, student_logits: Optional[torch.Tensor], cfg: MaskActionConfig
) -> torch.Tensor:
    """2D: teacher_logits [B,C,H,W] -> M [B,H,W]
       3D: teacher_logits [B,C,D,H,W] -> M [B,D,H,W]
    """
    assert action in (0, 1, 2, 3), f"action must be 0..3, got {action}"

    if teacher_logits.dim() == 4:
        B, C, H, W = teacher_logits.shape
        beta_use = cfg.beta if action == 0 else getattr(cfg, "guided_beta", cfg.beta)
        rh = max(1, int(round(H * beta_use)))
        rw = max(1, int(round(W * beta_use)))

        if action == 0:
            cx = np.random.randint(rh // 2, H - rh // 2 + 1) if H > rh else H // 2
            cy = np.random.randint(rw // 2, W - rw // 2 + 1) if W > rw else W // 2
            return make_rect_mask(B, H, W, cx, cy, rh, rw, teacher_logits.device)

        probsT = softmax_probs(teacher_logits)

        if action == 1:
            ent = entropy_map_from_probs(probsT, eps=cfg.eps)     # [B,H,W]
            if cfg.per_sample:
                Ms = []
                for b in range(B):
                    cx, cy = _sample_center_topk(ent[b], cfg.topk_percent)
                    Ms.append(make_rect_mask(1, H, W, cx, cy, rh, rw, teacher_logits.device)[0])
                return torch.stack(Ms, dim=0)
            score = ent.mean(dim=0)                               # [H,W]
            cx, cy = _sample_center_topk(score, cfg.topk_percent)
            return make_rect_mask(B, H, W, cx, cy, rh, rw, teacher_logits.device)

        if action == 2:
            if student_logits is None:
                raise ValueError("student_logits is required for action=2 (disagreement)")
            probsS = softmax_probs(student_logits)
            dis = disagree_map_teacher_student(probsT, probsS, eps=cfg.eps)   # [B,H,W]
            if cfg.per_sample:
                Ms = []
                for b in range(B):
                    cx, cy = _sample_center_topk(dis[b], cfg.topk_percent)
                    Ms.append(make_rect_mask(1, H, W, cx, cy, rh, rw, teacher_logits.device)[0])
                return torch.stack(Ms, dim=0)
            score = dis.mean(dim=0)
            cx, cy = _sample_center_topk(score, cfg.topk_percent)  # <-- 修复原来用到未定义 b 的 bug
            return make_rect_mask(B, H, W, cx, cy, rh, rw, teacher_logits.device)

        # action == 3
        edge = edge_strength_map_from_teacher_logits(teacher_logits, eps=cfg.eps)  # [B,H,W]
        if cfg.per_sample:
            Ms = []
            for b in range(B):
                cx, cy = _sample_center_topk(edge[b], cfg.topk_percent)
                Ms.append(make_rect_mask(1, H, W, cx, cy, rh, rw, teacher_logits.device)[0])
            return torch.stack(Ms, dim=0)
        score = edge.mean(dim=0)
        cx, cy = _sample_center_topk(score, cfg.topk_percent)
        return make_rect_mask(B, H, W, cx, cy, rh, rw, teacher_logits.device)

    # -------------------------
    # 3D branch
    # -------------------------
    if teacher_logits.dim() == 5:
        B, C, D, H, W = teacher_logits.shape
        beta_xy = cfg.beta if action == 0 else getattr(cfg, "guided_beta", cfg.beta)
        beta_z = cfg.beta_z if cfg.beta_z is not None else beta_xy
        if action != 0 and cfg.guided_beta_z is not None:
            beta_z = cfg.guided_beta_z

        rd = max(1, int(round(D * beta_z)))
        rh = max(1, int(round(H * beta_xy)))
        rw = max(1, int(round(W * beta_xy)))

        if action == 0:
            cz = np.random.randint(rd // 2, D - rd // 2 + 1) if D > rd else D // 2
            cx = np.random.randint(rh // 2, H - rh // 2 + 1) if H > rh else H // 2
            cy = np.random.randint(rw // 2, W - rw // 2 + 1) if W > rw else W // 2
            return make_box_mask(B, D, H, W, cz, cx, cy, rd, rh, rw, teacher_logits.device)

        probsT = softmax_probs(teacher_logits)

        if action == 1:
            ent = entropy_map_from_probs(probsT, eps=cfg.eps)       # [B,D,H,W]
            if cfg.per_sample:
                Ms = []
                for b in range(B):
                    cz, cx, cy = _sample_center_topk_3d(ent[b], cfg.topk_percent)
                    Ms.append(make_box_mask(1, D, H, W, cz, cx, cy, rd, rh, rw, teacher_logits.device)[0])
                return torch.stack(Ms, dim=0)
            score = ent.mean(dim=0)                                 # [D,H,W]
            cz, cx, cy = _sample_center_topk_3d(score, cfg.topk_percent)
            return make_box_mask(B, D, H, W, cz, cx, cy, rd, rh, rw, teacher_logits.device)

        if action == 2:
            if student_logits is None:
                raise ValueError("student_logits is required for action=2 (disagreement)")
            probsS = softmax_probs(student_logits)
            dis = disagree_map_teacher_student(probsT, probsS, eps=cfg.eps)   # [B,D,H,W]
            if cfg.per_sample:
                Ms = []
                for b in range(B):
                    cz, cx, cy = _sample_center_topk_3d(dis[b], cfg.topk_percent)
                    Ms.append(make_box_mask(1, D, H, W, cz, cx, cy, rd, rh, rw, teacher_logits.device)[0])
                return torch.stack(Ms, dim=0)
            score = dis.mean(dim=0)
            cz, cx, cy = _sample_center_topk_3d(score, cfg.topk_percent)
            return make_box_mask(B, D, H, W, cz, cx, cy, rd, rh, rw, teacher_logits.device)

        # action == 3
        edge = edge_strength_map_from_teacher_logits(teacher_logits, eps=cfg.eps)  # [B,D,H,W]
        if cfg.per_sample:
            Ms = []
            for b in range(B):
                cz, cx, cy = _sample_center_topk_3d(edge[b], cfg.topk_percent)
                Ms.append(make_box_mask(1, D, H, W, cz, cx, cy, rd, rh, rw, teacher_logits.device)[0])
            return torch.stack(Ms, dim=0)
        score = edge.mean(dim=0)
        cz, cx, cy = _sample_center_topk_3d(score, cfg.topk_percent)
        return make_box_mask(B, D, H, W, cz, cx, cy, rd, rh, rw, teacher_logits.device)

    raise ValueError(f"teacher_logits must be 4D or 5D, got {teacher_logits.dim()}D")


def edge_proxy_penalty(student_logits: torch.Tensor, target_seg: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    """2D/3D compatible boundary proxy penalty."""
    probsS = softmax_probs(student_logits)
    p_fg = fg_prob_from_probs(probsS)  # [B,1,...]
    y_fg = (target_seg != 0).float().unsqueeze(1)  # [B,1,...]
    g_pred = grad_mag(p_fg, eps=eps)
    g_tgt = grad_mag(y_fg, eps=eps)
    return torch.mean(torch.abs(g_pred - g_tgt))
