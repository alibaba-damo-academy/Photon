import math
import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Any, Dict, Callable, Union
import torch.nn.functional as F
from enum import Enum, auto
from functools import partial
from dataclasses import dataclass

def _random_mask(tensor: torch.Tensor, dim: int, keep_ratio: float) -> torch.Tensor:
    shape = [1] * tensor.ndim
    shape[dim] = tensor.shape[dim]
    keep_mask = torch.rand(shape, device=tensor.device) < keep_ratio
    return tensor * keep_mask.expand_as(tensor)


@dataclass
class AugmentationConfig:
    prob: float
    fn: Callable[[torch.Tensor], torch.Tensor]


class VolumeAugmenter:
    AUGMENTATION_PROB = 0.15
    
    AUGMENTATIONS = [
        AugmentationConfig(0.16, lambda x: x[torch.randperm(x.shape[0])]),
        AugmentationConfig(0.16, lambda x: x[torch.randperm(x.shape[0])][:, torch.randperm(x.shape[1])]),
        AugmentationConfig(0.16, partial(_random_mask, dim=0, keep_ratio=0.2)),
        AugmentationConfig(0.16, partial(_random_mask, dim=1, keep_ratio=0.1)),
        AugmentationConfig(0.16, torch.rand_like),
        AugmentationConfig(0.20, lambda x: x * 0),
    ]
    
    @staticmethod
    def data_augmentation(pixel_values_volumes: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        if torch.rand(1).item() >= VolumeAugmenter.AUGMENTATION_PROB:
            return pixel_values_volumes, False
        
        aug_type = torch.rand(1).item()
        cumulative_prob = 0.0
        
        for config in VolumeAugmenter.AUGMENTATIONS:
            cumulative_prob += config.prob
            if aug_type <= cumulative_prob:
                return config.fn(pixel_values_volumes), True
        
        return pixel_values_volumes, True

    def __call__(self, pixel_values_volumes: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        return self.data_augmentation(pixel_values_volumes)




def init_weights(module: nn.Module, activation: str = None) -> None:

    act = activation
    if act is None:
        act = infer_activation_from_parent(module) 
    
    # -------- Linear --------
    if isinstance(module, nn.Linear):
        if act in ('relu', 'leaky_relu', 'gelu'):
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5), nonlinearity='relu')
        else:
            nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    
    # -------- Conv1d/2d/3d --------
    elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        if act in ('relu', 'leaky_relu', 'gelu'):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        else:
            nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    # -------- Normalization --------
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

    else:
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()


def infer_activation_from_parent(module):
    parent = getattr(module, '_modules', None)
    return None


def get_question_mask(input_ids: torch.Tensor, vision_end_id: int, im_end_id: int) -> torch.Tensor:
    assert input_ids.dim() == 2, "input_ids should be (B, L)"
    B, L = input_ids.shape
    device = input_ids.device

    pos = torch.arange(L, device=device)            # (L,)
    is_vend = (input_ids == vision_end_id)          # (B, L)
    is_imend = (input_ids == im_end_id)             # (B, L)

    mask = torch.zeros((B, L), dtype=torch.bool, device=device)

    for b in range(B):
        vend_idx = torch.nonzero(is_vend[b], as_tuple=False).flatten()
        imend_idx = torch.nonzero(is_imend[b], as_tuple=False).flatten() 

        if vend_idx.numel() == 0:
            continue

        vend_idx, _  = torch.sort(vend_idx)
        imend_idx, _ = torch.sort(imend_idx)

        if imend_idx.numel() == 0:
            ends = torch.full_like(vend_idx, L)
        else:
            j = torch.searchsorted(imend_idx, vend_idx, right=True) 
            ends = torch.where(
                j < imend_idx.numel(),
                imend_idx[j],
                torch.full_like(j, L)
            )

        if vend_idx.numel() > 0:
            starts = vend_idx.view(-1, 1)     # (#pairs, 1)
            ends   = ends.view(-1, 1)         # (#pairs, 1)
            p      = pos.view(1, -1)          # (1, L)
            seg_mask = (p > starts) & (p < ends)   # (#pairs, L)
            mask[b] = seg_mask.any(dim=0)

    return mask


def build_its_kwargs(batch: Dict[str, Any], image_token_id, video_token_id, vision_end_id, im_end_id):
    ids = batch["input_ids"]
    question_mask = get_question_mask(ids, vision_end_id, im_end_id)

    vis_mask = (ids == image_token_id) | (ids == video_token_id)
    return {"vision_mask": vis_mask,
            "question_mask": question_mask}


def min_max_norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    x = x.to(torch.float32)
    x_min, _ = x.min(dim=dim, keepdim=True)
    x_max, _ = x.max(dim=dim, keepdim=True)
    output = (x - x_min) / (x_max - x_min + eps)
    return output



class ThresholdMLP(nn.Module):
    def __init__(self, hidden: int = 48):
        super().__init__()
        self.in_dim = 12
        self.mlp = nn.Sequential(
            nn.Linear(self.in_dim, hidden, bias=True), nn.GELU(),
            nn.Linear(hidden, 1, bias=True), nn.Sigmoid()
        )

    def get_feature(self, s, d):
        s32 = s.to(torch.float32)
        s_mean = s32.mean(dim=1, keepdim=True)
        s_std  = s32.std(dim=1, unbiased=False, keepdim=True)
        s_min  = s32.amin(dim=1, keepdim=True)
        s_max  = s32.amax(dim=1, keepdim=True)
        s_q25  = torch.quantile(s32, 0.25, dim=1, keepdim=True)
        s_med  = torch.quantile(s32, 0.50, dim=1, keepdim=True)

        d32 = torch.nan_to_num(d.to(torch.float32), nan=0.0, posinf=1e8, neginf=-1e8)

        d_mean = d32.mean(dim=1, keepdim=True)
        d_std  = d32.std(dim=1, unbiased=False, keepdim=True)
        d_min  = d32.amin(dim=1, keepdim=True)
        d_max  = d32.amax(dim=1, keepdim=True)

        d_slog = torch.sign(d32) * torch.log1p(torch.abs(d32))   # (B, L)
        d_slog_mean = d_slog.mean(dim=1, keepdim=True)
        d_slog_max  = d_slog.amax(dim=1, keepdim=True)

        feat32 = torch.cat(
            [s_mean, s_std, s_min, s_max, s_q25, s_med,   # 6
             d_mean, d_std, d_min, d_max,                 # 4
             d_slog_mean, d_slog_max],                    # 2
            dim=1
        )  # (B, 14)
        return feat32

    def forward(self, d, h_type) -> torch.Tensor:
        d = d.to(torch.float32)
        s = min_max_norm(d)                     # (B, L)
        feat32 = self.get_feature(s, d)
        out = self.mlp(feat32.to(h_type)).to(s.dtype)     # (B,1)
        return out
    
    def pdf_forward(self, d, h_type) -> torch.Tensor:
        feat32 = torch.empty(0, device='cuda', dtype=torch.float32)
        out_type = None
        for i in d:
            i = i.to(torch.float32)
            s = min_max_norm(i)
            out_type = s.dtype
            feat32=torch.cat([feat32,self.get_feature(s,i)], dim=0)
        out = self.mlp(feat32.to(h_type)).to(out_type)     # (B,1)
        return out


class _STEGather_v3_ada(torch.autograd.Function):
    eps        = 1e-6
    min_sigma  = 3e-3
    z_clip     = 1.5
    beta_const = 0.8
    use_scale  = True
    smin       = 0.5
    smax       = 2.0
    mag_mode   = 0
    zero_mean_dir = True

    gamma_dir  = 0.0
    sat_floor  = 5e-3
    strict_rules = False

    @staticmethod
    def forward(ctx, x, presence_mask, p_full, vision_mask=None):
        B, L, D = x.shape
        presence_mask = presence_mask.bool() 
        kept_per_row = presence_mask.sum(1) 
        M = max(int(kept_per_row.max().item()), 1)

        pos = (presence_mask.cumsum(1) - 1) * presence_mask 
        pos = pos.to(torch.long) 
        out = x.new_zeros((B, M, D))
        idx = pos.unsqueeze(-1).expand(-1, -1, D)
        out.scatter_add_(1, idx, x * presence_mask.unsqueeze(-1))

        if vision_mask is None:
            vision_mask = presence_mask
        else:
            vision_mask = vision_mask.bool()  

        ctx.save_for_backward(x, presence_mask, p_full, vision_mask)
        return out

    @staticmethod
    def no_strict_backward(ctx, grad_out):
        x, presence_mask, p_full, vision_mask = ctx.saved_tensors
        needs = ctx.needs_input_grad
        need_x = needs[0] if len(needs) > 0 else False
        need_p = needs[2] if len(needs) > 2 else False

        B, L, D = x.shape
        grad_x = grad_p_full = None

        if need_x or need_p:
            pos = ((presence_mask.cumsum(1) - 1) * presence_mask).to(torch.long) 
            idx = pos.unsqueeze(-1).expand(-1, -1, D)

            go32 = grad_out.to(torch.float32)
            gx32 = go32.gather(1, idx) * presence_mask.unsqueeze(-1)

            if need_x:
                grad_x = gx32.to(grad_out.dtype)

            if need_p:
                x32 = x.to(torch.float32)
                imp = (x32 * gx32).sum(dim=-1) * vision_mask.to(torch.float32)

                cnt = vision_mask.sum(dim=1, keepdim=True).clamp_min(1).to(torch.float32)
                mu  = (imp * vision_mask).sum(dim=1, keepdim=True) / cnt
                var = ((imp - mu)**2 * vision_mask).sum(dim=1, keepdim=True) / cnt
                sigma = var.sqrt().clamp_min(_STEGather_v3_ada.min_sigma)

                z   = ((imp - mu) / sigma).clamp(-_STEGather_v3_ada.z_clip, _STEGather_v3_ada.z_clip)
                rel = 0.5 + 0.5 * torch.tanh(z)

                dir_det = (0.5 - rel).detach()
                if _STEGather_v3_ada.gamma_dir > 0.0:
                    dir_term = dir_det + _STEGather_v3_ada.gamma_dir * (0.5 - rel)
                else:
                    dir_term = dir_det

                if _STEGather_v3_ada.zero_mean_dir:
                    dir_term = dir_term - (dir_term * vision_mask).sum(dim=1, keepdim=True) / cnt

                if _STEGather_v3_ada.use_scale:
                    mag = ((x32 * gx32).abs().sum(dim=-1) if _STEGather_v3_ada.mag_mode == 0
                        else gx32.abs().sum(dim=-1))
                    mag_masked = mag * vision_mask
                    row_mean = (mag_masked.sum(dim=1, keepdim=True) / cnt).clamp_min(_STEGather_v3_ada.eps)
                    scale = (mag_masked / row_mean).clamp(_STEGather_v3_ada.smin, _STEGather_v3_ada.smax).detach()
                else:
                    scale = vision_mask.to(torch.float32)

                beta = x32.new_tensor(_STEGather_v3_ada.beta_const)

                p32  = p_full.detach().to(torch.float32)
                sat  = (p32 * (1.0 - p32)).clamp_min(_STEGather_v3_ada.sat_floor)

                grad_p_full = (beta * dir_term * scale * sat)
                grad_p_full = (grad_p_full * vision_mask).to(p_full.dtype) 

        return grad_x, None, grad_p_full, None

    @staticmethod
    def strict_backward(ctx, grad_out):
        x, presence_mask, p_full, vision_mask = ctx.saved_tensors
        needs = ctx.needs_input_grad
        need_x = needs[0] if len(needs) > 0 else False
        need_p = needs[2] if len(needs) > 2 else False

        B, L, D = x.shape
        grad_x = grad_p_full = None

        if need_x or need_p:
            pos = ((presence_mask.cumsum(1) - 1) * presence_mask).to(torch.long)
            idx = pos.unsqueeze(-1).expand(-1, -1, D)

            go32 = grad_out.to(torch.float32)
            gx32 = go32.gather(1, idx) * presence_mask.unsqueeze(-1)

            if need_x:
                grad_x = gx32.to(grad_out.dtype)

            if need_p:
                x32 = x.to(torch.float32)
                vm32 = vision_mask.to(torch.float32)

                # Strict Taylor loss-increase score:
                # Delta L_j ≈ - <x_j, g_j>
                loss_increase = -(x32 * gx32).sum(dim=-1) * vm32

                cnt = vm32.sum(dim=1, keepdim=True).clamp_min(1.0)
                mu = (loss_increase * vm32).sum(dim=1, keepdim=True) / cnt
                var = ((loss_increase - mu) ** 2 * vm32).sum(dim=1, keepdim=True) / cnt
                sigma = var.sqrt().clamp_min(_STEGather_v3_ada.min_sigma)

                z = ((loss_increase - mu) / sigma).clamp(
                    -_STEGather_v3_ada.z_clip,
                    _STEGather_v3_ada.z_clip,
                )
                retain_score = 0.5 + 0.5 * torch.tanh(z)

                # Larger loss_increase -> more retention -> negative grad on p
                dir_det = (0.5 - retain_score).detach()
                if _STEGather_v3_ada.gamma_dir > 0.0:
                    dir_term = dir_det + _STEGather_v3_ada.gamma_dir * (0.5 - retain_score)
                else:
                    dir_term = dir_det

                if _STEGather_v3_ada.use_scale:
                    if _STEGather_v3_ada.mag_mode == 0:
                        mag = (x32 * gx32).abs().sum(dim=-1)
                    else:
                        mag = gx32.abs().sum(dim=-1)

                    mag_masked = mag * vm32
                    row_mean = (mag_masked.sum(dim=1, keepdim=True) / cnt).clamp_min(_STEGather_v3_ada.eps)
                    scale = (mag_masked / row_mean).clamp(
                        _STEGather_v3_ada.smin,
                        _STEGather_v3_ada.smax,
                    ).detach()
                else:
                    scale = vm32

                beta = x32.new_tensor(_STEGather_v3_ada.beta_const)

                p32 = p_full.detach().to(torch.float32)
                sat = (p32 * (1.0 - p32)).clamp_min(_STEGather_v3_ada.sat_floor)

                grad_p_full = beta * dir_term * scale * sat
                grad_p_full = (grad_p_full * vm32).to(p_full.dtype)

        return grad_x, None, grad_p_full, None

    @staticmethod
    def backward(ctx, grad_out):
        if _STEGather_v3_ada.strict_rules is True:
            return _STEGather_v3_ada.strict_backward(ctx, grad_out)
        else:
            return _STEGather_v3_ada.no_strict_backward(ctx, grad_out)
        

@torch.no_grad()
def pad_gather_nograd(x: torch.Tensor, mask: torch.Tensor):
    original_type = x.dtype
    x = x.to(torch.float32)
    B, L, D = x.shape
    pos = (mask.cumsum(1) - 1) * mask        # (B, L)
    kept_per_row = mask.sum(1)               # (B,)
    M = int(kept_per_row.max().item())
    out = x.new_zeros((B, M, D))
    idx = pos.unsqueeze(-1).expand(-1, -1, D)        # (B, L, D)
    out.scatter_add_(1, idx, x * mask.unsqueeze(-1)) 
    return out.to(original_type)


@torch.no_grad()
def prune_attention_mask_4d(
    attn_mask: torch.Tensor,      # (B, H_or_1, Lq, Lk), 
    presence_q: torch.Tensor,     # (B, Lq) bool，
    presence_k: torch.Tensor=None # (B, Lk) bool，
) -> torch.Tensor:
    assert attn_mask.dim() == 4, "attn_mask must be 4D (B,H,Lq,Lk)"
    B, H, Lq, Lk = attn_mask.shape
    if presence_k is None:
        presence_k = presence_q
    assert presence_q.shape == (B, Lq) and presence_k.shape == (B, Lk), "presence  shape not match"

    orig_dtype = attn_mask.dtype
    attn = attn_mask.reshape(B * H, Lq, Lk)

    pres_k_bh = presence_k.unsqueeze(1).expand(B, H, Lk).reshape(B * H, Lk)  # (B*H, Lk)
    attn_k = pad_gather_nograd(attn.transpose(1, 2), pres_k_bh)              # -> (B*H, Mk, Lq)
    attn_k = attn_k.transpose(1, 2)                                          # -> (B*H, Lq, Mk)

    pres_q_bh = presence_q.unsqueeze(1).expand(B, H, Lq).reshape(B * H, Lq)  # (B*H, Lq)
    attn_qk = pad_gather_nograd(attn_k, pres_q_bh)                           # -> (B*H, Mq, Mk)

    kept_q = presence_q.sum(1).max().item()
    kept_k = presence_k.sum(1).max().item()
    Mq = int(max(1, kept_q))
    Mk = int(max(1, kept_k))
    pruned = attn_qk.reshape(B, H, Mq, Mk).to(orig_dtype)
    return pruned



def proportion_loss_band_keep(u_list, tau=0.2, w=1.0,
                              dim=1, return_ratio=False):
    loss = 0.0
    layer_weights = [1.0] 
    r_min = [0.3]
    r_max = [0.8]
    ratios_out = []

    for i, (u, rmin, rmax) in enumerate(zip(u_list, r_min, r_max)):
        u = u.to(torch.float32)

        kept_prob = torch.sigmoid(u / tau)

        keep_ratio = kept_prob.mean(dim=dim)
        ratios_out.append(keep_ratio.detach())

        over  = torch.relu(keep_ratio - rmax)
        under = torch.relu(rmin - keep_ratio)

        layer_w = layer_weights[i] if i < len(layer_weights) else 1.0
        loss = loss + (over + under).mean() * layer_w

    loss = w * loss
    return (loss, ratios_out) if return_ratio else loss



class AugmentationType(Enum):
    NONE = auto()
    MISMATCH = auto()
    FLIP = auto()


_AUGMENTATION_LOSS_REGISTRY: dict[Union[AugmentationType, str], Callable] = {}


def register_augmentation_loss(*aug_types):
    def decorator(func: Callable) -> Callable:
        for aug_type in aug_types:
            _AUGMENTATION_LOSS_REGISTRY[aug_type] = func
        return func
    return decorator


@register_augmentation_loss(AugmentationType.MISMATCH, "MISMATCH")
def mismatch_loss(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    mask = labels.ne(ignore_index)
    if mask.sum() == 0:
        return logits.new_tensor(0.0)

    logits_sel = logits[mask]
    logp = F.log_softmax(logits_sel, dim=-1)
    p = logp.exp()
    entropy = -(p * logp).sum(dim=-1)
    return -entropy.mean()


@register_augmentation_loss(AugmentationType.FLIP, "FLIP")
def mask_flip_loss(logits: torch.Tensor, labels: torch.Tensor, 
                   ignore_index: int = -100, eps: float = 1e-6) -> torch.Tensor:
    mask = labels.ne(ignore_index)
    if mask.sum() == 0:
        return logits.new_tensor(0.0)

    logits_sel = logits[mask]
    labels_sel = labels[mask]

    p = F.softmax(logits_sel, dim=-1)
    p_y = p.gather(1, labels_sel.unsqueeze(1)).squeeze(1)
    return -torch.log(1.0 - p_y + eps).mean()


@register_augmentation_loss(AugmentationType.NONE, "NONE")
def standard_cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor, 
                                 vocab_size: int, ignore_index: int = -100) -> torch.Tensor:
    loss_fct = torch.nn.CrossEntropyLoss()
    return loss_fct(
        logits.view(-1, vocab_size),
        labels.view(-1).to(logits.device)
    )


def compute_augmentation_loss(logits: torch.Tensor, labels: torch.Tensor,
                              aug_type: Union[bool, str, AugmentationType],
                              vocab_size: int,
                              **kwargs) -> torch.Tensor:
    normalized_type = _normalize_aug_type(aug_type)
    
    loss_fn = _AUGMENTATION_LOSS_REGISTRY.get(normalized_type)
    if loss_fn is None:
        raise ValueError(f"Unknown augmentation type: {aug_type}")
    
    if normalized_type in (AugmentationType.NONE, "NONE"):
        return loss_fn(logits, labels, vocab_size=vocab_size, **kwargs)
    
    return loss_fn(logits, labels, **kwargs)

def _normalize_aug_type(aug_type: Union[bool, str, AugmentationType]) -> Union[AugmentationType, str]:
    if isinstance(aug_type, AugmentationType):
        return aug_type
    if aug_type is True:
        return AugmentationType.MISMATCH
    if aug_type is False:
        return AugmentationType.NONE
    if isinstance(aug_type, str):
        return aug_type.upper()
    raise ValueError(f"Invalid augmentation type: {aug_type}")