
import torch
import torch.nn as nn
import types
import torch.distributed as dist
import torch.nn.functional as F
from typing import List

def is_rank0():
    return (not dist.is_initialized()) or dist.get_rank() == 0


from .utils import _STEGather_v3_ada, pad_gather_nograd, prune_attention_mask_4d, init_weights, min_max_norm, ThresholdMLP


class AdaptiveTokenPruning_v2(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.predictor = ThresholdMLP(hidden = hidden_size // 8)
        self.pos_proj = nn.Linear(1, hidden_size, bias=False)
        self.last_p_full = None
        self.last_keep_bool = None

    def forward(
        self,
        h: torch.Tensor,              
        attn_logits: torch.Tensor,   
        vis_mask: torch.Tensor,   
    ):
        B, L, _ = h.shape
        device = h.device
        h_type = h.dtype

        d_raw = attn_logits.to(torch.float32)

        th_r = self.predictor(d_raw, h_type) * 0.9 
        s_cross = min_max_norm(d_raw)   

        delta = s_cross - (th_r).to(torch.float32)

        p_ce = torch.sigmoid(delta / 0.2)
        
        p_full = torch.zeros((B, L), dtype=p_ce.dtype, device=device)
        p_full = p_full.masked_scatter(vis_mask, p_ce)

        soft_p_full = torch.ones((B, L), dtype=p_ce.dtype, device=device)
        soft_p_full = soft_p_full.masked_scatter(vis_mask, p_ce)

        
        keep_bool = (p_full > 0.5)
        keep_bool_st = keep_bool.to(p_ce.dtype).detach() - p_full.detach() + p_full

        self.last_p_full = delta
        self.last_keep_bool = keep_bool_st

        return keep_bool_st, p_full, soft_p_full


def prune_and_assign_kv_cache_varlen(pkv, presence: torch.Tensor):

    B, L = presence.shape
    kept_per_row = presence.sum(1)                     # (B,)
    M = int(kept_per_row.max().item())
    pos = (presence.cumsum(1) - 1) * presence          # (B, L)
    pos4 = pos.view(B, 1, L, 1)   

    def gather_varlen(x: torch.Tensor) -> torch.Tensor:
        Bx, H, Lx, D = x.shape
        assert Bx == B and Lx == L
        out = x.new_zeros((B, H, M, D))
        idx = pos4.expand(B, H, L, D)
        mask4 = presence.view(B, 1, L, 1)              # (B,1,L,1)
        out.scatter_add_(2, idx, x * mask4) 
        return out.contiguous()

    pkv.key_cache  = [gather_varlen(t) for t in pkv.key_cache]
    pkv.value_cache = [gather_varlen(t) for t in pkv.value_cache]
    return pkv


def _patch_decoder_layer(layer: nn.Module, hidden_size: int):
    atp = AdaptiveTokenPruning_v2(hidden_size)
    layer.add_module("atp", atp)
    layer.atp.apply(init_weights)
    nn.init.constant_(layer.atp.predictor.mlp[2].bias, -5.0)
    layer_forward = layer.forward

    def wrapped(self,
                hidden_states,
                attention_mask=None,
                position_ids=None,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=None,
                position_embeddings=None,
                vision_mask=None,
                orig_idx=None,
                now_step=None,
                question_mask=None,
                augmentation=None,
                **kwargs):
        B, L_curr, _ = hidden_states.shape
        
        outs = layer_forward(hidden_states,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            past_key_value=past_key_value,
                            output_attentions=True,
                            use_cache=use_cache,
                            cache_position=cache_position,
                            position_embeddings=position_embeddings,
                            vision_mask=vision_mask,
                            question_mask=question_mask)
        h_out, attn_logits = outs[0], outs[1]
        pkv = outs[2] if len(outs) > 2 else None
        h_original_dtype = h_out.dtype


        if L_curr <= 1 or vision_mask.sum() == 0:
            return (h_out, None, pkv,
                    vision_mask, position_embeddings, orig_idx, attention_mask)


        keep_bool, p_full, soft_p_full = self.atp(         
                h_out, attn_logits.detach(),
                vision_mask.to(h_out.device)) 

        if now_step < 4000 and self.training: 
            return (h_out, None, pkv,
                    vision_mask, position_embeddings, orig_idx, attention_mask)
        
        if now_step < 8000 and self.training:
            if augmentation==True or augmentation=='FLIP':
                h_out = h_out * soft_p_full.unsqueeze(-1).detach()
                h_out = h_out.to(h_original_dtype)
            else:
                h_out = h_out * soft_p_full.unsqueeze(-1) 
                h_out = h_out.to(h_original_dtype)
            
            return (h_out, None, pkv,
                    vision_mask, position_embeddings, orig_idx, attention_mask)

        if not augmentation and self.training and now_step >= 9000:
            flip_aug = torch.rand(1)
            if flip_aug < 0.05:
                augmentation = 'FLIP'
                keep_bool = (keep_bool <= 0.5) & vision_mask

        presence = (keep_bool > 0.5) | (~(vision_mask.bool()))
        if self.training:
            if augmentation==True or augmentation=='FLIP':  
                h_pruned = _STEGather_v3_ada.apply(h_out, presence, p_full.detach(), vision_mask)
            else:
                h_pruned = _STEGather_v3_ada.apply(h_out, presence, p_full, vision_mask)
        else:
            h_pruned = pad_gather_nograd(h_out, presence)

        if use_cache and pkv is not None:
            pkv = prune_and_assign_kv_cache_varlen(pkv, presence)       
        
        if position_embeddings is not None:
            cos, sin = position_embeddings 
            cos_p, sin_p = [], []
            for i in range(cos.size(0)):                      # t/h/w
                cos_p.append(pad_gather_nograd(cos[i], presence))
                sin_p.append(pad_gather_nograd(sin[i], presence))
            position_embeddings = (torch.stack(cos_p, dim=0),
                                torch.stack(sin_p, dim=0))

        if attention_mask is not None and attention_mask.ndim!=4:
            attention_mask = pad_gather_nograd(
                attention_mask.unsqueeze(-1), presence
            ).squeeze(-1)
        elif attention_mask is not None and attention_mask.ndim==4:
            attention_mask = prune_attention_mask_4d(
                attention_mask, presence_q=presence, presence_k=presence
            )
        else:
            pass

        if position_ids is not None:
            posi_ids = []
            for i in range(position_ids.size(0)):  
                posi_ids.append(pad_gather_nograd(position_ids[i].unsqueeze(-1), presence).squeeze(-1))
            position_ids = torch.stack(posi_ids, dim=0)

        if cache_position is not None: 
            cache_position_in = cache_position.unsqueeze(0).unsqueeze(-1)
            mask_in = presence[0].unsqueeze(0)
            cache_position_out = pad_gather_nograd(cache_position_in, mask_in).squeeze()
            cache_position = cache_position_out

        vm_small = pad_gather_nograd(vision_mask.unsqueeze(-1),
                                     presence).squeeze(-1).bool()

        qm_small = pad_gather_nograd(question_mask.unsqueeze(-1),
                                     presence).squeeze(-1).bool()
        
        idx_small = pad_gather_nograd(orig_idx.unsqueeze(-1),
                                      presence).squeeze(-1)


        return (h_pruned, None, pkv,
                vm_small, position_embeddings, idx_small, attention_mask, position_ids, qm_small, cache_position, augmentation)

    layer.forward = types.MethodType(wrapped, layer)


def _find_layers(module: nn.Module) -> nn.ModuleList:
    queue, seen = [module], set()
    keys = ["model", "base_model", "transformer"]
    while queue:
        m = queue.pop(0)
        if id(m) in seen:
            continue
        seen.add(id(m))
        if hasattr(m, "layers") and isinstance(m.layers, nn.ModuleList):
            return m.layers
        for k in keys:
            if hasattr(m, k):
                queue.append(getattr(m, k))
    raise AttributeError("Cannot locate decoder layers")


def attach_atp_to_qwen(model: nn.Module, layer_ids: List, hidden_size: int):
    all_layers = _find_layers(model)

    layers = [all_layers[i] for i in layer_ids]

    atp_layers = layers

    for layer in layers:
        layer.self_attn.forward = layer.self_attn.atp_forward
        _patch_decoder_layer(layer, hidden_size)
    
    return model, atp_layers

