from typing import Optional, Tuple
from enum import Enum

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import GroupRMSNorm


class AttnQKVPackFormat(Enum):
    QKV = "qkv_packed"
    Q_KV = "q_kv_packed"
    Q_K_V = "q_k_v_packed"


class AttnQKVLayout(Enum):
    BSHD = "bshd"
    SBHD = "sbhd"
    THD = "thd"


class OfflineSlidingWindowAttn(nn.Module):
    """Offline Sliding-Window Attention module
    This is a generalized variant of standard self-attention equipped with the sliding-window trick \
        to make use of spatial locality in language for computational efficiency, \
        with applying other methods to improve stability.
    """
    def __init__(
        self,
        head_dim: int,
        num_q_head: int,
        num_kv_head: int,
        qkv_pack_format: AttnQKVPackFormat = AttnQKVPackFormat.Q_K_V,
        qkv_layout: AttnQKVLayout = AttnQKVLayout.BSHD,
        window_size: Optional[int] = None,
        causal: bool = False,
        softmax_dropout_rate: float = 0.0,
        softmax_dropout_seed: int = 42,
        softmax_scale: Optional[float] = None,
        softmax_cap: Optional[float] = None,
        softmax_temp: float = 1.0,
        softmax_clip_range: Tuple[float, float] = (0., 1.),
        group_size: int = 1,
        eps: float = 1e-5,
        init_range: tuple = (-1.0, 1.0),
        init_seed: int = 42,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ):
        """Initialize Offline Sliding-Window Attention module
        
        Args:
            head_dim(int): head dimension size
            num_q_head(int): number of query heads
            num_kv_head(int): number of key/value heads
            qkv_pack_format(AttnQKVPackFormat, default = "qkv_packed"): qkv packed format
            qkv_layout(AttnQKVLayout, default = "bshd"): qkv shape layout
            window_size(int, default = None): window size
            causal(bool, default = False): if True, then apply causal masking as a prior to only allow unidirectional self-attention, otherwise bidirectional
            softmax_dropout_rate(float, default = 0.0): dropout probability for the softmax probs
            softmax_dropout_seed(int, default = 42): random seed for softmax drooput
            softmax_scale(float, default = None): softmax scale factor, if None, then applying the standard value: 1/√d
            softmax_cap(float, default = None): softmax capping to control the magnitude of the logits, if None, then NO capping is applied
            softmax_temp(float, default = 1.0): softmax temperature to control the sharpness of the distribution, only apply when softmax_cap is None
            softmax_clip_range(float, default = (0.0, 1.0): the range for softmax clipping to prevent the outliers from growing further
            group_size(int): group size to split hidden size of query / key for GroupRMSNorm, to apply qk norm
            eps(float, default = 1e-5): epsilon for GroupRMSNorm, to apply qk norm
            init_range(tuple, default = (-1.0, 1.0)): the range of the initialization uniform distribution for GroupRMSNorm, to apply qk norm
            init_seed(int, default = 42): initialization seed for GroupRMSNorm, to apply qk norm
            dtype(torch.dtype, default = torch.float32): parameter dtype for GroupRMSNorm, to apply qk norm
            device(str, default = "cpu"): parameter device for GroupRMSNorm, to apply qk norm
        """
        super().__init__()
        # raise NotImplementedError("Assignment3 - Task1")
        
        assert head_dim % group_size == 0, f"The head dimension ({head_dim}) must be divisible by the group size ({group_size})"
        assert num_q_head % num_kv_head == 0, f"The number of query heads ({num_q_head}) must be divisible by the number of key/value heads ({num_kv_head})"
        assert softmax_temp > 0., "The softmax temperature must be greater than 0"
        assert softmax_cap is None or softmax_cap > 0., "The softmax capping must be greater than 0 if given"
        assert softmax_clip_range[0] < softmax_clip_range[1], "The softmax clip range must be a valid range (l,r), s.t. l < r"
        
        self.head_dim = head_dim
        self.num_q_head = num_q_head
        self.num_kv_head = num_kv_head
        
        self.qkv_pack_format = qkv_pack_format
        self.qkv_layout = qkv_layout
        
        self.window_size = window_size
        self.causal = causal
        
        self.softmax_dropout_rate = softmax_dropout_rate
        self.softmax_dropout_seed = softmax_dropout_seed
        
        self.softmax_scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(head_dim)
        self.softmax_cap = softmax_cap
        self.softmax_temp = softmax_temp
        self.softmax_clip_range = softmax_clip_range
        
        self.group_size = group_size
        self.eps = eps
        self.init_range = init_range
        self.init_seed = init_seed
        self.dtype = dtype
        self.device = device
        
        self.q_hidden_size = self.num_q_head * self.head_dim
        self.k_hidden_size = self.num_kv_head * self.head_dim
        self.kv_repeat_times = self.num_q_head // self.num_kv_head
        
        self.softmax_clip_range_len = self.softmax_clip_range[1] - self.softmax_clip_range[0]
        self.softmax_clip_range_min = self.softmax_clip_range[0]
        
        # init dropout layer
        self.softmax_dropout = nn.Dropout(softmax_dropout_rate)
        
        # init q,k,v,o reshape funtion
        # i.e. q, k, v from the original shape to "bshd", while o from "bshd" to the original shape
        if self.qkv_layout is AttnQKVLayout.BSHD:
            # (b, s, h, d) -> (b, s, h, d)
            self.qkv_reshape_func = lambda q, k, v: (q, k, v)
            # (b, s, h, d) -> (b, s, h, d)
            self.o_reshape_func = lambda o: o
        elif self.qkv_layout is AttnQKVLayout.SBHD:
            # (s, b, h, d) -> (b, s, h, d)
            self.qkv_reshape_func = lambda q, k, v: [
                x.transpose(0, 1) for x in (q, k, v)
            ]
            # (b, s, h, d) -> (s, b, h, d)
            self.o_reshape_func = lambda o: o.transpose(0, 1)
        elif self.qkv_layout is AttnQKVLayout.THD:
            # (t, h, d) -> (1, t, h, d)
            self.qkv_reshape_func = lambda q, k, v: [
                x.unsqueeze(0) for x in (q, k, v)
            ]
            # (1, t, h, d) -> (t, h, d)
            self.o_reshape_func = lambda o: o.squeeze(0)
        
        # init q,k,v,o s/nh transpose function
        # i.e. q, k, v from "bshd" to "bhsd", while o from "bhsd" to "bshd"
        self.qkv_trans_func = lambda q, k, v: [
            x.transpose(1, 2) for x in (q, k, v)
        ]
        self.o_trans_func = lambda o: o.transpose(1, 2)
        
        # init q,k norm layers and qk norm function
        self.q_norm_layer = GroupRMSNorm(
            hidden_size=self.q_hidden_size,
            group_size=self.group_size,
            eps=self.eps,
            init_range=self.init_range,
            init_seed=self.init_seed,
            dtype=self.dtype,
            device=self.device,
        )
        self.k_norm_layer = GroupRMSNorm(
            hidden_size=self.k_hidden_size,
            group_size=self.group_size,
            eps=self.eps,
            init_range=self.init_range,
            init_seed=self.init_seed,
            dtype=self.dtype,
            device=self.device,
        )
        self.qk_norm_func = lambda q, k: [ # assuming q,k already have shape: (b, s, h, d)
            norm_layer(x.view(*x.shape[:2], -1)).view(*x.shape[:2], num_head, self.head_dim)
            for x, norm_layer, num_head in zip(
                (q, k), 
                (self.q_norm_layer, self.k_norm_layer), 
                (self.num_q_head, self.num_kv_head)
            )
        ]
        
        # init qkv split function
        if self.qkv_pack_format is AttnQKVPackFormat.QKV:
            self.qkv_split_func = lambda qkv, _k, _v: (
                torch.split(
                    qkv,
                    split_size_or_sections=[self.num_q_head, self.num_kv_head, self.num_kv_head],
                    dim=-2, # nh dim
                )
            )
        elif self.qkv_pack_format is AttnQKVPackFormat.Q_KV:
            self.qkv_split_func = lambda q, kv, _v: (
                q, 
                *torch.split(
                    kv, 
                    split_size_or_sections=self.num_kv_head,
                    dim=-2 # nh dim
                )
            )
        elif self.qkv_pack_format is AttnQKVPackFormat.Q_K_V:
            self.qkv_split_func = lambda q, k, v: (q, k, v)
            
        # init kv repeat function
        if self.kv_repeat_times == 1:
            self.kv_repeat_func = lambda k, v: (k, v)
        else:
            self.kv_repeat_func = lambda k, v: [
                x.repeat_interleave(repeats=self.kv_repeat_times, dim=-2)
                for x in [k, v]
            ]
    
        # init attn forward function
        if self.qkv_layout is AttnQKVLayout.THD:
            self.attn_fwd_func = self._varlen_attn_fwd_func
        else:
            self.attn_fwd_func = self._non_varlen_attn_fwd_func
    
    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """The forward pass of Offline Sliding-Window Attention module
        
        Args:
            q(torch.Tensor): query tensor, or query-key-value packed tensor if the qkv_pack_format is "qkv_packed"
            k(Optional[torch.Tensor], default = None): key tensor, or key-value packed tensor if the qkv_pack_format is "q_kv_packed", or None if qkv_pack_format is "qkv_packed"
            v(Optional[torch.Tensor], default = None): value tensor if the qkv_pack_format is "q_k_v_packed", otherwise None
            cu_seqlens_q(Optional[torch.Tensor], default = None): cumulative sequence lengths for query tensor, with shape: [batch_size + 1, ]
            cu_seqlens_k(Optional[torch.Tensor], default = None): cumulative sequence lengths for key tensor, with shape: [batch_size + 1, ]
        Returns:
            torch.Tensor: output tensor o, with the same shape as q
        """
        # raise NotImplementedError("Assignment3 - Task1")
        
        # split q,k,v
        q, k, v = self.qkv_split_func(q, k, v)
        
        # reshape q,k,v
        q, k, v = self.qkv_reshape_func(q, k, v)
        
        # normalize q,k
        q, k = self.qk_norm_func(q, k)
        
        # repeat k,v
        k, v = self.kv_repeat_func(k, v)
        
        # apply attn forward
        o = self.attn_fwd_func(q, k, v, cu_seqlens_q, cu_seqlens_k)
        
        # reshape o
        o = self.o_reshape_func(o)
        
        return o
    
    def _attn_fwd_func(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """non-varlen attn forward function
        
        Args:
            q(torch.Tensor): query tensor, with shape: [batch_size, seq_len_q, num_head, head_dim]
            k(torch.Tensor): key tensor, with shape: [batch_size, seq_len_kv, num_head, head_dim]
            v(torch.Tensor): value tensor, with shape: [batch_size, seq_len_kv, num_head, head_dim]
            
        Returns:
            torch.Tensor: output tensor o, with shape: [batch_size, seq_len_q, num_head, head_dim]
            torch.Tensor: softmax lse, with shape: [batch_size, num_head, seq_len_q]
        """
        # transpose q,k,v from "bshd" to "bhsd"
        q, k, v = self.qkv_trans_func(q, k, v)
        
        # compute logits = q @ k.T * softmax_scale, with shape: (b, h, sq, skv)
        attn_logits = q @ k.transpose(-2, -1) * self.softmax_scale
        
        # compute softmax lse, with shape: (b, h, sq)
        softmax_lse = torch.logsumexp(attn_logits, dim=-1)
        
        # adjust logits magnitude
        if self.softmax_cap is not None: # apply softmax capping
            attn_logits = F.tanh(attn_logits / self.softmax_cap) * self.softmax_cap
        else: # apply softmax temperature
            attn_logits /= self.softmax_temp
        
        # generate and apply attn mask, with shape: (1, 1, sq, skv)
        attn_mask = self._generate_attn_mask(q, k)
        attn_logits += attn_mask
        
        # apply softmax to get probs, with shape: (b, h, sq, skv)
        attn_probs = F.softmax(attn_logits, dim=-1, dtype=torch.float32).to(q.dtype)
        
        # apply softmax clipping to prevent outlier
        attn_probs = torch.clip(
            self.softmax_clip_range_len * attn_probs + self.softmax_clip_range_min,
            min=0.0, max=1.0
        )
        
        # apply softmax dropout
        torch.manual_seed(self.softmax_dropout_seed)
        attn_probs = self.softmax_dropout(attn_probs)
        
        # compute o = att_probs @ v, with shape: (b, h, sq, d)
        o = attn_probs @ v
        
        # transpose o from "bhsd" to "bshd"
        o = self.o_trans_func(o)
        
        return o, softmax_lse
    
    def _non_varlen_attn_fwd_func(
        self,
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
    ) -> torch.Tensor:
        """non-varlen attn forward function
        
        Args:
            q(torch.Tensor): query tensor, with shape: [batch_size, seq_len_q, num_head, head_dim]
            k(torch.Tensor): key tensor, with shape: [batch_size, seq_len_kv, num_head, head_dim]
            v(torch.Tensor): value tensor, with shape: [batch_size, seq_len_kv, num_head, head_dim]
            cu_seqlens_q(torch.Tensor): cumulative sequence lengths for query tensor, with shape: [batch_size + 1, ]
            cu_seqlens_k(torch.Tensor): cumulative sequence lengths for key tensor, with shape: [batch_size + 1, ]
            
        Returns:
            torch.Tensor: output tensor o, with shape: [batch_size, seq_len_q, num_head, head_dim]
        """
        o, _ = self._attn_fwd_func(q, k, v)
        
        return o
    
    def _varlen_attn_fwd_func(
        self,
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
    ) -> torch.Tensor:
        """varlen attn forward function
        
        Args:
            q(torch.Tensor): query tensor, with shape: [1, total_seq_len_q, num_head, head_dim]
            k(torch.Tensor): key tensor, with shape: [1, total_seq_len_kv, num_head, head_dim]
            v(torch.Tensor): value tensor, with shape: [1, total_seq_len_kv, num_head, head_dim]
            cu_seqlens_q(torch.Tensor): cumulative sequence lengths for query tensor, with shape: [batch_size + 1, ]
            cu_seqlens_k(torch.Tensor): cumulative sequence lengths for key tensor, with shape: [batch_size + 1, ]
            
        Returns:
            torch.Tensor: output tensor o, with shape: [1, total_seq_len_q, num_head, head_dim]
        """
        # get batch size
        b = cu_seqlens_q.shape[0] - 1
        
        # init output buffer, with shape: (1, t, h, d)
        o = torch.zeros_like(q)
        
        # apply attn fwd for each seq in the batch
        for bi in range(b):
            # compute the [start_idx, end_idx) of q, kv for the i-th seq in the batch
            sq_si, sq_ei = cu_seqlens_q[bi], cu_seqlens_q[bi + 1]
            skv_si, skv_ei = cu_seqlens_k[bi], cu_seqlens_k[bi + 1]
            
            o[:, sq_si: sq_ei, ...].add_(
                self._attn_fwd_func(
                    q[:, sq_si: sq_ei, ...],
                    k[:, skv_si: skv_ei, ...],
                    v[:, skv_si: skv_ei, ...],
                )[0]
            )
            
        return o
        
    def _generate_attn_mask(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> torch.Tensor:
        """Generate attention mask
        
        Args:
            q(torch.Tensor): query tensor, with shape: [batch_size, seq_len_q, num_head, head_dim]
            k(torch.Tensor): key tensor, with shape: [batch_size, seq_len_kv, num_head, head_dim]
            
        Returns:
            torch.Tensor: attention mask, with shape: [1, 1, seq_len_q, seq_len_kv]
        """
        w = self.window_size
        causal = self.causal
        
        # get seqlen shape
        sq, skv = q.shape[1], k.shape[1]
        
        # init attn mask, with shape: [sq, skv]
        attn_mask = torch.zeros((sq, skv), dtype=q.dtype)

        # init q row-index and k col-index
        qi = torch.arange(sq).view(-1, 1)  # [sq, 1]
        kj = torch.arange(skv).view(1, -1)  # [1, skv]

        # compute [lb, ub) of kj for each qi
        # non causal: [i-w, i] | causal: [i-w, i+w]
        lb = torch.clamp(
            qi - (w if w is not None else qi),
            min=0
        )
        ub = torch.clamp(
            qi + 1 + (w if w is not None else skv),
            max=skv
        ) if not causal else (qi + 1)

        # fill the attn mask
        # where '0' means the position to keep,
        # while '-inf' means the position to be masked out
        attn_mask.masked_fill_(
            (kj < lb) | (kj >= ub),
            float("-inf")
        )
        
        # return with shape: (1, 1, sq, skv) to broadcast
        return attn_mask.unsqueeze(0).unsqueeze(0).to(q.device)
    

class OnlineSlidingWindowAttn(nn.Module):
    def __init__(self,):
        super().__init__()
        raise NotImplementedError("Assignment3 - Task2")
    
    def forward(self,):
        raise NotImplementedError("Assignment3 - Task2")