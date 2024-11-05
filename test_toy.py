import sys
sys.path.insert(0, ".")

from typing import List, Optional, Sequence

import pytest

import torch
from torch.testing import assert_close

from src.modeling import (
    AttnQKVPackFormat,
    AttnQKVLayout,
    OfflineSlidingWindowAttn,
    OnlineSlidingWindowAttn,
)

# constants for all toy test cases
ATOL = 1e-3
RTOL = 1e-3
SEED = 42
PARAM_DEVICE = "cpu"
PARAM_DTYPE = torch.float32

# configs for each toy test case
toy_test_cases = {
    "task1": {
        "case1": {
            "b": 1,
            "sq": 7,
            "skv": 5,
            "hq": 1,
            "hkv": 1,
            "hd": 4,
            
            "qkv_pack_format": AttnQKVPackFormat.QKV,
            "qkv_layout": AttnQKVLayout.SBHD,
            
            "seqlens_q": None,
            "seqlens_kv": None,
            
            "window_size": None,
            "causal": True,
            "softmax_scale": None,
            "softmax_cap": None,
            "softmax_temp": 0.8,
            "softmax_clip_range": (-0.03, 1.03),
            
            "group_size": 1,
            "eps": 1e-5,
            "init_range": (-1.1, 1.1),
            "init_seed": 42,
            
            "activation_dtype": torch.bfloat16,
            "activation_device": "cpu",
        },
        "case2": {
            "b": 1,
            "sq": 7,
            "skv": 5,
            "hq": 2,
            "hkv": 1,
            "hd": 4,
            
            "qkv_pack_format": AttnQKVPackFormat.Q_KV,
            "qkv_layout": AttnQKVLayout.THD,
            
            "seqlens_q": [1, 2, 4],
            "seqlens_kv": [2, 2, 1],
            
            "window_size": 2,
            "causal": False,
            "softmax_scale": None,
            "softmax_cap": 10,
            "softmax_temp": 1.0,
            "softmax_clip_range": (-0.01, 1.01),
            
            "group_size": 2,
            "eps": 1e-5,
            "init_range": (-1.2, 1.2),
            "init_seed": 42,
            
            "activation_dtype": torch.float32,
            "activation_device": "cpu",
        }
    },
    "task2": {
        "case1": {
        }
    },
}


def construct_attn_args(
    b: int,
    sq: int,
    skv: int,
    hq: int,
    hkv: int,
    hd: int,
    qkv_pack_format: AttnQKVPackFormat,
    qkv_layout: AttnQKVLayout,
    seqlens_q: Optional[List[int]] = None,
    seqlens_kv: Optional[List[int]] = None,
    dtype: torch.dtype = PARAM_DTYPE,
    device: str = PARAM_DEVICE,
    seed: int = SEED,
) -> Sequence[Optional[torch.Tensor]]:
    torch.manual_seed(seed)
    q = torch.randn((b, sq, hq, hd), dtype=dtype, device=device)
    k = torch.randn((b, skv, hkv, hd), dtype=dtype, device=device)
    v = torch.randn((b, skv, hkv, hd), dtype=dtype, device=device)
    
    if qkv_layout == AttnQKVLayout.THD:
        assert seqlens_q is not None, "THD layout requires cu_seqlens_q"
        assert seqlens_kv is not None, "THD layout requires cu_seqlens_kv"
        
        cu_seqlens_q, cu_seqlens_kv =[
            torch.concat([
                torch.zeros(1, dtype=torch.int32, device=device),
                torch.tensor(x, dtype=torch.int32, device=device).cumsum(dim=0)
            ], dim=0)
            for x in (seqlens_q, seqlens_kv)
        ]
        
        q, k, v = [
            x.view(-1, *x.shape[-2:]).contiguous() 
            for x in (q, k, v)
        ]
    else:
        assert seqlens_q is None, "QKV layout does not require cu_seqlens_q"
        assert seqlens_kv is None, "QKV layout does not require cu_seqlens_kv"
        cu_seqlens_q, cu_seqlens_kv = None, None
        
        if qkv_layout == AttnQKVLayout.SBHD:
            q, k, v = [
                x.transpose(0, 1).contiguous() 
                for x in (q, k, v)
            ]
    
    if qkv_pack_format == AttnQKVPackFormat.QKV:
        assert sq == skv, "QKV pack format requires sq == skv"
        q = torch.concat((q, k, v), dim=-2)
        k, v = None
    elif qkv_pack_format == AttnQKVPackFormat.Q_KV:
        k = torch.concat((k, v), dim=-2)
        v = None
    
    return q, k, v, cu_seqlens_q, cu_seqlens_kv


@pytest.mark.parametrize(
    "case_key, case_config",
    toy_test_cases["task1"].items(),
)
def test_task1(case_key, case_config):
    # set hyper parameters
    b, sq, skv = case_config["b"], case_config["sq"], case_config["skv"], 
    hq, hkv, hd = case_config["hq"], case_config["hkv"], case_config["hd"]
    qkv_pack_format, qkv_layout = case_config["qkv_pack_format"], case_config["qkv_layout"]
    seqlens_q, seqlens_kv = case_config["seqlens_q"], case_config["seqlens_kv"]
    w, causal = case_config["window_size"], case_config["causal"]
    softmax_scale, softmax_cap, softmax_temp, softmax_clip_range = case_config["softmax_scale"], \
        case_config["softmax_cap"], case_config["softmax_temp"], case_config["softmax_clip_range"]
    group_size, eps = case_config["group_size"], case_config["eps"]
    init_range, init_seed = case_config["init_range"], case_config.pop("init_seed", SEED)
    atol, rtol = case_config.pop("atol", ATOL), case_config.pop("rtol", RTOL)
    activation_dtype, param_dtype = case_config["activation_dtype"], case_config.pop("param_dtype", PARAM_DTYPE)
    activation_device, param_device = case_config["activation_device"], case_config.pop("param_device", PARAM_DEVICE)
    
    # construct the reference output tensor
    if case_key == "case1":
        output_ref = torch.tensor(
            [
                [[-0.6133, -0.7031,  0.4023,  0.8086],
                [-4.4375,  0.8555, -2.1875,  4.4688]]
            ],
            dtype=activation_dtype,
            device=activation_device,
        )
    elif case_key == "case2":
        output_ref = torch.tensor(
            [
                [[-0.6133, -0.7031,  0.4023,  0.8086],
                [-4.4375,  0.8555, -2.1875,  4.4688]]
            ],
            dtype=activation_dtype,
            device=activation_device,
        )
    else:
        raise ValueError(f"Unknown key for toy test cases: {case_key}")

    # construct the input tensors
    q, k, v, cu_seqlens_q, cu_seqlens_kv = construct_attn_args(
        b, sq, skv, hq, hkv, hd, 
        qkv_pack_format, qkv_layout, seqlens_q, seqlens_kv, 
        dtype=activation_dtype, device=activation_device, seed=SEED,
    )
    
    # instantiate the module
    off_swa = OfflineSlidingWindowAttn(
        head_dim=hd,
        num_q_head=hq,
        num_kv_head=hkv,
        qkv_pack_format=qkv_pack_format,
        qkv_layout=qkv_layout,
        window_size=w,
        causal=causal,
        softmax_dropout_rate=0.0,
        softmax_dropout_seed=42,
        softmax_scale=softmax_scale,
        softmax_cap=softmax_cap,
        softmax_temp=softmax_temp,
        softmax_clip_range=softmax_clip_range,
        group_size=group_size,
        eps=eps,
        init_range=init_range,
        init_seed=init_seed,
        dtype=param_dtype,
        device=param_device,
    )
    
    # apply the forward pass
    output = off_swa(
        q, k, v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_kv,
    )
    
    # check if the output tensor is correct
    assert_close(output, output_ref, atol=atol, rtol=rtol)
    

# @pytest.mark.parametrize(
#     "case_key, case_config",
#     toy_test_cases["task2"].items(),
# )
# def test_task2(case_key, case_config):
#     # set hyper parameters
#     b, s, h, ffh = case_config["b"], case_config["s"], case_config["h"], case_config["ffh"]
#     activation_type = case_config["activation_type"]
#     ne, k, rank, world_size = case_config["ne"], case_config["k"], case_config["rank"], case_config["world_size"]
#     init_mean, init_std = case_config["init_mean"], case_config["init_std"]
#     r, alpha, dropout = case_config["r"], case_config.pop("alpha", None), case_config.pop("dropout", 0.0)
#     init_base_seed, lora_init_base_seed, lora_dropout_seed = case_config.pop("init_base_seed", SEED), \
#         case_config.pop("lora_init_base_seed", SEED), \
#         case_config.pop("lora_dropout_seed", SEED)
#     atol, rtol = case_config.pop("atol", ATOL), case_config.pop("rtol", RTOL)
#     activation_dtype, param_dtype = case_config["activation_dtype"], case_config.pop("param_dtype", PARAM_DTYPE)
#     activation_device, param_device = case_config["activation_device"], case_config.pop("param_device", PARAM_DEVICE)
    
#     # construct the reference output tensor
#     if case_key == "case1":
#         output_ref = torch.tensor(
#             [
#                 [[ 3.7969, -3.1406, -4.7188,  0.0608],
#                 [ 0.4453, -0.9219, -0.1196, -1.0625],
#                 [ 2.6406, -1.2656, -1.3594,  0.9258]],
        
#                 [[ 0.7617,  0.1196, -0.6523,  1.1719],
#                 [ 0.6562, -1.2578, -0.3965, -1.3125],
#                 [ 6.7500, -3.7969, -6.7188,  2.6406]]
#             ],
#             dtype=activation_dtype,
#             device=activation_device,
#         )
#     else:
#         raise ValueError(f"Unknown key for toy test cases: {case_key}")

#     # construct the input tensor
#     torch.manual_seed(init_base_seed + 1)
#     input = torch.randn(b, s, h, dtype=dtype, device=device)
    
#     # instantiate the module
#     sparse_mlp = SparseMLPWithLoRA(
#         hidden_size=h,
#         ffh_size=ffh,
#         activation_type=activation_type,
#         num_experts=ne,
#         moe_topk=k,
#         rank=rank,
#         world_size=world_size,
#         init_mean=init_mean,
#         init_std=init_std,
#         init_base_seed=init_base_seed,
#         lora_rank=r,
#         lora_alpha=alpha,
#         lora_dropout_rate=dropout,
#         lora_dropout_seed=lora_dropout_seed,
#         lora_init_base_seed=lora_init_base_seed,
#         dtype=param_dtype,
#         device=param_device,
#     )
    
#     # apply the forward pass
#     output = sparse_mlp(input)
    
#     # check if the output tensor is correct
#     assert_close(output, output_ref, atol=atol, rtol=rtol)



if __name__ == "__main__":
    pytest.main()
    