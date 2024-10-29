import sys
import os

import pytest

import torch
from torch.testing import assert_close

from test_utils import (
    ResultCapture, score_results,
    check_if_io_meta_is_match,
    check_if_param_reset_is_fine,
)

# global test settings for all assignments
TOTAL_SCORE = 100
SCORE_FEEDBACK_FILENAME = "score.md"
ERROR_MSG_PREFIX_TO_CUTOFF = "E   "
MAX_ERROR_MSG_LENGTH = 200

# get student repo path from env
student_repo_path = os.getenv("STUDENT_REPO_PATH", None)
if student_repo_path is None:
    print("env variable `STUDENT_REPO_PATH` is not set")
    sys.exit(1)
sys.path.insert(0, os.path.abspath(os.path.join(student_repo_path)))

# import reference module
from ref.modeling import (
    DenseMLPWithLoRA as DenseMLPWithLoRARef,
    SparseMLPWithLoRA as SparseMLPWithLoRARef,
    MLPActivationType as MLPActivationTypeRef,
)

# import student's source module
from src.modeling import (
    DenseMLPWithLoRA,
    SparseMLPWithLoRA,
    MLPActivationType,
)

# constants for all toy test cases
ATOL = 1e-5
RTOL = 1e-5
SEED = 142
TIMEOUT = 10

# mapping from ref mlp_activation_type to student mlp_activation_type
mlp_activation_type_ref_to_student = {
    MLPActivationTypeRef.SIGMOID: MLPActivationType.SIGMOID,
    MLPActivationTypeRef.SILU: MLPActivationType.SILU,
    MLPActivationTypeRef.GELU: MLPActivationType.GELU,
    MLPActivationTypeRef.BILINEAR: MLPActivationType.BILINEAR,
    MLPActivationTypeRef.RELU: MLPActivationType.RELU,
}

# configs for each toy test case
score_test_cases = {
    "task1": {
        "case1": {
            "score": 10,
            
            "b": 2,
            "s": 2048,
            "h": 1024,
            "ffh": 2048,
            
            "activation_type": MLPActivationTypeRef.SILU,
            
            "r": 8,
            "alpha": None,
            "dropout": 0.0,
            
            "activation_dtype": torch.float32,
            "activation_device": "cpu",
            
            "param_dtype": torch.float32,
            "param_device": "cpu",
        },
        "case2": {
            "score": 10,
            
            "b": 1,
            "s": 4096,
            "h": 4096,
            "ffh": 6144,
            
            "activation_type": MLPActivationTypeRef.RELU,
            
            "r": 2,
            "alpha": 1.5,
            "dropout": 0.0,
            
            "activation_dtype": torch.bfloat16,
            "activation_device": "cuda",
            
            "param_dtype": torch.float32,
            "param_device": "cuda",
        },
        "case3": {
            "score": 10,
            
            "b": 4,
            "s": 512,
            "h": 2048,
            "ffh": 1024,
            
            "activation_type": MLPActivationTypeRef.GELU,
            
            "r": 1,
            "alpha": 0.1,
            "dropout": 0.1,
            
            "activation_dtype": torch.bfloat16,
            "activation_device": "cuda",
            
            "param_dtype": torch.float32,
            "param_device": "cuda",
        },
        "case4": {
            "score": 10,
            
            "b": 2,
            "s": 2048,
            "h": 1024,
            "ffh": 2048,
            
            "activation_type": MLPActivationTypeRef.BILINEAR,
            
            "r": 8,
            "alpha": 3.2,
            "dropout": 0.2,
            
            "activation_dtype": torch.bfloat16,
            "activation_device": "cpu",
            
            "param_dtype": torch.float32,
            "param_device": "cuda",
        },
    },
    "task2": {
        "case1": {
            "score": 10,
            
            "b": 2,
            "s": 2048,
            "h": 1024,
            "ffh": 2048,
            
            "activation_type": MLPActivationTypeRef.SILU,
            
            "ne": 4,
            "k": 2,
            
            "rank": 0,
            "world_size": 2,
            
            "init_mean": 0.1,
            "init_std": 1.1,
            
            "r": 8,
            "alpha": None,
            "dropout": 0.0,
            
            "activation_dtype": torch.float32,
            "activation_device": "cpu",
            
            "param_dtype": torch.float32,
            "param_device": "cpu",
        },
        "case2": {
            "score": 10,
            
            "b": 1,
            "s": 4096,
            "h": 4096,
            "ffh": 6144,
            
            "activation_type": MLPActivationTypeRef.RELU,
            
            "ne": 8,
            "k": 4,
            
            "rank": 1,
            "world_size": 2,
            
            "init_mean": 0.2,
            "init_std": 1.2,
            
            "r": 2,
            "alpha": 1.5,
            "dropout": 0.0,
            
            "activation_dtype": torch.bfloat16,
            "activation_device": "cuda",
            
            "param_dtype": torch.float32,
            "param_device": "cuda",
        },
        "case3": {
            "score": 10,
            
            "b": 4,
            "s": 512,
            "h": 2048,
            "ffh": 1024,
            
            "activation_type": MLPActivationTypeRef.GELU,
            
            "ne": 4,
            "k": 2,
            
            "rank": 2,
            "world_size": 4,
            
            "init_mean": 0.3,
            "init_std": 1.3,
            
            "r": 1,
            "alpha": 0.1,
            "dropout": 0.1,
            
            "activation_dtype": torch.bfloat16,
            "activation_device": "cuda",
            
            "param_dtype": torch.float32,
            "param_device": "cuda",
        },
        "case4": {
            "score": 10,
            
            "b": 2,
            "s": 2048,
            "h": 1024,
            "ffh": 2048,
            
            "activation_type": MLPActivationTypeRef.BILINEAR,
            
            "ne": 16,
            "k": 1,
            
            "rank": 3,
            "world_size": 4,
            
            "init_mean": 0.4,
            "init_std": 1.4,
            
            "r": 8,
            "alpha": 3.2,
            "dropout": 0.2,
            
            "activation_dtype": torch.bfloat16,
            "activation_device": "cpu",
            
            "param_dtype": torch.float32,
            "param_device": "cuda",
        },
        "case5": {
            "score": 10,
            
            "b": 2,
            "s": 2048,
            "h": 1024,
            "ffh": 2048,
            
            "activation_type": MLPActivationTypeRef.BILINEAR,
            
            "ne": 16,
            "k": 1,
            
            "rank": 5,
            "world_size": 8,
            
            "init_mean": 0.5,
            "init_std": 1.5,
            
            "r": 16,
            "alpha": None,
            "dropout": 0.15,
            
            "activation_dtype": torch.bfloat16,
            "activation_device": "cuda",
            
            "param_dtype": torch.float32,
            "param_device": "cuda",
        },
        "case6": {
            "score": 10,
            
            "b": 2,
            "s": 2048,
            "h": 1024,
            "ffh": 2048,
            
            "activation_type": MLPActivationTypeRef.BILINEAR,
            
            "ne": 16,
            "k": 2,
            
            "rank": 15,
            "world_size": 16,
            
            "init_mean": 0.6,
            "init_std": 1.6,
            
            "r": 1,
            "alpha": 0.1,
            "dropout": 0.0,
            
            "activation_dtype": torch.bfloat16,
            "activation_device": "cuda",
            
            "param_dtype": torch.float32,
            "param_device": "cuda",
        },
    },
}


@pytest.mark.parametrize(
    "case_key, case_config",
    score_test_cases["task1"].items(),
)
def test_task1(case_key, case_config):
    # set hyper parameters
    b, s, h, ffh = case_config["b"], case_config["s"], case_config["h"], case_config["ffh"]
    activation_type = case_config["activation_type"]
    r, alpha, dropout = case_config["r"], case_config.pop("alpha", None), case_config.pop("dropout", 0.0)
    init_base_seed, lora_init_base_seed, lora_dropout_seed = case_config.pop("init_base_seed", SEED), \
        case_config.pop("lora_init_base_seed", SEED), \
        case_config.pop("lora_dropout_seed", SEED)
    atol, rtol = case_config.pop("atol", ATOL), case_config.pop("rtol", RTOL)
    activation_dtype, param_dtype = case_config["activation_dtype"], case_config["param_dtype"]
    activation_device, param_device = case_config["activation_device"], case_config["param_device"]
    
    # construct the input tensor
    torch.manual_seed(init_base_seed)
    input = torch.randn(b, s, h, dtype=activation_dtype, device=activation_device)
    
    # instantiate the reference module
    dense_mlp_ref = DenseMLPWithLoRARef(
        hidden_size=h,
        ffh_size=ffh,
        activation_type=activation_type,
        init_base_seed=init_base_seed,
        lora_rank=r,
        lora_alpha=alpha,
        lora_dropout_rate=dropout,
        lora_dropout_seed=lora_dropout_seed,
        lora_init_base_seed=lora_init_base_seed,
        dtype=param_dtype,
        device=param_device,
    )
    
    # apply the forward pass to get the reference output tensor
    output_ref = dense_mlp_ref(input.clone())
    
    # instantiate the module
    dense_mlp = DenseMLPWithLoRA(
        hidden_size=h,
        ffh_size=ffh,
        activation_type=mlp_activation_type_ref_to_student[activation_type],
        init_base_seed=init_base_seed,
        lora_rank=r,
        lora_alpha=alpha,
        lora_dropout_rate=dropout,
        lora_dropout_seed=lora_dropout_seed,
        lora_init_base_seed=lora_init_base_seed,
        dtype=param_dtype,
        device=param_device,
    )
    
    # apply the forward pass
    output = dense_mlp(input)
    
    # check if the output tensor is correct
    assert_close(output, output_ref, atol=atol, rtol=rtol)
    # check if the meta attribute of outout tensor is correct
    check_if_io_meta_is_match(output, input)
    # check if the reset_parameters function works fine
    check_if_param_reset_is_fine(dense_mlp, atol=atol, rtol=rtol)
    

@pytest.mark.timeout(TIMEOUT)
@pytest.mark.parametrize(
    "case_key, case_config",
    score_test_cases["task2"].items(),
)
def test_task2(case_key, case_config):
    # set hyper parameters
    b, s, h, ffh = case_config["b"], case_config["s"], case_config["h"], case_config["ffh"]
    activation_type = case_config["activation_type"]
    ne, k, rank, world_size = case_config["ne"], case_config["k"], case_config["rank"], case_config["world_size"]
    init_mean, init_std = case_config["init_mean"], case_config["init_std"]
    r, alpha, dropout = case_config["r"], case_config.pop("alpha", None), case_config.pop("dropout", 0.0)
    init_base_seed, lora_init_base_seed, lora_dropout_seed = case_config.pop("init_base_seed", SEED), \
        case_config.pop("lora_init_base_seed", SEED), \
        case_config.pop("lora_dropout_seed", SEED)
    atol, rtol = case_config.pop("atol", ATOL), case_config.pop("rtol", RTOL)
    activation_dtype, param_dtype = case_config["activation_dtype"], case_config["param_dtype"]
    activation_device, param_device = case_config["activation_device"], case_config["param_device"]

    # construct the input tensor
    torch.manual_seed(init_base_seed + 1)
    input = torch.randn(b, s, h, dtype=activation_dtype, device=activation_device)
    
    # instantiate the reference module
    sparse_mlp_ref = SparseMLPWithLoRARef(
        hidden_size=h,
        ffh_size=ffh,
        activation_type=activation_type,
        num_experts=ne,
        moe_topk=k,
        rank=rank,
        world_size=world_size,
        init_mean=init_mean,
        init_std=init_std,
        init_base_seed=init_base_seed,
        lora_rank=r,
        lora_alpha=alpha,
        lora_dropout_rate=dropout,
        lora_dropout_seed=lora_dropout_seed,
        lora_init_base_seed=lora_init_base_seed,
        dtype=param_dtype,
        device=param_device,
    )
    
    # apply the forward pass to get the reference output tensor
    output_ref = sparse_mlp_ref(input.clone())
    
    # instantiate the module
    sparse_mlp = SparseMLPWithLoRA(
        hidden_size=h,
        ffh_size=ffh,
        activation_type=mlp_activation_type_ref_to_student[activation_type],
        num_experts=ne,
        moe_topk=k,
        rank=rank,
        world_size=world_size,
        init_mean=init_mean,
        init_std=init_std,
        init_base_seed=init_base_seed,
        lora_rank=r,
        lora_alpha=alpha,
        lora_dropout_rate=dropout,
        lora_dropout_seed=lora_dropout_seed,
        lora_init_base_seed=lora_init_base_seed,
        dtype=param_dtype,
        device=param_device,
    )
    
    # apply the forward pass
    output = sparse_mlp(input)
    
    # check if the output tensor is correct
    assert_close(output, output_ref, atol=atol, rtol=rtol)
    # check if the meta attribute of outout tensor is correct
    check_if_io_meta_is_match(output, input)
    # check if the reset_parameters function works fine
    check_if_param_reset_is_fine(sparse_mlp, atol=atol, rtol=rtol)


def main():
    capture = ResultCapture()
    
    pytest.main(
        [
            '--quiet',
            '--tb=no', 
            '--disable-warnings',
            os.path.abspath(__file__) # to filter to only execute this test file
        ], 
        plugins=[capture]
    )
    
    score_results(
        capture=capture,
        score_test_cases=score_test_cases,
        student_repo_path=student_repo_path,
    )


if __name__ == "__main__":
    main()