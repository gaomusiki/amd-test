from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupRMSNorm(nn.Module):
    """Group RMS Norm module
    This is a variant of RMS Norm that \
        evenly splits the hidden dimension into groups, and \
        applies root-mean-square normalization with \
            learnable scaling transformation on each i-th group individually.
    """
    
    def __init__(
        self, 
        hidden_size: int, 
        group_size: Optional[int] = None,
        eps: float = 1e-5,
        init_range: tuple = (-1.0, 1.0),
        init_seed: int = 42,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ) -> None:
        """Initialize Group RMS Norm module
        
        Args:
            hidden_size(int): hidden dimension size
            group_size(int, optional): group size, if None, then set it to hidden_size to fall back to RMSNorm
            eps(float, default = 1e-5): epsilon
            init_range(tuple, default = (-1.0, 1.0)): the range of the uniform distribution to initialize learnable scaling parameters
            init_seed(int, default = 42): seed for the initialization
            dtype(torch.dtype, default = torch.float32): parameter dtype
            device(str, default = "cpu"): parameter device
        """
        super().__init__()
        # raise NotImplementedError("TODO: Assignment1 - Task1")
        
        assert group_size is None or hidden_size % group_size == 0, "hidden_size must be divisible by group_size"
        
        self.hidden_size = hidden_size # h
        self.group_size = group_size if group_size is not None else hidden_size # gz
        self.eps = eps
        self.init_range = init_range
        self.init_seed = init_seed
        
        self.num_groups = self.hidden_size // self.group_size # ng = h // gz
        
        self.weight = nn.Parameter( # shape: (1, 1, ng, gz) for broadcasting
            torch.empty(
                (1, 1, self.num_groups, self.group_size),
                dtype=dtype,
                device=device
            )
        )
        
        self.reset_parameters()
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """The forward pass for Group RMS Norm module

        Args:
            input(torch.Tensor): input tensor, with shape: [batch_size, seq_len, hidden_size]
            
        Returns:
            output(torch.Tensor): normalized output tensor, with shape: [batch_size, seq_len, hidden_size]
        """
        # raise NotImplementedError("TODO: Assignment1 - Task1")
        
        b, s, h = input.shape
        assert h == self.hidden_size, f"hidden_size must be {self.hidden_size}, but got {h}"
        
        # split input into groups, shape: (b, s, h) => (b, s, ng, gz)
        input = input.view(b, s, self.num_groups, self.group_size)
        
        # upcast to float32
        normalized_input = input.float()
        
        # normalize input on each group of hidden dim
        group_rms_norm_factor = torch.rsqrt( # shape: (b, s, ng, 1)
            normalized_input.pow(2).mean(dim=-1, keepdim=True)
            + self.eps
        )
        normalized_input = (normalized_input * group_rms_norm_factor).to(self.weight.device)
        
        # apply learnable scaling transformation on normalized input to get output, shape: (b, s, ng, gz)
        output = normalized_input * self.weight
        
        # concatenate grouped output back, shape: (b, s, ng, gz) => (b, s, h)
        output = output.view(b, s, h).to(dtype=input.dtype, device=input.device)
        
        return output
    
    def reset_parameters(self) -> None:
        """Initialize learnable scaling parameters for Group RMS Norm from a uniform distribution"""
        # raise NotImplementedError("TODO: Assignment1 - Task1")
        
        torch.manual_seed(self.init_seed)
        nn.init.uniform_(self.weight, *self.init_range)

