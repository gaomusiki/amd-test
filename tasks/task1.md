### Task 1: Offline Sliding-Window Attention (60 points)

#### TODO

You are required to implement a pytorch module named `OfflineSlidingWindowAttn` in `src/modeling/attention.py`.


#### Explanation

* 

#### Summary

In summary, 


#### Notice

* 


#### References

*Hints: Here're some references which may be helpful to your task, or just deepen / broaden your knowledge to attention layers particularly in transformer:*

**!! Remember: it is a fundemental and essential capability to search, read, think and learn from the paper, source code, and official documentation for your answer, try NOT to rely too much on some biased and superficial blogs, e.g. CSDN !!**


* [Llama Attention Layer](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L277)
* [Google MHA paper](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
* [Google MQA paper](https://arxiv.org/pdf/1911.02150)
* [Google GQA paper](https://arxiv.org/pdf/2305.13245)
* [Pytorch SDPA Functional](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html#torch.nn.functional.scaled_dot_product_attention)
* [Flash Attention 2 Paper](https://arxiv.org/pdf/2307.08691.pdf)
* [Flash Attention Interface](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_interface.py)
* [Pytorch FlexAttention Functional](https://pytorch.org/docs/main/nn.attention.flex_attention.html#module-torch.nn.attention.flex_attention)
* [Nvidia Methods of improving LLM training stability](https://arxiv.org/pdf/2410.16682)
* [Pytorch Repeat Interleave Functional](https://pytorch.org/docs/stable/generated/torch.repeat_interleave.html#torch.repeat_interleave)