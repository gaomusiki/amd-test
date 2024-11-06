### Task 2: Online Sliding-Window Attention (40 points)

#### TODO

You are required to implement a pytorch module named `OnlineSlidingWindowAttn` in `src/modeling/attention.py`.


#### Explanation

* Building upon the `OfflineSlidingWindowAttn` module described in [task1](./task1.md), we continue to implement the `OnlineSlidingWindowAttn` module, which is the online version of the former one, only applying attention on a block of $Q_{bq_i},K_{bkv_j},V_{bkv_j}$ in `AttnQKVLayout.BSHD` layout and `AttnQKVPackFormat.Q_K_V` packing format, and aggregate the local output $O^{(bkv_j)}_{bq_i}$ of this block to the global output $O$, with the help of `log-sum-exp`-style softmax calibration coefficient $lse$.
* To be more specific, although both the computation cost and the memory footprint of the `attention` operation generally follow the quadratic complexity, we can reduce the memory complexity to almost linear by transforming the `offline softmax` to `online softmax` (*See the Online Softmax Paper in [References](#references)*). The basic idea is to split the `sq`-dim and `skv`-dim of $Q$ and $K,V$ equally to `bq`-dim and `bkv`-dim respectively as blocks, and each time only apply attention on a single block of $Q_{bq_i},K_{bkv_j},V_{bkv_j}$, where the indices $bq_i \in [0, \frac{sq}{bq}]$, $bkv_j \in [0, \frac{skv}{bkv}]$. 
* The local attention output of this block is denoted as $O^{(bkv_j)}_{bq_i}$, with the shape `[b, bq, hq, hd]`. Give the global output buffer $O$ with the shape `[b, sq, hq, hd]`, how can we aggregate $O^{(bkv_j)}_{bq_i}$ to $O$ accurately? As the equation shown below, the key is to re-calculate the softmax weights with the new normalization factor and new maximum value.
$$
\text{softmax}(X) = \cfrac{\exp(X - \max{X})}{}
$$


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
* [Online Softmax Paper](https://arxiv.org/pdf/2112.05682)
* [LSE Wiki](https://en.wikipedia.org/wiki/LogSumExp)
* [Pytorch LSE Functional](https://pytorch.org/docs/stable/generated/torch.logsumexp.html#torch-logsumexp)