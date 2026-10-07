# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

import warnings

__all__ = [
    'flash_attention',
    'attention',
]


def _xpu_scaled_dot_product_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    dtype=torch.bfloat16,
):
    out_dtype = q.dtype
    half_dtypes = (torch.float16, torch.bfloat16)
    q = q.transpose(1, 2).to(dtype if q.dtype not in half_dtypes else q.dtype)
    k = k.transpose(1, 2).to(dtype if k.dtype not in half_dtypes else k.dtype)
    v = v.transpose(1, 2).to(dtype if v.dtype not in half_dtypes else v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if q.size(1) != k.size(1):
        if q.size(1) % k.size(1) != 0:
            raise ValueError('XPU SDPA requires query heads to be divisible by key heads')
        repeat_factor = q.size(1) // k.size(1)
        k = k.repeat_interleave(repeat_factor, dim=1)
        v = v.repeat_interleave(repeat_factor, dim=1)

    batch_size, query_heads, query_length, _ = q.shape
    key_length = k.shape[2]
    attn_mask = None
    if q_lens is not None or k_lens is not None or causal or window_size != (-1, -1):
        attn_mask = torch.ones(
            (batch_size, query_heads, query_length, key_length),
            dtype=torch.bool,
            device=q.device,
        )
        if q_lens is not None:
            query_positions = torch.arange(query_length, device=q.device)
            attn_mask &= query_positions.view(1, 1, -1, 1) < q_lens.to(q.device).view(-1, 1, 1, 1)
        if k_lens is not None:
            key_positions = torch.arange(key_length, device=q.device)
            attn_mask &= key_positions.view(1, 1, 1, -1) < k_lens.to(q.device).view(-1, 1, 1, 1)
        if causal:
            query_positions = torch.arange(query_length, device=q.device).view(-1, 1)
            key_positions = torch.arange(key_length, device=q.device).view(1, -1)
            attn_mask &= key_positions <= query_positions + key_length - query_length
        if window_size != (-1, -1):
            query_positions = torch.arange(query_length, device=q.device).view(-1, 1)
            key_positions = torch.arange(key_length, device=q.device).view(1, -1)
            if window_size[0] >= 0:
                attn_mask &= key_positions >= query_positions + key_length - query_length - window_size[0]
            if window_size[1] >= 0:
                attn_mask &= key_positions <= query_positions + key_length - query_length + window_size[1]

    out = torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=False,
        scale=softmax_scale,
    )
    return out.transpose(1, 2).contiguous().to(out_dtype)


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type in ('cuda', 'xpu') and q.size(-1) <= 256

    if q.device.type == 'xpu':
        return _xpu_scaled_dot_product_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            dtype=dtype,
        )

    if not FLASH_ATTN_2_AVAILABLE and not FLASH_ATTN_3_AVAILABLE:
        out_dtype = q.dtype
        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        attn_mask = None
        if k_lens is not None or causal:
            attn_mask = torch.ones(
                (q.size(0), 1, q.size(2), k.size(2)),
                dtype=torch.bool,
                device=q.device,
            )
            if k_lens is not None:
                key_positions = torch.arange(k.size(2), device=q.device)
                attn_mask &= key_positions.view(1, 1, 1, -1) < k_lens.to(q.device).view(-1, 1, 1, 1)
            if causal:
                attn_mask &= torch.tril(
                    torch.ones((q.size(2), k.size(2)), dtype=torch.bool, device=q.device)
                ).view(1, 1, q.size(2), k.size(2))

        out = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            scale=softmax_scale,
        )
        return out.transpose(1, 2).contiguous().to(out_dtype)

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        attn_mask = None

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
        return out
