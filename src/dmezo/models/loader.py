"""HuggingFace model loaders.

Defaults to Qwen3-4B (standard transformer, Apache 2.0). Optionally supports:
    - Qwen3-8B for scale-up experiments
    - Qwen3.5-4B for Gated DeltaNet hybrid (experimental; MeZO behavior unverified)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase


SUPPORTED_MODELS = {
    "Qwen/Qwen3-4B": {"family": "qwen3", "size_b": 4.0, "fp16_gb": 8.0, "transformer": True},
    "Qwen/Qwen3-8B": {"family": "qwen3", "size_b": 8.0, "fp16_gb": 16.0, "transformer": True},
    "Qwen/Qwen3.5-4B": {"family": "qwen35", "size_b": 4.0, "fp16_gb": 8.0, "transformer": False},
    "Qwen/Qwen3.5-9B": {"family": "qwen35", "size_b": 9.0, "fp16_gb": 18.0, "transformer": False},
    "facebook/opt-1.3b": {"family": "opt", "size_b": 1.3, "fp16_gb": 2.6, "transformer": True},
}


def load_causal_lm(
    model_name: str = "Qwen/Qwen3-4B",
    *,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str | dict = "auto",
    trust_remote_code: bool = True,
    use_flash_attention: bool = True,
) -> Tuple[AutoModelForCausalLM, PreTrainedTokenizerBase]:
    """Load a causal LM and its tokenizer.

    Args:
        model_name: HuggingFace model id. Defaults to Qwen3-4B.
        dtype: Compute dtype. bfloat16 is recommended on Blackwell/A100/H100;
            float16 on older GPUs. MeZO works fine in bf16.
        device_map: HF device_map. "auto" works on single-GPU; for multi-client
            simulation we typically load the full model on one device.
        trust_remote_code: Required for some Qwen variants (custom code).
        use_flash_attention: Enable flash-attn-2 if available.

    Returns:
        ``(model, tokenizer)`` tuple.

    Note:
        For Qwen3.5 (Gated DeltaNet hybrid), MeZO's "low effective rank"
        assumption is unverified. Run sanity checks before using these for
        federated experiments.
    """
    info = SUPPORTED_MODELS.get(model_name)
    if info is None:
        print(f"[warn] Model {model_name!r} not in SUPPORTED_MODELS table; proceeding anyway.")
    elif not info["transformer"]:
        print(
            f"[warn] {model_name} is NOT a standard transformer "
            f"(family={info['family']}). MeZO behavior is unverified — run sanity check."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    attn_impl = "flash_attention_2" if use_flash_attention else "eager"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_impl,
    )
    return model, tokenizer


def load_with_lora(
    model_name: str = "Qwen/Qwen3-4B",
    *,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
    target_modules: Optional[list] = None,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str | dict = "auto",
) -> Tuple[AutoModelForCausalLM, PreTrainedTokenizerBase]:
    """Load a causal LM with LoRA adapters attached.

    Only the LoRA parameters will have ``requires_grad=True``. This means MeZO
    will only perturb LoRA params, drastically reducing memory and enabling
    multi-client simulation of 7B+ models on a single GPU.

    Args:
        model_name: HF model id.
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha (scaling).
        lora_dropout: LoRA dropout. Set to 0 for MeZO (we use inference_mode).
        target_modules: Which linear modules to apply LoRA to. If None, uses
            HF default for the model family.
        dtype: Compute dtype.
        device_map: HF device_map.

    Returns:
        ``(peft_model, tokenizer)`` tuple. ``peft_model.print_trainable_parameters()``
        is called for diagnostic.
    """
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as e:
        raise ImportError("Install `peft` to use load_with_lora") from e

    base, tokenizer = load_causal_lm(
        model_name, dtype=dtype, device_map=device_map, use_flash_attention=True
    )

    if target_modules is None:
        # Reasonable default for Qwen/Llama-style models.
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(base, peft_config)
    model.print_trainable_parameters()
    return model, tokenizer
