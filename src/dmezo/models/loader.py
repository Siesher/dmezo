"""HuggingFace model loaders.

Defaults to Qwen3-4B (standard transformer, Apache 2.0). Optionally supports:
    - Qwen3-8B for scale-up experiments (same architecture as 4B)
    - Qwen3.5-4B / Qwen3.5-4B-Base: hybrid linear-attention + full-attention
      vision-language model (architecture: `Qwen3_5ForConditionalGeneration`,
      text decoder has 3:1 ratio linear/full attention layers, plus 24-layer
      ViT). Loaded via `AutoModelForImageTextToText`; vision tower is frozen
      so MeZO only perturbs the text decoder.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

# loader_kind values:
#   "causal_lm"    — vanilla AutoModelForCausalLM (most models)
#   "vl_text_task" — AutoModelForImageTextToText, vision tower frozen, text-only fwd
SUPPORTED_MODELS = {
    "Qwen/Qwen3-4B": {"family": "qwen3", "size_b": 4.0, "fp16_gb": 8.0, "loader_kind": "causal_lm"},
    "Qwen/Qwen3-8B": {
        "family": "qwen3",
        "size_b": 8.0,
        "fp16_gb": 16.0,
        "loader_kind": "causal_lm",
    },
    "Qwen/Qwen3.5-4B": {
        "family": "qwen35",
        "size_b": 4.7,
        "fp16_gb": 10.0,
        "loader_kind": "vl_text_task",
    },
    "Qwen/Qwen3.5-4B-Base": {
        "family": "qwen35",
        "size_b": 4.7,
        "fp16_gb": 10.0,
        "loader_kind": "vl_text_task",
    },
    "Qwen/Qwen3.5-9B": {
        "family": "qwen35",
        "size_b": 9.7,
        "fp16_gb": 20.0,
        "loader_kind": "vl_text_task",
    },
    "Qwen/Qwen3.5-9B-Base": {
        "family": "qwen35",
        "size_b": 9.7,
        "fp16_gb": 20.0,
        "loader_kind": "vl_text_task",
    },
    "facebook/opt-1.3b": {
        "family": "opt",
        "size_b": 1.3,
        "fp16_gb": 2.6,
        "loader_kind": "causal_lm",
    },
}


def load_causal_lm(
    model_name: str = "Qwen/Qwen3-4B",
    *,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str | dict = "auto",
    trust_remote_code: bool = True,
    use_flash_attention: bool = True,
) -> tuple[nn.Module, PreTrainedTokenizerBase]:
    """Load a model + tokenizer ready for MeZO fine-tuning.

    Dispatches based on ``SUPPORTED_MODELS[model_name]["loader_kind"]``:
        - ``"causal_lm"`` (default): vanilla AutoModelForCausalLM.
        - ``"vl_text_task"``: AutoModelForImageTextToText with vision tower
          frozen. Returns the full V-L model so HF forward signature works;
          vision params have requires_grad=False so MeZO only perturbs the
          text decoder.

    Args:
        model_name: HuggingFace model id. Defaults to Qwen3-4B.
        dtype: Compute dtype. bfloat16 is recommended on Blackwell/A100/H100;
            float16 on older GPUs. MeZO works fine in bf16.
        device_map: HF device_map. "auto" works on single-GPU.
        trust_remote_code: Required for some Qwen variants (custom code).
        use_flash_attention: Enable flash-attn-2 if available (some V-L models
            do not yet support FA2 for the text branch — set False if it fails).

    Returns:
        ``(model, tokenizer)`` tuple.

    Note:
        For Qwen3.5 (hybrid linear/full-attention vision-language), MeZO's
        "low effective rank" assumption is unverified — that's the whole
        point of running it.
    """
    info = SUPPORTED_MODELS.get(model_name, {})
    if not info:
        print(
            f"[warn] Model {model_name!r} not in SUPPORTED_MODELS; defaulting to causal_lm loader."
        )
    loader_kind = info.get("loader_kind", "causal_lm")

    if loader_kind == "vl_text_task":
        return _load_vl_for_text_task(
            model_name,
            dtype=dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            use_flash_attention=use_flash_attention,
        )

    # Default: standard CausalLM.
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


def _load_vl_for_text_task(
    model_name: str,
    *,
    dtype: torch.dtype,
    device_map: str | dict,
    trust_remote_code: bool,
    use_flash_attention: bool,
) -> tuple[nn.Module, PreTrainedTokenizerBase]:
    """Load a vision-language model and freeze its vision branch.

    Returns the full multimodal model with ``vision_tower`` (or equivalent)
    parameters set to requires_grad=False, so MeZO only perturbs the text
    decoder. Forward calls without ``pixel_values`` flow through the text
    path only.
    """
    from transformers import AutoModelForImageTextToText

    # Tokenizer first. AutoTokenizer normally works for Qwen3.5; for V-L models
    # that only register AutoProcessor, fall back to extracting tokenizer from it.
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    except Exception as e:
        print(
            f"[info] AutoTokenizer raised {type(e).__name__}; falling back to AutoProcessor.tokenizer"
        )
        from transformers import AutoProcessor

        proc = AutoProcessor.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        tokenizer = proc.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    attn_impl = "flash_attention_2" if use_flash_attention else "eager"

    print(f"[info] Loading V-L model {model_name} for text-only task...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_impl,
    )

    # Freeze the vision encoder so MeZO doesn't perturb its params.
    # V-L models expose the vision branch under one of these attribute paths.
    candidates = [
        ("vision_tower", lambda m: m.vision_tower),
        ("vision_model", lambda m: m.vision_model),
        ("visual", lambda m: m.visual),
        ("model.vision_tower", lambda m: m.model.vision_tower),
        ("model.vision_model", lambda m: m.model.vision_model),
        ("model.visual", lambda m: m.model.visual),
    ]
    n_frozen = 0
    frozen_path = None
    for path, accessor in candidates:
        try:
            vis = accessor(model)
        except AttributeError:
            continue
        for p in vis.parameters():
            p.requires_grad_(False)
            n_frozen += 1
        frozen_path = path
        break

    if n_frozen:
        print(f"[info] Froze {n_frozen} params at model.{frozen_path} (MeZO skips vision branch)")
    else:
        print(
            "[warn] Could not auto-detect vision encoder on the model. "
            "MeZO will perturb ALL params (including vision) — wasteful but correct."
        )

    return model, tokenizer


def load_with_lora(
    model_name: str = "Qwen/Qwen3-4B",
    *,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
    target_modules: list | None = None,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str | dict = "auto",
) -> tuple[AutoModelForCausalLM, PreTrainedTokenizerBase]:
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
