import argparse
import torch


DEFAULT_PROMPT = "AIの未来について考えてください。"


def add_prompt_generation_args(parser: argparse.ArgumentParser, default_max_new_tokens: int = 10) -> None:
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt to send to the model.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=default_max_new_tokens,
        help="Maximum number of tokens to generate.",
    )


def apply_generation_defaults(model) -> None:
    for setting_name in ("temperature", "top_p", "min_p", "top_k"):
        if hasattr(model.generation_config, setting_name):
            setattr(model.generation_config, setting_name, None)


def resolve_warmup_token_id(tokenizer) -> int:
    for token_id in (tokenizer.eos_token_id, tokenizer.bos_token_id, tokenizer.unk_token_id):
        if token_id is not None:
            return int(token_id)

    warmup_tokens = tokenizer.encode("a", add_special_tokens=False)
    if not warmup_tokens:
        raise ValueError("Tokenizer could not produce a warmup token.")
    return int(warmup_tokens[0])


def build_warmup_inputs(tokenizer) -> tuple[dict[str, torch.Tensor], int]:
    warmup_token_id = resolve_warmup_token_id(tokenizer)
    return {
        "input_ids": torch.tensor([[warmup_token_id]], dtype=torch.long),
        "attention_mask": torch.ones((1, 1), dtype=torch.long),
    }, warmup_token_id


def patch_num_logits_to_keep(model) -> None:
    original_prepare = model._prepare_onnx_inputs

    def patched_prepare(use_torch, model_inputs):
        if "num_logits_to_keep" not in model_inputs:
            model_inputs["num_logits_to_keep"] = torch.tensor(1, dtype=torch.int64)
        return original_prepare(use_torch, model_inputs)

    model._prepare_onnx_inputs = patched_prepare


def print_generation_header(prompt: str, library_load_elapsed: float, model_load_elapsed: float) -> None:
    print(f"Library load time: {library_load_elapsed:.2f}s")
    print(f"Model load time: {model_load_elapsed:.2f}s")
    print()
    print(">", prompt)
    print()


def build_chat_inputs(tokenizer, prompt: str) -> dict[str, torch.Tensor]:
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer(formatted_prompt, return_tensors="pt")
