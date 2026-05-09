"""
Export MTP layer for multiple model types (DeepSeek-V3, DeepSeek-V3.2, DeepSeek-R1, DeepSeek-V4, GLM4.5, GLM4.7 etc.).
The exported model can be used for speculative decoding.

Usage:
    # DeepSeek V3
    python3 export_mtp.py --input-dir /path/to/DeepSeek-V3 --output-dir /path/to/DeepSeek-V3-mtp

    # DeepSeek V3.2
    python3 export_mtp.py --input-dir /path/to/DeepSeek-V3.2 --output-dir /path/to/DeepSeek-V3.2-mtp

    # DeepSeek R1
    python3 export_mtp.py --input-dir /path/to/DeepSeek-R1 --output-dir /path/to/DeepSeek-R1-mtp

    # DeepSeek V4
    python3 export_mtp.py --input-dir /path/to/DeepSeek-V4 --output-dir /path/to/DeepSeek-V4-mtp

    # GLM4 MoE
    python3 export_mtp.py --input-dir /path/to/GLM-4.5-Air --output-dir /path/to/GLM-4.5-Air-mtp
"""
# adapted from https://github.com/sgl-project/sglang/blob/main/scripts/export_deepseek_nextn.py
import argparse
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, PretrainedConfig


def load_config(input_dir):
    try:
        return AutoConfig.from_pretrained(input_dir, trust_remote_code=True)
    except (KeyError, ValueError):
        config_path = os.path.join(input_dir, "config.json")
        with open(config_path, encoding="utf-8") as f:
            config_dict = json.load(f)

        if str(config_dict.get("model_type", "")).lower() != "deepseek_v4":
            raise

        print("AutoConfig does not support deepseek_v4 in this transformers version, falling back to PretrainedConfig.")
        return PretrainedConfig(**config_dict)


def detect_model_type(config):
    """Detect model type from config."""
    model_type = getattr(config, "model_type", "").lower()
    architectures = getattr(config, "architectures", [])

    if "deepseek" in model_type or any("deepseek" in arch.lower() for arch in architectures):
        if "v4" in model_type:
            return "deepseek_v4"
        elif hasattr(config, "index_head_dim") or hasattr(config, "index_n_heads") or hasattr(config, "index_topk"):
            return "deepseek_v32"
        else:
            return "deepseek_v3"

    if "glm4" in model_type.lower() or any("glm4" in arch.lower() for arch in architectures):
        if hasattr(config, "n_routed_experts") and getattr(config, "n_routed_experts", 0) > 0:
            return "glm4_moe"
        else:
            return "glm4"

    if model_type:
        return model_type

    raise ValueError(f"Unable to detect model type from config. model_type={model_type}, architectures={architectures}")


def get_mtp_model_type(model_type):
    """Get the MTP model type name for the output config."""
    mapping = {
        "deepseek_v3": "deepseek_v3_mtp",
        "deepseek_v32": "deepseek_v32_mtp",
        "deepseek_v4": "deepseek_v4_mtp",
        "glm4_moe": "glm4_moe_mtp",
        "glm_moe_dsa": "glm_moe_dsa_mtp",
    }
    return mapping.get(model_type, f"{model_type}_mtp")


def get_mtp_architecture(model_type):
    """Get the architecture name for the output config."""
    mapping = {
        "deepseek_v3": "DeepseekMTPForCausalLM",
        "deepseek_v32": "DeepseekV32MtpForCausalLM",
        "deepseek_v4": "DeepseekV4MtpForCausalLM",
        "glm4_moe": "Glm4MoeMtpForCausalLM",
        "glm_moe_dsa": "GlmMoeDsaMtpForCausalLM",
    }
    return mapping.get(model_type, "MtpForCausalLM")


def update_and_save_config(config, output_dir, model_type):
    """Update and save config for MTP model."""
    new_config = config.to_dict()
    mtp_model_type = get_mtp_model_type(model_type)
    mtp_architecture = get_mtp_architecture(model_type)

    updates = {
        "num_hidden_layers": 1,
        "architectures": [mtp_architecture],
        "model_type": mtp_model_type,
        "quantization_config": "",
    }

    updates["first_k_dense_replace"] = 0

    new_config.update(updates)

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(new_config, f, indent=2, ensure_ascii=False, sort_keys=True)


def copy_non_safetensors_files(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        src_file_path = os.path.join(input_dir, filename)
        if (
            os.path.isfile(src_file_path)
            and not filename.endswith(".safetensors")
            and not filename.endswith(".safetensors.index.json")
        ):
            dst_file_path = os.path.join(output_dir, filename)
            shutil.copy2(src_file_path, dst_file_path)
    print(f"All non-safetensors files have been copied to {output_dir}")


def block_dequant(
    x_q_block: torch.Tensor,
    x_s: torch.Tensor,
    block_size: list[int],
) -> torch.Tensor:
    """This function conducts block-wise dequantization.
    The inputs are block-wise quantization tensor `x_q_block`,
    block-wise quantization scale and the block size.
    The outputs are dequantized tensor.
    """
    block_n, block_k = block_size[0], block_size[1]
    n, k = x_q_block.shape
    n_tiles = (n + block_n - 1) // block_n
    k_tiles = (k + block_k - 1) // block_k
    assert n_tiles == x_s.shape[0]
    assert k_tiles == x_s.shape[1]

    x_dq_block = x_q_block.to(torch.float32)

    for i in range(k_tiles):
        for j in range(n_tiles):
            x_dq_block[
                j * block_n:min((j + 1) * block_n, n),
                i * block_k:min((i + 1) * block_k, k),
            ] *= x_s[j][i]

    return x_dq_block.to(torch.bfloat16)


def normalize_deepseek_v4_mtp_key(key):
    if key == "rot.weight":
        return "model.rot.weight"

    bare_shared_mapping = {
        "embed.": "model.embed_tokens.",
        "embed_tokens.": "model.embed_tokens.",
        "emb.tok_emb.": "model.embed_tokens.",
        "norm.": "model.norm.",
        "head.": "model.head.",
        "shared_head.norm.": "model.norm.",
        "shared_head.head.": "model.head.",
    }
    for old_prefix, new_prefix in bare_shared_mapping.items():
        if key.startswith(old_prefix):
            return key.replace(old_prefix, new_prefix, 1)

    if key in ["hc_head_fn", "hc_head_base", "hc_head_scale"]:
        return f"model.layers.0.{key}"

    prefix = "mtp.0."
    if not key.startswith(prefix):
        return None

    name = key[len(prefix):]
    mtp_shared_mapping = {
        "embed.": "model.embed_tokens.",
        "embed_tokens.": "model.embed_tokens.",
        "emb.tok_emb.": "model.embed_tokens.",
        "norm.": "model.norm.",
        "head.": "model.head.",
        "shared_head.norm.": "model.norm.",
        "shared_head.head.": "model.head.",
    }
    for old_prefix, new_prefix in mtp_shared_mapping.items():
        if name.startswith(old_prefix):
            return name.replace(old_prefix, new_prefix, 1)

    return f"model.layers.0.{name}"


def is_deepseek_v4_mtp_key(key):
    return normalize_deepseek_v4_mtp_key(key) is not None


def export_mtp_layer_parameters(input_dir, output_dir, config, model_type):
    """Export MTP layer parameters for the specified model type."""
    if model_type == "deepseek_v4":
        prefix = None
    else:
        if not hasattr(config, "num_hidden_layers"):
            raise ValueError("'num_hidden_layers' not found in model config.")
        prefix = f"model.layers.{config.num_hidden_layers}."

    output_path = os.path.join(output_dir, "mtp_layer_parameters.safetensors")
    params = {}

    for filename in os.listdir(input_dir):
        if not filename.endswith(".safetensors"):
            continue

        file_path = os.path.join(input_dir, filename)
        print(f"Processing: {filename}")

        try:
            with safe_open(file_path, framework="pt") as f:
                if model_type == "deepseek_v4":
                    matching_keys = [k for k in f.keys() if is_deepseek_v4_mtp_key(k)]
                    no_match_message = "  No DeepSeek V4 MTP parameters found"
                else:
                    matching_keys = [k for k in f.keys() if (k.startswith(prefix) or k == "rot.weight")]
                    no_match_message = f"  No parameters starting with '{prefix}' found"

                if not matching_keys:
                    print(no_match_message)
                    continue

                for key in matching_keys:
                    if model_type == "deepseek_v4":
                        new_key = normalize_deepseek_v4_mtp_key(key)
                    elif key == "rot.weight":
                        new_key = "model.rot.weight"
                    elif any(special in key for special in ["embed_tokens", "shared_head", "enorm", "hnorm", "eh_proj"]):
                        new_key = key.replace(prefix, "model")
                    else:
                        new_key = key.replace(prefix, "model.layers.0")
                    params[new_key] = f.get_tensor(key)

        except Exception as e:
            print(f"  Error processing {filename}: {str(e)}")

    if params:
        new_params = {}
        for key, w_tensor in params.items():
            if "weight_scale_inv" in key and model_type in ["deepseek_v3", "deepseek_v32", "deepseek_v4"]:
                weight_scale = w_tensor
                weight_key = key.replace("weight_scale_inv", "weight")
                if weight_key in params:
                    weight = params[weight_key]
                    weight = block_dequant(weight, weight_scale, [128, 128])
                    new_params[weight_key] = weight
            elif key not in new_params:
                new_params[key] = params[key]
        params = new_params
        print(f"Saving {len(params)} parameters to {output_path}")
        save_file(params, output_path)
    else:
        print("No matching parameters found.")
        raise ValueError("No MTP layer parameters found")

    index_path = os.path.join(output_dir, "model.safetensors.index.json")
    print(f"Updating safetensors index to {index_path}")
    index_data = {"weight_map": {}}
    for key in params:
        index_data["weight_map"][key] = "mtp_layer_parameters.safetensors"
    with open(index_path, "w") as f:
        json.dump(index_data, f, indent=4)

    print("All done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export MTP layer parameters for multiple model types (DeepSeek-V3, DeepSeek-V3.2, DeepSeek-R1, DeepSeek-V4, GLM4, etc.)"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Input HuggingFace model directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output MTP model directory.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        help="Model type (deepseek_v3, deepseek_v32, deepseek_v4, glm4_moe, glm_moe_dsa). If not specified, will auto-detect. Note: DeepSeek V3 and R1 use 'deepseek_v3', V3.2 uses 'deepseek_v32', V4 uses 'deepseek_v4'.",
    )
    args = parser.parse_args()

    config = load_config(args.input_dir)

    if args.model_type:
        model_type = args.model_type.lower()
    else:
        model_type = detect_model_type(config)

    print(f"Detected model type: {model_type}")

    if not hasattr(config, "num_nextn_predict_layers"):
        raise ValueError("Model does not have 'num_nextn_predict_layers' attribute. This model may not support MTP.")
    if config.num_nextn_predict_layers != 1:
        raise ValueError(f"Only 1 MTP layer is supported, but found {config.num_nextn_predict_layers}.")

    os.makedirs(args.output_dir, exist_ok=True)

    copy_non_safetensors_files(args.input_dir, args.output_dir)

    update_and_save_config(config, args.output_dir, model_type)

    export_mtp_layer_parameters(args.input_dir, args.output_dir, config, model_type)

    print(f"\nMTP model exported successfully to: {args.output_dir}")
