"""Convert explicitly downloaded Wan2.2 frontends on CPU; never download assets.

SPDX-License-Identifier: AGPL-3.0-only
Wan VAE configuration and name mapping follow the Apache-2.0 Diffusers recipe:
https://github.com/huggingface/diffusers/blob/v0.37.1/scripts/convert_wan_to_diffusers.py
The T5 mapping preserves all raw tensors and the shared embedding alias.
No quantization, dtype cast, transformer conversion, or GPU execution is performed.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import re

WAN_REPO = 'Wan-AI/Wan2.2-TI2V-5B'
WAN_REVISION = '921dbaf3f1674a56f47e83fb80a34bac8a8f203e'
RAW_FILES = {
    'vae': ('Wan2.2_VAE.pth', 2818839170,
            '20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36'),
    't5': ('models_t5_umt5-xxl-enc-bf16.pth', 11361920418,
           '7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d'),
}
TOKENIZER_FILES = {
    'special_tokens_map.json': (6623, '7b8a9f5040adb67b5805abdfd42c1f8d0f3d0e711f10726580eb3789cd0ad61d'),
    'spiece.model': (4548313, 'e3909a67b780650b35cf529ac782ad2b6b26e6d1f849d3fbb6a872905f452458'),
    'tokenizer.json': (16837417, '6e197b4d3dbd71da14b4eb255f4fa91c9c1f2068b20a2de2472967ca3d22602b'),
    'tokenizer_config.json': (61728, 'ed9a3a8b0faa71a70a32847e0435fe036e6e112d4df4edb7bb48a921e344dc05'),
}
VAE_CONFIG = {
    'base_dim': 160, 'z_dim': 48, 'is_residual': True,
    'in_channels': 12, 'out_channels': 12, 'decoder_base_dim': 256,
    'scale_factor_temporal': 4, 'scale_factor_spatial': 16, 'patch_size': 2,
    'latents_mean': [
        -.2289, -.0052, -.1323, -.2339, -.2799, .0174, .1838, .1557,
        -.1382, .0542, .2813, .0891, .1570, -.0098, .0375, -.1825,
        -.2246, -.1207, -.0698, .5109, .2665, -.2108, -.2158, .2502,
        -.2055, -.0322, .1109, .1567, -.0729, .0899, -.2799, -.1230,
        -.0313, -.1649, .0117, .0723, -.2839, -.2083, -.0520, .3748,
        .0152, .1957, .1433, -.2944, .3573, -.0548, -.1681, -.0667,
    ],
    'latents_std': [
        .4765, 1.0364, .4514, 1.1677, .5313, .4990, .4818, .5013,
        .8158, 1.0344, .5894, 1.0901, .6885, .6165, .8454, .4978,
        .5759, .3523, .7135, .6804, .5833, 1.4146, .8986, .5659,
        .7069, .5338, .4889, .4917, .4069, .4999, .6866, .4093,
        .5709, .6065, .6415, .4944, .5726, 1.2042, .5458, 1.6887,
        .3971, 1.0600, .3943, .5537, .5444, .4089, .7468, .7744,
    ],
}
T5_CONFIG = {
    'vocab_size': 256384, 'd_model': 4096, 'd_kv': 64, 'd_ff': 10240,
    'num_layers': 24, 'num_decoder_layers': 24, 'num_heads': 64,
    'relative_attention_num_buckets': 32, 'relative_attention_max_distance': 128,
    'dropout_rate': .1, 'layer_norm_epsilon': 1e-6, 'feed_forward_proj': 'gated-gelu',
    'tie_word_embeddings': False, 'pad_token_id': 0, 'eos_token_id': 1,
}


def vae_key(key: str) -> str:
    """Wan2.2 48-channel VAE renaming; values and tensor layouts stay intact."""
    for source, target in (('conv1.', 'quant_conv.'), ('conv2.', 'post_quant_conv.')):
        if key.startswith(source):
            return target + key[len(source):]
    match = re.fullmatch(r'(encoder|decoder)\.(.+)', key)
    if not match:
        raise ValueError(f'Unknown VAE key: {key}')
    tower, suffix = match.groups()
    for source, target in (('conv1.', 'conv_in.'), ('head.0.', 'norm_out.'), ('head.2.', 'conv_out.')):
        if suffix.startswith(source):
            return f'{tower}.{target}{suffix[len(source):]}'
    middle = re.fullmatch(r'middle\.([012])\.(.+)', suffix)
    if middle:
        index, suffix = middle.groups()
        prefix = f'{tower}.mid_block.'
        if index == '1':
            return prefix + 'attentions.0.' + suffix
        prefix += f'resnets.{int(index) // 2}.'
    else:
        block = re.fullmatch(r'(downsamples|upsamples)\.(\d+)\.\1\.(\d+)\.(.+)', suffix)
        if not block:
            raise ValueError(f'Unknown VAE key: {key}')
        direction, block_id, unit, suffix = block.groups()
        prefix = f'{tower}.{"down_blocks" if direction == "downsamples" else "up_blocks"}.{block_id}.'
        if suffix.startswith(('resample.', 'time_conv.')):
            return prefix + ('downsampler.' if direction == 'downsamples' else 'upsampler.') + suffix
        prefix += f'resnets.{unit}.'
    if suffix.startswith('shortcut.'):
        return prefix + 'conv_shortcut.' + suffix[len('shortcut.'):]
    residual = re.fullmatch(r'residual\.([0236])\.(gamma|weight|bias)', suffix)
    if not residual:
        raise ValueError(f'Unknown VAE key: {key}')
    component, parameter = residual.groups()
    return prefix + {'0': 'norm1', '2': 'conv1', '3': 'norm2', '6': 'conv2'}[component] + '.' + parameter


def t5_key(key: str) -> str:
    if key == 'token_embedding.weight':
        return 'shared.weight'
    if key == 'norm.weight':
        return 'encoder.final_layer_norm.weight'
    match = re.fullmatch(r'blocks\.(\d+)\.(.+)', key)
    if not match or not 0 <= int(match[1]) < 24:
        raise ValueError(f'Unknown T5 key: {key}')
    suffixes = {
        'norm1.weight': 'layer.0.layer_norm.weight', 'norm2.weight': 'layer.1.layer_norm.weight',
        **{f'attn.{p}.weight': f'layer.0.SelfAttention.{p}.weight' for p in 'qkvo'},
        'pos_embedding.embedding.weight': 'layer.0.SelfAttention.relative_attention_bias.weight',
        'ffn.gate.0.weight': 'layer.1.DenseReluDense.wi_0.weight',
        'ffn.fc1.weight': 'layer.1.DenseReluDense.wi_1.weight',
        'ffn.fc2.weight': 'layer.1.DenseReluDense.wo.weight',
    }
    if match[2] not in suffixes:
        raise ValueError(f'Unknown T5 key: {key}')
    return f'encoder.block.{match[1]}.' + suffixes[match[2]]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, expected_bytes: int, expected_sha256: str) -> None:
    if path.stat().st_size != expected_bytes or sha256_file(path) != expected_sha256:
        raise ValueError(f'Source size/SHA256 mismatch: {path.name}')


def validate_paths(raw_root: Path, output: Path) -> None:
    for path in (raw_root, output):
        if not path.is_absolute() or any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError('Use absolute direct paths without symbolic links.')
    source, target = raw_root.resolve(), output.resolve()
    if source == target or source in target.parents or target in source.parents:
        raise ValueError('Converted output must be separate from source assets.')
    if output.exists():
        raise FileExistsError('Choose a fresh output directory; existing assets are never overwritten.')


def convert_component(kind: str, source: Path, destination: Path) -> dict:
    import torch
    from accelerate import init_empty_weights
    from diffusers import AutoencoderKLWan
    from transformers import UMT5Config, UMT5EncoderModel

    dtype = torch.float32 if kind == 'vae' else torch.bfloat16
    mapper = vae_key if kind == 'vae' else t5_key
    model_class = AutoencoderKLWan if kind == 'vae' else UMT5EncoderModel
    try:
        raw = torch.load(source, map_location='cpu', weights_only=True, mmap=True)
    except RuntimeError as error:
        if 'mmap' not in str(error):
            raise
        raw = torch.load(source, map_location='cpu', weights_only=True)
    if not isinstance(raw, dict) or any(not isinstance(t, torch.Tensor) or t.dtype != dtype for t in raw.values()):
        raise ValueError(f'{kind}: expected a plain tensor state dictionary in {dtype}.')
    mapped = {mapper(key): tensor for key, tensor in raw.items()}
    if len(mapped) != len(raw):
        raise ValueError('Distinct raw keys collapsed during conversion.')
    if kind == 't5':
        mapped['encoder.embed_tokens.weight'] = mapped['shared.weight']
    with init_empty_weights(include_buffers=True):
        model = AutoencoderKLWan(**VAE_CONFIG) if kind == 'vae' else UMT5EncoderModel(UMT5Config(**T5_CONFIG))
    expected = model.state_dict()
    if set(mapped) != set(expected) or any(mapped[key].shape != expected[key].shape for key in expected):
        raise ValueError(f'{kind}: native state keys or shapes differ; conversion aborted.')
    model.load_state_dict(mapped, strict=True, assign=True)
    if kind == 't5' and model.shared.weight.data_ptr() != model.encoder.embed_tokens.weight.data_ptr():
        raise ValueError('T5 shared embedding alias was lost during assignment.')
    model.eval().requires_grad_(False)
    model.save_pretrained(destination, safe_serialization=True, max_shard_size='5GB')
    del model, expected
    gc.collect()
    reloaded, info = model_class.from_pretrained(
        destination, local_files_only=True, torch_dtype=dtype, output_loading_info=True,
    )
    if any(info.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError(f'{kind}: native reload was not strict: {info}')
    restored = reloaded.state_dict()
    if kind == 't5' and reloaded.shared.weight.data_ptr() != reloaded.encoder.embed_tokens.weight.data_ptr():
        raise ValueError('T5 shared embedding alias was lost during reload.')
    if set(restored) != set(mapped) or any(
        restored[key].dtype != value.dtype or not torch.equal(restored[key], value)
        for key, value in mapped.items()
    ):
        raise ValueError(f'{kind}: serialized/reloaded tensor values or dtypes differ.')
    result = {'native_keys': len(restored), 'dtype': str(dtype), 'strict_reload': True,
              'all_tensor_values_preserved': True, 'forward_equivalence_tested': False}
    del reloaded, restored, mapped, raw
    gc.collect()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', required=True, type=Path, help='Downloaded Wan2.2 directory.')
    parser.add_argument('--output', required=True, type=Path, help='Fresh directory for vae/ and text_encoder/.')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--execute', action='store_true', help='Hash, convert, and strictly reload on CPU; default only prints a plan.')
    args = parser.parse_args()
    raw_root, output = args.raw_root.expanduser(), args.output.expanduser()
    validate_paths(raw_root, output)
    if args.threads < 1:
        parser.error('--threads must be positive')
    result = {'status': 'plan_only', 'source_repo': WAN_REPO, 'source_revision': WAN_REVISION,
              'output': str(output), 'tokenizer_direct_path': str(raw_root / 'google/umt5-xxl'),
              'cpu_only': True, 'downloads': False,
              'sources': {kind: {'file': row[0], 'bytes': row[1], 'sha256': row[2]} for kind, row in RAW_FILES.items()}}
    if not args.execute:
        print(json.dumps(result, indent=2))
        return 0
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    for name, version in (('torch', '2.11.0'), ('diffusers', '0.37.1'), ('transformers', '5.10.4'), ('accelerate', '1.13.0')):
        if metadata.version(name).split('+')[0] != version:
            raise ValueError(f'Use the fixed environment before conversion: {name}=={version}.')
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError('Asset conversion requires a CPU-only process.')
    torch.set_num_threads(args.threads)
    for filename, size, digest in RAW_FILES.values():
        path = raw_root / filename
        if path.is_symlink():
            raise ValueError('Raw weight files must be direct paths.')
        verify_file(path, size, digest)
    tokenizer_root = raw_root / 'google/umt5-xxl'
    for filename, (size, digest) in TOKENIZER_FILES.items():
        path = tokenizer_root / filename
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError('Tokenizer files must be direct paths.')
        verify_file(path, size, digest)
    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained(tokenizer_root, local_files_only=True)
    output.mkdir(parents=True, exist_ok=False)
    manifest = output / 'conversion_manifest.json'
    result['status'] = 'conversion_in_progress'
    manifest.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    for kind, subdir in (('vae', 'vae'), ('t5', 'text_encoder')):
        print(f'Converting {kind} on CPU...', flush=True)
        result[kind] = convert_component(kind, raw_root / RAW_FILES[kind][0], output / subdir)
    if torch.cuda.is_initialized():
        raise RuntimeError('Unexpected CUDA initialization.')
    result['output_files'] = {
        str(path.relative_to(output)): {'bytes': path.stat().st_size, 'sha256': sha256_file(path)}
        for path in sorted(output.rglob('*')) if path.is_file() and path != manifest
    }
    result['status'] = 'converted_and_strictly_reloaded'
    manifest.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
