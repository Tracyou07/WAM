"""Small provenance files only; no text encoder load or cache generation."""
import hashlib
from pathlib import Path

import pytest

from open_wam.configs import SharedVideoTransformerConfig
from open_wam.cli import encode_prompt_cache


def put(root: Path, name: str, data: bytes):
    root.mkdir(parents=True,exist_ok=True)
    (root/name).write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def test_fingerprint_resolves_absolute_component_paths_without_copy(tmp_path):
    root = tmp_path/'assets'; root.mkdir()
    encoder = tmp_path/'original_encoder'; tokenizer = tmp_path/'original_tokenizer'
    a = put(encoder,'config.json',b'encoder')
    b = put(tokenizer,'tokenizer.json',b'tokenizer')
    cfg = SharedVideoTransformerConfig(pretrained_model_name_or_path=str(root),
        text_encoder_subdir=str(encoder),tokenizer_subdir=str(tokenizer))
    hashes = encode_prompt_cache._encoder_files_sha256(cfg)
    assert hashes == {'text_encoder/config.json':a,'tokenizer/tokenizer.json':b}
    assert not (root/'text_encoder').exists() and not (root/'tokenizer').exists()


def test_fingerprint_preserves_default_relative_path_keys_and_values(tmp_path):
    a = put(tmp_path/'text_encoder','model.safetensors',b'weights')
    b = put(tmp_path/'tokenizer','tokenizer.json',b'tokens')
    cfg = SharedVideoTransformerConfig(pretrained_model_name_or_path=str(tmp_path))
    assert encode_prompt_cache._encoder_files_sha256(cfg) == {
        'text_encoder/model.safetensors':a,'tokenizer/tokenizer.json':b}


def test_fingerprint_rejects_missing_resolved_component(tmp_path):
    put(tmp_path/'text_encoder','config.json',b'config')
    cfg = SharedVideoTransformerConfig(pretrained_model_name_or_path=str(tmp_path))
    with pytest.raises(FileNotFoundError,match='tokenizer'):
        encode_prompt_cache._encoder_files_sha256(cfg)
