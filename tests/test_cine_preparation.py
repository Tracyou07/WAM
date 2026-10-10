from dataclasses import replace
import importlib
import json

import pytest
import torch

from tests.test_cine_v3 import raw_repo, config


def module():
    return importlib.import_module('gradientwam.cine_preparation')


def setup_data(tmp_path):
    train = raw_repo(tmp_path / 'train')
    val = raw_repo(tmp_path / 'validation', action_offset=1000)
    return config(train, val, tmp_path / 'prepared')


def limits():
    return dict(train_episode_limit=1, train_windows_per_episode=1,
                val_episode_limit=1, val_windows_per_episode=1)


def test_plan_is_explicit_and_does_not_load_models_or_write(tmp_path, monkeypatch):
    cfg = setup_data(tmp_path)
    mod = module()
    monkeypatch.setattr(mod, '_load_frontends', lambda *a, **k: pytest.fail('model loaded in plan'))
    result = mod.prepare_cine(cfg, frontend_root=tmp_path/'assets',
                              tokenizer_root=tmp_path/'tokenizer', **limits())
    assert result['status'] == 'plan_only'
    assert result['selection']['train']['episode_indices'] == [0]
    assert result['selection']['train']['window_count'] == 1
    assert result['selection']['validation']['window_count'] == 1
    assert not (tmp_path/'prepared').exists()


@pytest.mark.parametrize('options', [{}, {'train_episode_limit': 1},
    {'all_windows': True, **limits()}, {**limits(), 'train_windows_per_episode': 0}])
def test_invalid_or_ambiguous_selection_fails_before_encoding(tmp_path, options):
    cfg = setup_data(tmp_path)
    with pytest.raises(ValueError):
        module().prepare_cine(cfg, frontend_root=tmp_path/'assets',
                               tokenizer_root=tmp_path/'tokenizer', **options)
    assert not (tmp_path/'prepared').exists()


def test_all_windows_uses_declared_stride_and_covers_both_roots(tmp_path):
    cfg = replace(setup_data(tmp_path), sample_stride=3)
    result = module().prepare_cine(cfg, frontend_root=tmp_path/'assets',
                                   tokenizer_root=tmp_path/'tokenizer', all_windows=True)
    assert result['selection']['window_stride'] == 3
    assert result['selection']['scope'] == 'all_windows'
    assert result['selection']['train']['raw_starts'] == {'0': [1, 4, 7]}


def test_existing_preparation_is_never_overwritten(tmp_path):
    cfg = setup_data(tmp_path)
    cache = tmp_path/'prepared'
    cache.mkdir()
    (cache/'sentinel').write_text('keep')
    with pytest.raises(FileExistsError):
        module().prepare_cine(cfg, frontend_root=tmp_path/'assets',
                               tokenizer_root=tmp_path/'tokenizer', execute=True, **limits())
    assert (cache/'sentinel').read_text() == 'keep'


def test_real_small_vae_preparation_roundtrips_into_native_samples(tmp_path, monkeypatch):
    from diffusers import AutoencoderKLWan
    mod = module()
    cfg = replace(setup_data(tmp_path), canonical_height=16, canonical_width=16)
    torch.manual_seed(77)
    vae = AutoencoderKLWan(base_dim=4, decoder_base_dim=4, z_dim=4,
        num_res_blocks=1, dim_mult=[1,2,4,4], temperal_downsample=[False,True,True],
        latents_mean=[0.]*4, latents_std=[1.]*4).eval().requires_grad_(False)

    class TinyText:
        def encode_prompts(self, prompts, *, device, dtype):
            return torch.stack([torch.full((3,16), 0.1 + len(p)/100,
                device=device, dtype=dtype) for p in prompts])

    enc = {'vae_identity': 'tiny-real-wan-test', 'text_encoder_identity': 'test-text-double'}
    monkeypatch.setattr(mod, '_load_frontends', lambda *a, **k: (vae, TinyText(), enc))
    result = mod.prepare_cine(cfg, frontend_root=tmp_path/'assets',
                              tokenizer_root=tmp_path/'tokenizer', execute=True, **limits())
    assert result['status'] == 'prepared'
    manifest = json.loads((tmp_path/'prepared/manifest.json').read_text())
    assert manifest['complete'] is True
    assert manifest['selection']['scope'] == 'subset'
    assert manifest['action_statistics']['root'] == str(tmp_path/'train')
    assert len(list((tmp_path/'prepared/prompt_cache/embeddings').glob('*.pt'))) == 2
    from open_wam.data.cine_v3_latent import build_cine_latent_train_val_datasets
    train, val = build_cine_latent_train_val_datasets(cfg)
    a, b = train[0], val[0]
    assert a.video_latents.shape[:2] == (4,9)
    assert a.condition_latents.shape[:2] == (4,1)
    assert a.actions.shape == (36,7)
    torch.testing.assert_close(a.actions[4:,0], torch.arange(1,33).float())
    torch.testing.assert_close(b.actions[4:,0], torch.arange(1001,1033).float())
    assert a.metadata['source_root'] != b.metadata['source_root']
    assert not torch.cuda.is_initialized()


def test_local_native_frontends_load_and_encode_without_downloads(tmp_path):
    from diffusers import AutoencoderKLWan
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import T5TokenizerFast, UMT5Config, UMT5EncoderModel
    from open_wam.models.visual_tower.vae_encoding import encode_clip

    root, tokenizer_root = tmp_path/'frontend', tmp_path/'tokenizer'
    AutoencoderKLWan(base_dim=4, decoder_base_dim=4, z_dim=4, num_res_blocks=1,
        dim_mult=[1,2,4,4], temperal_downsample=[False,True,True],
        latents_mean=[0.]*4, latents_std=[1.]*4).save_pretrained(root/'vae')
    encoder = UMT5EncoderModel(UMT5Config(vocab_size=16, d_model=16, d_ff=32,
        d_kv=8, num_layers=1, num_heads=2, dropout_rate=0.))
    encoder.save_pretrained(root/'text_encoder')
    vocab = [(token, -1.) for token in ('<pad>', '</s>', '<unk>', 'move', 'camera', 'left')]
    tokenizer = Tokenizer(models.Unigram(vocab, unk_id=2))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    T5TokenizerFast(tokenizer_object=tokenizer, extra_ids=0).save_pretrained(tokenizer_root)
    vae, assets, identity = module()._load_frontends(root, tokenizer_root, torch.device('cpu'))
    text = assets.encode_prompts(['move camera left', ''], device=torch.device('cpu'), dtype=torch.bfloat16)
    assert text.shape == (2,512,16)
    assert torch.isfinite(text).all()
    assert identity['vae_identity'] and identity['text_encoder_identity']
    assert identity['text_encoder_files_sha256']
    assert encode_clip(vae, torch.rand(33,3,16,16), normalize=True).shape[0] == 9
    assert not torch.cuda.is_initialized()
