"""Independent cache/physical-identity regression checks on small v3 fixtures."""
import copy
import json
import os
from pathlib import Path

import pytest
import torch

from tests.test_cine_v3 import cache_manifest, config, raw_repo


def prepared(tmp_path):
    train = raw_repo(tmp_path / 'train')
    validation = raw_repo(tmp_path / 'validation', action_offset=1000)
    cache = tmp_path / 'cache'
    manifest = cache_manifest(train, validation, cache)
    return train, validation, cache, manifest


def datasets(train, validation, cache):
    from open_wam.data.cine_v3_latent import build_cine_latent_train_val_datasets
    return build_cine_latent_train_val_datasets(config(train, validation, cache))


def test_cine_sample_preserves_native_state_anchor_sequence(tmp_path):
    train, validation, cache, _ = prepared(tmp_path)
    sample = datasets(train, validation, cache)[0][0]
    assert sample.state[0, 0].item() == 0  # Independent condition at start-1.
    assert sample.metadata['condition_frame_id'] == 0
    assert sample.metadata['proprio_context_frame_indices'] == [1, 5, 9, 13, 17, 21, 25, 29, 33]
    torch.testing.assert_close(sample.proprio_context_frames[:, 0],
        torch.tensor([1., 5., 9., 13., 17., 21., 25., 29., 33.]), rtol=0, atol=0)
    assert sample.metadata['frame_shift'] == 1
    assert sample.metadata['chunk_origin_frame'] == 1
    assert sample.metadata['action_loss_frame_start'] == 1
    assert sample.metadata['action_loss_frame_end'] == 9


def test_train_payload_cannot_be_loaded_as_same_id_validation_episode(tmp_path):
    train, validation, cache, manifest = prepared(tmp_path)
    source = cache / manifest['samples']['train'][0]['path']
    target = cache / manifest['samples']['validation'][0]['path']
    target.write_bytes(source.read_bytes())
    with pytest.raises(ValueError):
        _, heldout = datasets(train, validation, cache)
        heldout[0]


def test_duplicate_sample_window_is_rejected(tmp_path):
    train, validation, cache, manifest = prepared(tmp_path)
    manifest['samples']['train'].append(copy.deepcopy(manifest['samples']['train'][0]))
    (cache / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError):
        datasets(train, validation, cache)


def test_overlapping_hardlinked_video_is_rejected_across_split_roots(tmp_path):
    train = raw_repo(tmp_path / 'train')
    validation = raw_repo(tmp_path / 'validation', action_offset=1000)
    relative = Path('videos/observation.images.color/chunk-000/file-000.mp4')
    (validation / relative).unlink()
    os.link(train / relative, validation / relative)
    cache = tmp_path / 'cache'
    cache_manifest(train, validation, cache)
    with pytest.raises(ValueError):
        datasets(train, validation, cache)


def test_episode_video_mapping_change_invalidates_prepared_cache(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    train, validation, cache, _ = prepared(tmp_path)
    path = train / 'meta/episodes/chunk-000/file-000.parquet'
    records = pq.read_table(path).to_pylist()
    for key in ('from_timestamp', 'to_timestamp'):
        records[0]['videos/observation.images.color/' + key] += 1 / 30
    pq.write_table(pa.Table.from_pylist(records), path)
    with pytest.raises(ValueError):
        training, _ = datasets(train, validation, cache)
        training[0]


def test_partial_manifest_cannot_claim_all_windows(tmp_path):
    train, validation, cache, manifest = prepared(tmp_path)
    selection = {'episode_indices': [0], 'raw_starts': {'0': [1]},
                 'window_count': 1, 'available_episode_count': 1}
    manifest['selection'] = {'scope': 'all_windows', 'window_stride': 1,
                             'train': copy.deepcopy(selection),
                             'validation': copy.deepcopy(selection)}
    (cache / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    # Forty raw frames permit starts 1..7, not just the sole cached start 1.
    with pytest.raises(ValueError):
        datasets(train, validation, cache)


def test_cine_check_data_reads_actual_native_cache(tmp_path):
    from types import SimpleNamespace
    from gradientwam.cine_runner import _check_data

    train, validation, cache, _ = prepared(tmp_path)
    native = SimpleNamespace(data=config(train, validation, cache))
    settings = SimpleNamespace(native_config=lambda: native, latent_root=cache)
    report = _check_data(settings, sample_limit=1)
    assert report['status'] == 'valid'
    assert report['train_samples'] == report['validation_samples'] == 1
    assert report['selection']['scope'] == 'subset'
    assert report['checked'] == {'train': 1, 'validation': 1}
    assert not torch.cuda.is_initialized()
