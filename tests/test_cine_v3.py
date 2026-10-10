"""Real v3 sharded parquet/video fixtures, no external data or GPU."""
from dataclasses import replace
import importlib
import importlib.util
import json
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from open_wam.configs import (GenericDataConfig, ActionSchemaConfig, ActionTargetConfig, ViewLayoutConfig)


def extension():
    assert importlib.util.find_spec('open_wam.data.cine_v3') is not None, 'Cine v3 reader is missing'
    return importlib.import_module('open_wam.data.cine_v3')


def raw_repo(root, action_offset=0.):
    (root/'meta/episodes/chunk-000').mkdir(parents=True)
    (root/'data/chunk-000').mkdir(parents=True)
    video_dir = root/'videos/observation.images.color/chunk-000'
    video_dir.mkdir(parents=True)
    info = {'codebase_version': 'v3.0', 'fps': 30, 'total_episodes': 1, 'total_frames': 40,
            'data_path': 'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
            'video_path': 'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4',
            'features': {'action': {'dtype': 'float32', 'shape': [7], 'names': [f'joint_{i}' for i in range(1,8)]},
                         'observation.state': {'dtype': 'float32', 'shape': [7]},
                         'observation.images.color': {'dtype': 'video', 'shape': [16,16,3], 'video_info': {'video.fps': 30}}}}
    (root/'meta/info.json').write_text(json.dumps(info), encoding='utf-8')
    # LeRobot's prompt is commonly stored as the pandas index, not a `task` column.
    pq.write_table(pa.Table.from_pylist([{'task_index': 0, '__index_level_0__': 'move camera left'}]), root/'meta/tasks.parquet')
    episode = {'episode_index': 0, 'length': 40, 'dataset_from_index': 0, 'dataset_to_index': 40,
               'data/chunk_index': 0, 'data/file_index': 0,
               'videos/observation.images.color/chunk_index': 0,
               'videos/observation.images.color/file_index': 0,
               'videos/observation.images.color/from_timestamp': .2,
               'videos/observation.images.color/to_timestamp': 46/30}
    pq.write_table(pa.Table.from_pylist([episode]), root/'meta/episodes/chunk-000/file-000.parquet')
    rows = [{'episode_index': 0, 'frame_index': i, 'index': i, 'timestamp': i/30, 'task_index': 0,
             'action': [action_offset + i + j/10 for j in range(7)],
             'observation.state': [i, 2, 3, 0, 0, 0, 1]} for i in range(40)]
    # One episode deliberately crosses two files; both are real parquet shards.
    pq.write_table(pa.Table.from_pylist(rows[:20]), root/'data/chunk-000/file-000.parquet')
    pq.write_table(pa.Table.from_pylist(rows[20:]), root/'data/chunk-000/file-001.parquet')
    with av.open(str(video_dir/'file-000.mp4'), mode='w') as container:
        stream = container.add_stream('libx264', rate=30)
        stream.width = stream.height = 16
        stream.pix_fmt = 'yuv420p'
        for i in range(46):
            frame = av.VideoFrame.from_ndarray(np.full((16,16,3), i*4, dtype=np.uint8), format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return root


def config(train, val, cache, **options):
    return GenericDataConfig(dataset_name='cine', dataset_type='cine_v3_latent',
        local_root=str(train), val_local_root=str(val), latent_root=str(cache),
        camera_names=('observation.images.color',), latent_camera_names=('observation.images.color',),
        canonical_height=224, canonical_width=448,
        view_layout=(ViewLayoutConfig('observation.images.color','color',0,0,224,448),),
        num_frames=9, frame_stride=1, train_fraction=1.,
        action_schema=ActionSchemaConfig(action_dim=7, action_horizon=36, state_dim=7),
        action_target=ActionTargetConfig(representation='raw', source_key='action', pose_source_key='observation.state'),
        adapter_options=options)


def cache_manifest(train, val, cache, action_normalization='none'):
    cache.mkdir()
    module = extension()
    a, b = module.CineV3Repository(train), module.CineV3Repository(val)
    encoding = {'vae_identity': 'synthetic-fixture-no-real-vae', 'text_encoder_identity': 'synthetic-fixture-no-real-t5'}
    samples = {}
    for split in ('train','validation'):
        (cache/split).mkdir()
        path = f'{split}/episode0-start1.pt'
        torch.save({'video_latents': torch.randn(48,9,2,4), 'condition_latents': torch.randn(48,1,2,4),
                    'text_context': torch.randn(3,16), 'negative_text_context': torch.zeros(3,16),
                    'frame_ids': list(range(1,34)), 'condition_frame_id': 0, 'episode_index': 0,
                    'source_identity': a.identity if split == 'train' else b.identity,
                    'encoding': encoding}, cache/path)
        samples[split] = [{'path': path, 'episode_index': 0, 'raw_start': 1}]
    value = {'schema_version': 'cine_v3_latent_v1', 'complete': True, 'fps': 30, 'camera': 'observation.images.color',
             'action_semantics': 'raw_joint_command', 'raw_window_frames': 33,
             'action_normalization': action_normalization,
             'latent_frames': 9, 'latent_stride_frames': 4, 'canonical_size': [224,448],
             'encoding': encoding, 'sources': {'train': a.identity, 'validation': b.identity},
             'selection': {'scope': 'subset', 'window_stride': 1,
                 'train': {'episode_indices':[0], 'raw_starts':{'0':[1]},'window_count':1,'available_episode_count':1},
                 'validation': {'episode_indices':[0], 'raw_starts':{'0':[1]},'window_count':1,'available_episode_count':1}},
             'action_statistics': module.fit_cine_action_statistics(a, [0]), 'samples': samples}
    (cache/'manifest.json').write_text(json.dumps(value), encoding='utf-8')
    return value


def test_sharded_episode_rows_tasks_and_offset_video(tmp_path):
    repo = extension().CineV3Repository(raw_repo(tmp_path/'raw'))
    assert repo.tasks == {0: 'move camera left'}
    assert len(repo.episode_records) == 1
    rows = repo.read_rows(0, start=18, stop=23)
    assert [r['frame_index'] for r in rows] == [18,19,20,21,22]
    assert rows[-1]['action'][0] == 22
    rgb = repo.read_rgb_window(0, start=1, count=3)
    assert rgb.shape == (3,3,16,16) and rgb.dtype == torch.uint8
    # Episode starts six video frames into the shared file; row 1 is video frame 7.
    torch.testing.assert_close(rgb.float().mean((1,2,3)), torch.tensor([28.,32.,36.]), rtol=0, atol=3)
    with pytest.raises(ValueError):
        repo.read_rows(0, start=39, stop=42)


def test_registered_dataset_preserves_raw7_and_native_wan_alignment(tmp_path):
    train, val = raw_repo(tmp_path/'train'), raw_repo(tmp_path/'val', 1000.)
    cache = tmp_path/'cache'
    manifest = cache_manifest(train, val, cache)
    from open_wam.data.latent_factory import build_train_val_latent_datasets
    a, b = build_train_val_latent_datasets(config(train,val,cache))
    x, y = a[0], b[0]
    assert x.video_latents.shape[1] == 9 and x.condition_latents.shape[1] == 1
    assert x.actions.shape == (36,7) and x.state.shape == (1,7)
    assert not x.action_mask[:4].any() and x.action_mask[4:].all()
    torch.testing.assert_close(x.actions[:4], torch.zeros(4,7))
    torch.testing.assert_close(x.actions[4:,0], torch.arange(1,33).float())
    torch.testing.assert_close(y.actions[4:,0], torch.arange(1001,1033).float())
    assert x.metadata['action_semantics'] == 'raw_joint_command'
    assert x.metadata['action_frame_indices'] == [-1]*4 + list(range(1,33))
    assert x.metadata['observed_frame_indices'] == [1,5,9,13,17,21,25,29,33]
    assert x.metadata['source_root'] != y.metadata['source_root']
    assert a.action_statistics == b.action_statistics == manifest['action_statistics']
    assert x.task_text == y.task_text == 'move camera left'
    torch.testing.assert_close(x.proprio_context_frames[:,0], torch.arange(1,34,4).float())
    assert x.state[0,0].item() == 0  # independent preceding observation


def test_gaussian_uses_only_train_stats_for_both_physical_roots(tmp_path):
    train, val = raw_repo(tmp_path/'train'), raw_repo(tmp_path/'val',1000.)
    cache = tmp_path/'cache'
    manifest = cache_manifest(train,val,cache,action_normalization='gaussian')
    from open_wam.data.latent_factory import build_train_val_latent_datasets
    a,b = build_train_val_latent_datasets(config(train,val,cache,action_normalization='gaussian'))
    mean = torch.tensor(manifest['action_statistics']['mean'])
    std = torch.tensor(manifest['action_statistics']['std'])
    torch.testing.assert_close(b[0].actions[4] - a[0].actions[4], torch.full((7,),1000.)/std)
    torch.testing.assert_close(a[0].actions[4], (torch.tensor([1+j/10 for j in range(7)])-mean)/std)
    assert b.action_statistics['root'] == str(train.resolve())


@pytest.mark.parametrize('bad', ['fps','version','state8','wrist'])
def test_wrong_raw_contract_fails(tmp_path, bad):
    root = raw_repo(tmp_path/'raw')
    path=root/'meta/info.json'
    info=json.loads(path.read_text())
    if bad=='fps': info['fps']=20
    if bad=='version': info['codebase_version']='v2.1'
    if bad=='state8': info['features']['observation.state']['shape']=[8]
    if bad=='wrist': info['features'].pop('observation.images.color')
    path.write_text(json.dumps(info))
    with pytest.raises(ValueError): extension().CineV3Repository(root)


@pytest.mark.parametrize('bad', ['video_fps','missing_video_fps','shape4'])
def test_real_info_alias_has_strict_video_fps_and_rgb_shape(tmp_path,bad):
    root=raw_repo(tmp_path/'raw')
    path=root/'meta/info.json'
    info=json.loads(path.read_text())
    camera=info['features']['observation.images.color']
    camera['info']=camera.pop('video_info')
    if bad=='video_fps': camera['info']['video.fps']=20
    if bad=='missing_video_fps': camera['info'].pop('video.fps')
    if bad=='shape4': camera['shape']=[16,16,4]
    path.write_text(json.dumps(info))
    with pytest.raises(ValueError): extension().CineV3Repository(root)


@pytest.mark.parametrize('bad',['latent_frames','frame_ids','prefix','identity','val_statistics','same_root','path_escape'])
def test_stale_misaligned_or_leaking_cache_rejected(tmp_path,bad):
    train,val=raw_repo(tmp_path/'train'),raw_repo(tmp_path/'val',100.)
    cache=tmp_path/'cache'
    value=cache_manifest(train,val,cache)
    cfg=config(train,val,cache)
    if bad in ('latent_frames','frame_ids','prefix','identity'):
        path=cache/'train/episode0-start1.pt'
        payload=torch.load(path,weights_only=True)
        if bad=='latent_frames': payload['video_latents']=payload['video_latents'][:,:8]
        if bad=='frame_ids': payload['frame_ids'][2]=100
        if bad=='prefix': payload['condition_frame_id']=1
        if bad=='identity': payload['encoding']['vae_identity']='other'
        torch.save(payload,path)
    if bad=='val_statistics': value['action_statistics']['root']=str(val.resolve())
    if bad=='same_root': cfg=replace(cfg,val_local_root=str(train))
    if bad=='path_escape': value['samples']['train'][0]['path']='../outside.pt'
    (cache/'manifest.json').write_text(json.dumps(value))
    from open_wam.data.latent_factory import build_train_val_latent_datasets
    with pytest.raises((ValueError,FileNotFoundError)):
        a,b=build_train_val_latent_datasets(cfg)
        a[0]


@pytest.mark.parametrize('chunk_size',[2,4])
def test_actual_adapter_native_prefix_past_states_and_cagrad_update(tmp_path, monkeypatch, chunk_size):
    from tests.test_variational_native_pipeline import fixture
    from open_wam.pipelines import build_variant_pipeline_from_config
    from open_wam.models.policy_variants.contracts import PolicyTrainBatch
    from open_wam.models.policy_variants.dual_expert.vrfm import configure_vrfm
    from open_wam.models.policy_variants.dual_expert import packed_training
    from open_wam.data.latent_factory import build_train_val_latent_datasets
    from gradientwam.cagrad_training import (configure_native_trainability,
        CAGradGradientAccumulator,cagrad_candidate_parameters)
    torch.set_num_threads(1)
    train,val=raw_repo(tmp_path/'train'),raw_repo(tmp_path/'val',1000.)
    cache=tmp_path/'cache'
    cache_manifest(train,val,cache)
    data=config(train,val,cache)
    data=replace(data,sample_construction=replace(data.sample_construction,chunk_size=chunk_size,window_size=8))
    sample=build_train_val_latent_datasets(data)[0][0]
    *_,native=fixture(return_config=True)
    native=replace(native,data=data,action_decoder=replace(native.action_decoder,action_dim=7,action_horizon=36),
        training=replace(native.training,chunk_size=chunk_size),inference=replace(native.inference,frame_chunk_size=9))
    pipeline=build_variant_pipeline_from_config(native)
    configure_native_trainability(pipeline,expected_layers=2)
    configure_vrfm(pipeline,latent_dim=4)
    batch=PolicyTrainBatch(actions=sample.actions[None],action_mask=sample.action_mask[None],state=sample.state,
        extra={'condition_latents':sample.condition_latents[None],
            'proprio_context_frames':sample.proprio_context_frames[None],
            'proprio_context_frames_mask':sample.proprio_context_frames_mask[None],
            'metadata':(sample.metadata,)})
    projections=[]
    original=packed_training.project_hidden_proprio_context_to_frames
    def capture(*args,**kwargs):
        value=original(*args,**kwargs)
        projections.append(value.detach().clone())
        return value
    monkeypatch.setattr(packed_training,'project_hidden_proprio_context_to_frames',capture)
    def forward():
        torch.manual_seed(727)
        return pipeline.forward_train_from_latents(sample.video_latents[None],batch,text_context=sample.text_context[None])
    output=forward()
    expected=[0.,0.,1.,1.,9.,9.,17.,17.,25.,25.] if chunk_size==2 else [0.,0.,1.,1.,1.,1.,17.,17.,17.,17.]
    torch.testing.assert_close(projections[0][0,:,0],torch.tensor(expected),rtol=0,atol=0)
    artifacts=output.policy_output.decoder_artifacts.payload
    assert artifacts.video.target_latents.shape[2]==10
    assert not artifacts.video.future_loss_mask[:,:,:1].any() and artifacts.video.future_loss_mask[:,:,1:].all()
    assert not artifacts.action.action_mask[:,:4].any() and artifacts.action.action_mask[:,4:].all()
    accumulator=CAGradGradientAccumulator(cagrad_candidate_parameters(pipeline),c=.4)
    accumulator.accumulate(output.decoder_output.aux['task_losses'],{'video':True,'action':True},scale=1.)
    output.decoder_output.loss.backward()
    assert accumulator.finalize(pipeline.parameters())['applied']
    parameters=[p for p in pipeline.parameters() if p.requires_grad]
    assert all(torch.isfinite(p.grad).all() for p in parameters if p.grad is not None)
    torch.nn.utils.clip_grad_norm_(parameters,2.)
    torch.optim.AdamW(parameters,lr=1e-4).step()
    # Last target state is a future endpoint, never a preceding chunk boundary.
    pipeline.eval()
    first=forward().policy_output.decoder_artifacts.payload.action.flow_pred
    batch.extra['proprio_context_frames']=batch.extra['proprio_context_frames'].clone()
    batch.extra['proprio_context_frames'][:,-1]+=10000
    second=forward().policy_output.decoder_artifacts.payload.action.flow_pred
    torch.testing.assert_close(first,second,rtol=0,atol=0)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('bad',[None,'wrong_positive','wrong_negative','both','task_mismatch','escape'])
def test_shared_prompt_paths_bind_actual_task_and_empty_negative(tmp_path,bad):
    import hashlib
    train,val=raw_repo(tmp_path/'train'),raw_repo(tmp_path/'val',1000.)
    cache=tmp_path/'cache'
    cache_manifest(train,val,cache)
    path=cache/'train/episode0-start1.pt'
    payload=torch.load(path,weights_only=True)
    (cache/'prompt_cache/embeddings').mkdir(parents=True)
    for key,prompt in (('text_context','move camera left'),('negative_text_context','')):
        relative=f'prompt_cache/embeddings/{hashlib.sha256(prompt.encode()).hexdigest()}.pt'
        torch.save(payload.pop(key),cache/relative)
        payload[key+'_path']=relative
    payload['task_text']='move camera left'
    if bad=='wrong_positive': payload['text_context_path']=payload['negative_text_context_path']
    if bad=='wrong_negative': payload['negative_text_context_path']=payload['text_context_path']
    if bad=='both': payload['text_context']=torch.zeros(3,16)
    if bad=='task_mismatch': payload['task_text']='other task'
    if bad=='escape': payload['text_context_path']='../outside.pt'
    torch.save(payload,path)
    from open_wam.data.latent_factory import build_train_val_latent_datasets
    if bad is None:
        sample=build_train_val_latent_datasets(config(train,val,cache))[0][0]
        assert sample.text_context.shape==(3,16) and sample.negative_text_context.shape==(3,16)
    else:
        with pytest.raises(ValueError):
            build_train_val_latent_datasets(config(train,val,cache))[0][0]


@pytest.mark.parametrize('bad',['incomplete','missing_second'])
def test_partial_manifest_cannot_be_ready_under_first_sample_only_check(tmp_path,bad):
    train,val=raw_repo(tmp_path/'train'),raw_repo(tmp_path/'val',100.)
    cache=tmp_path/'cache'
    manifest=cache_manifest(train,val,cache)
    if bad=='incomplete': manifest['complete']=False
    else:
        manifest['samples']['train'].append({'episode_index':0,'raw_start':2,'path':'train/missing-second.pt'})
        manifest['selection']['train']['raw_starts']['0']=[1,2]
        manifest['selection']['train']['window_count']=2
    (cache/'manifest.json').write_text(json.dumps(manifest))
    from open_wam.data.latent_factory import build_train_val_latent_datasets
    with pytest.raises(ValueError):
        build_train_val_latent_datasets(config(train,val,cache))[0][0]


@pytest.mark.parametrize('bad',['mode_mismatch','config_stride','off_grid_subset'])
def test_mode_and_stride_must_match_the_declared_cache_recipe(tmp_path,bad):
    train,val=raw_repo(tmp_path/'train'),raw_repo(tmp_path/'val',100.)
    cache=tmp_path/'cache'
    manifest=cache_manifest(train,val,cache)
    data=config(train,val,cache)
    if bad=='mode_mismatch': data=replace(data,adapter_options={'action_normalization':'gaussian'})
    else:
        manifest['selection']['window_stride']=2
        if bad=='off_grid_subset':
            data=replace(data,sample_stride=2)
            manifest['samples']['train'][0]['raw_start']=2
            manifest['selection']['train']['raw_starts']['0']=[2]
    (cache/'manifest.json').write_text(json.dumps(manifest))
    from open_wam.data.latent_factory import build_train_val_latent_datasets
    with pytest.raises(ValueError): build_train_val_latent_datasets(data)
