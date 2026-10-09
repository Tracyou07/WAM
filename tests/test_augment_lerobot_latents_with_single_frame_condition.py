from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
import numpy as np


def _load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "augment_lerobot_latents_with_single_frame_condition.py"
    spec = importlib.util.spec_from_file_location("augment_lerobot_latents_with_single_frame_condition", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_condition_source_frame_indices_follow_next_wan_source_boundaries() -> None:
    module = _load_script_module()

    indices = module._condition_source_frame_indices(
        frame_ids=list(range(10, 25)),
        latent_num_frames=4,
    )

    assert indices == [11, 15, 19, 23]


def test_condition_source_frame_indices_support_previous_frame_offset() -> None:
    module = _load_script_module()

    indices = module._condition_source_frame_indices(
        frame_ids=list(range(10, 25)),
        latent_num_frames=4,
        source_frame_offset=-1,
    )

    assert indices == [10, 14, 18, 22]


def test_condition_source_frame_indices_apply_offset_before_final_clamp() -> None:
    module = _load_script_module()

    indices = module._condition_source_frame_indices(
        frame_ids=list(range(17)),
        latent_num_frames=5,
        source_frame_offset=-1,
    )

    assert indices == [0, 4, 8, 12, 16]


def test_condition_source_frame_indices_clamp_explicit_frame_ids() -> None:
    module = _load_script_module()

    indices = module._condition_source_frame_indices(
        frame_ids=[3, 7, 11],
        latent_num_frames=5,
    )

    assert indices == [7, 11, 11, 11, 11]


def test_libero_canonical_video_from_single_view_duplicates_width_slots() -> None:
    module = _load_script_module()
    single_view = torch.randn(2, 3, 1, 128, 128)

    canonical = module._libero_canonical_video_from_single_view(single_view)

    assert canonical.shape == (2, 3, 1, 128, 256)
    torch.testing.assert_close(canonical[..., :128], single_view)
    torch.testing.assert_close(canonical[..., 128:], single_view)


def test_libero_camera_slot_maps_known_camera_names() -> None:
    module = _load_script_module()

    assert module._libero_camera_slot("observation.images.agentview_rgb") == 0
    assert module._libero_camera_slot("observation.images.eye_in_hand_rgb") == 1
    assert module._libero_camera_slot("observation.images.wrist_image") == 1
    with pytest.raises(ValueError, match="Could not map LIBERO camera name"):
        module._libero_camera_slot("observation.images.side_rgb")


def test_batch_encoding_max_diff_skips_batch_check_for_size_one() -> None:
    module = _load_script_module()

    class FakeAssets:
        def __init__(self) -> None:
            self.calls = 0

        def encode_video(self, video, *, placements=None, reset_cache=True):
            self.calls += 1
            return torch.zeros(video.shape[0], 48, 1, 8, 8)

    assets = FakeAssets()
    diff = module._batch_encoding_max_diff(
        torch.zeros(1, 3, 1, 128, 128),
        placements=None,
        assets=assets,
        batch_size=1,
    )

    assert diff == 0.0
    assert assets.calls == 0


def test_build_payload_tasks_groups_libero_camera_pair(tmp_path: Path) -> None:
    module = _load_script_module()
    root = tmp_path / "latents" / "chunk-000"
    agent = root / "observation.images.agentview_rgb" / "episode_000000_0_272.pth"
    wrist = root / "observation.images.eye_in_hand_rgb" / "episode_000000_0_272.pth"
    side = root / "observation.images.side_rgb" / "episode_000001_0_272.pth"
    for path in (agent, wrist, side):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    tasks = module._build_payload_tasks([side, wrist, agent])

    assert len(tasks) == 2
    assert set(tasks[0]) == {agent, wrist}
    assert tasks[1] == (side,)


def test_save_payload_atomic_replaces_payload_without_tmp_leftover(tmp_path: Path) -> None:
    module = _load_script_module()
    payload_path = tmp_path / "episode_000000.pth"
    torch.save({"old": torch.tensor([1])}, payload_path)

    module._save_payload_atomic({"new": torch.tensor([2])}, payload_path)

    loaded = torch.load(payload_path, map_location="cpu", weights_only=False)
    assert loaded["new"].item() == 2
    assert not payload_path.with_name(f"{payload_path.name}.tmp").exists()


def test_libero_raw_preprocessing_matches_base_encoder_letterbox():
    from open_wam.data.preparation.frames import fit_frames
    module = _load_script_module()
    raw = np.arange(173*257*3,dtype=np.uint8).reshape(173,257,3)
    actual = module._frames_to_video_tensor([raw],device=torch.device('cpu'),target_size=(128,128))
    expected = fit_frames(raw[None],128,128,'letterbox_pad').unsqueeze(2)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)


def test_libero_raw512_views_keep_camera_order_and_48_channel_layout():
    from open_wam.configs import SharedVideoTransformerConfig
    from open_wam.models.visual_tower.reference_assets import LingbotReferenceAssets
    torch.set_num_threads(1)
    module = _load_script_module()
    agent = module._frames_to_video_tensor([np.zeros((512,512,3),np.uint8)],
        device=torch.device('cpu'),target_size=(128,128))
    wrist = module._frames_to_video_tensor([np.full((512,512,3),255,np.uint8)],
        device=torch.device('cpu'),target_size=(128,128))
    canvas = torch.cat((agent,wrist),dim=-1)
    assert canvas.shape == (1,3,1,128,256)
    assets = LingbotReferenceAssets(config=SharedVideoTransformerConfig())
    placements = module._libero_placements()
    crops = [assets._crop_placed_view(canvas,placement=p,canvas_height=128,canvas_width=256)
        for p in placements]
    assert crops[0].count_nonzero()==0 and torch.all(crops[1]==1)
    # Synthetic latent markers exercise layout only; no VAE weights/encoding.
    result = assets._assemble_placed_view_latents((torch.zeros(1,48,1,8,8),
        torch.ones(1,48,1,8,8)),placements=placements,canvas_height=128,canvas_width=256)
    assert result.shape == (1,48,1,8,16)
    assert result[...,:8].count_nonzero()==0 and torch.all(result[...,8:]==1)


def test_generic_frame_conversion_retains_original_size_without_target():
    module = _load_script_module()
    actual = module._frames_to_video_tensor([np.full((256,384,3),255,np.uint8)],device=torch.device('cpu'))
    assert actual.shape == (1,3,1,256,384) and torch.all(actual==1)


def test_libero_condition_caller_preprocesses_views_and_preserves_source_frames(tmp_path,monkeypatch):
    from open_wam.configs import SharedVideoTransformerConfig
    from open_wam.models.visual_tower.reference_assets import LingbotReferenceAssets
    torch.set_num_threads(1)
    module = _load_script_module()
    cameras = ('observation.images.image','observation.images.wrist_image')
    tasks = tuple(tmp_path/'latents/chunk-000'/camera/'episode_000378_0_244.pth' for camera in cameras)
    for camera in cameras:
        video = tmp_path/'videos/chunk-000'/camera/'episode_000378.mp4'
        video.parent.mkdir(parents=True,exist_ok=True); video.touch()
    read_ids = {camera:[] for camera in cameras}
    class Reader:
        def __init__(self,camera): self.camera = camera
        def get_data(self,index):
            read_ids[self.camera].append(index)
            return np.full((512,512,3),255 if 'wrist' in self.camera else 0,np.uint8)
        def close(self): pass
    monkeypatch.setattr(module.imageio,'get_reader',lambda path:Reader(Path(path).parent.name))
    native = LingbotReferenceAssets(config=SharedVideoTransformerConfig())
    class LayoutProbe:
        calls = 0
        def encode_video(self,video,*,placements,reset_cache):
            self.calls += 1
            assert video.shape==(1,3,1,128,256) and reset_cache
            crops = [native._crop_placed_view(video,placement=p,canvas_height=128,canvas_width=256)
                for p in placements]
            assert crops[0].count_nonzero()==0 and torch.all(crops[1]==1)
            return native._assemble_placed_view_latents((torch.zeros(1,48,1,8,8),
                torch.ones(1,48,1,8,8)),placements=placements,canvas_height=128,canvas_width=256)
    # This probe supplies markers instead of VAE output; the actual caller,
    # preprocessing, source-frame selector, layout and split/flatten run unchanged.
    probe = LayoutProbe()
    payload = {'latent_num_frames':61,'latent_height':8,'latent_width':8,
        'video_num_frames':241,'frame_ids':list(range(241))}
    output = module._encode_libero_condition_latents_for_task(task=tasks,
        payloads={camera:dict(payload) for camera in cameras},video_root=tmp_path/'videos',
        chunk_size=1000,assets=probe,device=torch.device('cpu'),output_dtype_name='float16',
        batch_size=1,source_frame_offset=-1)
    assert probe.calls==61 and all(ids==list(range(0,241,4)) for ids in read_ids.values())
    assert output[cameras[0]].shape==output[cameras[1]].shape==(61*64,48)
    assert output[cameras[0]].dtype==torch.float16 and output[cameras[0]].count_nonzero()==0
    assert torch.all(output[cameras[1]]==1)
