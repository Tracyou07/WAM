from pathlib import Path

from torch import nn

from gradientwam.settings import load_cine_settings
from open_wam.models.visual_tower.reference_loader import resolve_pretrained_component_dir
from open_wam.models.visual_tower.runtime_backbone import initialize_runtime_backbone


def test_cine_encoder_assets_do_not_trigger_runtime_transformer_load(tmp_path, monkeypatch):
    roots = {
        "GW_CHECKPOINT": tmp_path / "openwam" / "model_state.pt",
        "GW_FRONTEND_ROOT": tmp_path / "encoders",
        "GW_TOKENIZER_ROOT": tmp_path / "tokenizer",
        "GW_CINE_TRAIN_ROOT": tmp_path / "train_0.6k",
        "GW_CINE_VAL_ROOT": tmp_path / "validation_200",
        "GW_CINE_LATENT_ROOT": tmp_path / "cache",
        "GW_OUTPUT_ROOT": tmp_path / "run",
    }
    for name, path in roots.items():
        monkeypatch.setenv(name, str(path))
    monkeypatch.setenv("GW_CHECKPOINT_SHA256", "a" * 64)
    frontend = roots["GW_FRONTEND_ROOT"]
    (frontend / "vae").mkdir(parents=True)
    (frontend / "text_encoder").mkdir()
    roots["GW_TOKENIZER_ROOT"].mkdir()
    config_path = Path(__file__).parents[1] / "configs/cine_v3/vrfm_cagrad.yaml"
    settings = load_cine_settings(config_path)
    backbone = settings.native_config().backbone

    # Exercise the real initialization guard without allocating the full WAM.
    assert initialize_runtime_backbone(
        current_report=None, core=nn.Identity(), config=backbone, action_dim=7
    ) is None
    assert settings.checkpoint == roots["GW_CHECKPOINT"]
    assert settings.frontend_root == frontend
    for subdir, expected in (
        (backbone.vae_subdir, frontend / "vae"),
        (backbone.text_encoder_subdir, frontend / "text_encoder"),
        (backbone.tokenizer_subdir, roots["GW_TOKENIZER_ROOT"]),
    ):
        assert resolve_pretrained_component_dir(
            backbone.pretrained_model_name_or_path, subdir
        ) == expected
    assert not (frontend / "transformer").exists()
