from __future__ import annotations

import pytest

from gradientwam.settings import (
    LEGACY_ARMS,
    GradientWAMMethod,
    parse_method_config,
)


@pytest.mark.parametrize("method", [method.value for method in GradientWAMMethod])
def test_new_method_schema_uses_four_explicit_arms(method: str) -> None:
    config = parse_method_config({"gradientwam": {"method": method}})

    assert config.method.value == method
    assert config.legacy_v02 is False
    assert config.uses_vrfm is (method in {"vrfm", "vrfm_cagrad"})
    assert config.uses_cagrad is (method in {"cagrad", "vrfm_cagrad"})


def test_method_defaults_and_all_hyperparameters_are_identity() -> None:
    default = parse_method_config({})
    changed = parse_method_config(
        {
            "gradientwam": {
                "method": "vrfm_cagrad",
                "latent_dim": 64,
                "kl_weight": 0.002,
                "cagrad_c": 0.7,
            }
        }
    )

    assert default.method is GradientWAMMethod.BASELINE
    assert (default.latent_dim, default.kl_weight, default.cagrad_c) == (32, 0.001, 0.4)
    assert default.identity() != changed.identity()
    assert changed.identity() == {
        "method": "vrfm_cagrad",
        "latent_dim": 64,
        "kl_weight": 0.002,
        "cagrad_c": 0.7,
        "legacy_v02": False,
    }


@pytest.mark.parametrize(
    "raw",
    [
        {"gradientwam": {"method": "variational_sharing"}},
        {"arm": "variational_sharing"},
        {"legacy_v02": False, "arm": "native_joint"},
        {"gradientwam": {"cagrad_c": 1.0}},
        {"gradientwam": {"cagrad_c": float("nan")}},
        {"gradientwam": {"latent_dim": True}},
        {"gradientwam": {"kl_weight": -0.1}},
    ],
)
def test_invalid_or_implicit_legacy_method_is_rejected(raw: dict) -> None:
    with pytest.raises((TypeError, ValueError)):
        parse_method_config(raw)


@pytest.mark.parametrize("arm", LEGACY_ARMS)
def test_legacy_private_route_requires_explicit_marker(arm: str) -> None:
    config = parse_method_config({"legacy_v02": True, "arm": arm})

    assert config.legacy_v02 is True
    assert config.legacy_arm == arm
    assert config.identity()["method"] == "legacy_v02"
    assert config.identity()["arm"] == arm


def test_legacy_marker_cannot_be_combined_with_new_method() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        parse_method_config(
            {
                "legacy_v02": True,
                "arm": "native_joint",
                "gradientwam": {"method": "baseline"},
            }
        )
