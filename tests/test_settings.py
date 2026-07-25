from __future__ import annotations

import pytest

from config.settings import (
    AppSettings,
    RiskSettings,
    RISK_DISCLAIMER,
    get_settings,
)


def test_default_settings_have_valid_risk_configuration() -> None:
    settings = get_settings()

    assert sum(settings.risk.weights.values()) == pytest.approx(1.0)
    assert 0 <= settings.risk.pass_threshold < settings.risk.caution_threshold <= 100
    assert settings.risk.disclaimer == RISK_DISCLAIMER


def test_default_storage_paths_are_absolute() -> None:
    settings = AppSettings()

    assert settings.storage.database_path.is_absolute()
    assert settings.storage.asset_dir.is_absolute()


def test_invalid_risk_weights_are_rejected() -> None:
    with pytest.raises(ValueError, match="权重之和"):
        RiskSettings(confidence_weight=0.5)


def test_invalid_risk_thresholds_are_rejected() -> None:
    with pytest.raises(ValueError, match="阈值"):
        RiskSettings(pass_threshold=60, caution_threshold=30)
