from __future__ import annotations

import sqlite3
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pytest
from PIL import Image

from services.assets import AssetStore
from services.database import SQLiteRepository


ANALYSIS_COLUMNS = [
    "id",
    "created_at",
    "image_name",
    "predicted_class",
    "original_confidence",
    "quality_score",
    "risk_score",
    "decision",
    "model_version",
    "report",
    "original_image_path",
    "gradcam_path",
]

PERTURBATION_COLUMNS = [
    "id",
    "analysis_id",
    "perturbation_name",
    "changed_class",
    "changed_confidence",
    "confidence_change",
    "class_unchanged",
    "gradcam_cosine_similarity",
    "gradcam_ssim",
    "perturbed_image_path",
    "perturbed_gradcam_path",
]


def _analysis(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "created_at": "2026-07-24T08:00:00+08:00",
        "image_name": "case.png",
        "predicted_class": "演示类别A",
        "original_confidence": 0.8,
        "quality_score": 82.0,
        "risk_score": 20.0,
        "decision": "通过",
        "model_version": "demo-fixed-v1",
        "report": "deterministic report",
        "original_image_path": "assets/original.png",
        "gradcam_path": "assets/gradcam.png",
    }
    record.update(overrides)
    return record


def _perturbation(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "perturbation_name": "高斯模糊",
        "changed_class": "演示类别A",
        "changed_confidence": 0.76,
        "confidence_change": -0.04,
        "class_unchanged": True,
        "gradcam_cosine_similarity": 0.9,
        "gradcam_ssim": 0.85,
        "perturbed_image_path": "assets/blur.png",
        "perturbed_gradcam_path": "assets/blur_gradcam.png",
    }
    record.update(overrides)
    return record


def test_asset_store_saves_pil_rgb_and_heatmaps_as_png_in_uuid_folder(
    tmp_path: Path, rgb_image: Image.Image
) -> None:
    rgb_array = np.asarray(rgb_image, dtype=np.float32) / 255.0
    heatmap = np.linspace(0.0, 1.0, 80 * 96, dtype=np.float32).reshape(80, 96)
    heatmap[0, 0] = np.nan
    store = AssetStore(tmp_path / "assets")

    saved = store.save_case(
        original_image=rgb_image,
        original_gradcam=heatmap,
        perturbations=[
            {
                "name": "高斯模糊/../unsafe",
                "image": rgb_array,
                "gradcam": heatmap,
            }
        ],
    )

    UUID(saved["case_id"])
    case_dir = tmp_path / "assets" / saved["case_id"]
    assert case_dir.is_dir()
    paths = [
        Path(saved["original_image_path"]),
        Path(saved["gradcam_path"]),
        Path(saved["perturbations"][0]["perturbed_image_path"]),
        Path(saved["perturbations"][0]["perturbed_gradcam_path"]),
    ]
    assert all(path.parent == case_dir for path in paths)
    assert all(path.suffix == ".png" and path.is_file() for path in paths)
    for path in paths:
        with Image.open(path) as saved_image:
            assert saved_image.format == "PNG"


def test_asset_store_rejects_unsafe_names_and_unsupported_arrays(
    tmp_path: Path,
) -> None:
    store = AssetStore(tmp_path)
    case_id = store.create_case_directory().name

    with pytest.raises(ValueError, match="文件名"):
        store.save_png(np.zeros((4, 4), dtype=np.float32), case_id, "../escape.png")
    with pytest.raises(ValueError, match="形状"):
        store.save_png(np.zeros((4, 4, 2), dtype=np.uint8), case_id, "bad.png")
    assert not (tmp_path.parent / "escape.png").exists()


def test_asset_store_rejects_uuid_directory_redirected_by_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = AssetStore(tmp_path / "assets")
    target = store.create_case_directory()
    redirected_id = uuid4()
    redirected = store.root / str(redirected_id)
    try:
        redirected.symlink_to(target, target_is_directory=True)
    except OSError:
        redirected.mkdir()
        real_resolve = Path.resolve

        def simulated_resolve(
            path: Path, *args: object, **kwargs: object
        ) -> Path:
            if path == redirected:
                return target
            return real_resolve(path, *args, **kwargs)

        monkeypatch.setattr(Path, "resolve", simulated_resolve)

    with pytest.raises(ValueError, match="案例目录"):
        store.save_png(
            np.zeros((4, 4, 3), dtype=np.uint8),
            redirected_id,
            "redirected.png",
        )
    assert not (target / "redirected.png").exists()


def test_repository_creates_only_the_requested_columns_and_enables_foreign_keys(
    tmp_path: Path,
) -> None:
    repository = SQLiteRepository(tmp_path / "nested" / "watch.db")

    with repository.connection() as connection:
        tables = [
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
            )
        ]
        analysis_columns = [
            row["name"] for row in connection.execute("PRAGMA table_info(analyses)")
        ]
        perturbation_columns = [
            row["name"]
            for row in connection.execute("PRAGMA table_info(perturbation_results)")
        ]
        foreign_keys = connection.execute("PRAGMA foreign_keys").fetchone()[0]
        fk_rows = connection.execute(
            "PRAGMA foreign_key_list(perturbation_results)"
        ).fetchall()

    assert tables == ["analyses", "perturbation_results"]
    assert analysis_columns == ANALYSIS_COLUMNS
    assert perturbation_columns == PERTURBATION_COLUMNS
    assert foreign_keys == 1
    assert len(fk_rows) == 1
    assert fk_rows[0]["table"] == "analyses"
    assert fk_rows[0]["from"] == "analysis_id"


def test_atomic_insert_and_single_record_retrieval_use_parameterized_values(
    tmp_path: Path,
) -> None:
    repository = SQLiteRepository(tmp_path / "watch.db")
    quoted_value = "类别A'; DROP TABLE analyses; --"

    analysis_id = repository.save_analysis(
        _analysis(image_name="O'Brien.png", predicted_class=quoted_value),
        [_perturbation(changed_class=quoted_value)],
    )
    saved = repository.get_analysis(analysis_id)

    assert saved is not None
    assert saved["id"] == analysis_id
    assert saved["created_at"] == "2026-07-24T00:00:00Z"
    assert saved["image_name"] == "O'Brien.png"
    assert saved["predicted_class"] == quoted_value
    assert saved["perturbations"][0]["changed_class"] == quoted_value
    with repository.connection() as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM analyses").fetchone()[0] == 1
        )


def test_foreign_key_rejects_orphans_and_cascades_parent_deletion(
    tmp_path: Path,
) -> None:
    repository = SQLiteRepository(tmp_path / "watch.db")
    analysis_id = repository.save_analysis(_analysis(), [_perturbation()])

    with repository.connection() as connection:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO perturbation_results (
                    analysis_id,
                    perturbation_name,
                    changed_class,
                    changed_confidence,
                    confidence_change,
                    class_unchanged,
                    gradcam_cosine_similarity,
                    gradcam_ssim,
                    perturbed_image_path,
                    perturbed_gradcam_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    analysis_id + 1000,
                    "孤立扰动",
                    "演示类别A",
                    0.7,
                    -0.1,
                    1,
                    0.9,
                    0.8,
                    "orphan.png",
                    "orphan_gradcam.png",
                ),
            )
        connection.rollback()
        connection.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))
        connection.commit()
        remaining = connection.execute(
            "SELECT COUNT(*) FROM perturbation_results WHERE analysis_id = ?",
            (analysis_id,),
        ).fetchone()[0]

    assert remaining == 0


def test_analysis_and_perturbations_roll_back_as_one_transaction(
    tmp_path: Path,
) -> None:
    repository = SQLiteRepository(tmp_path / "watch.db")

    with pytest.raises(sqlite3.IntegrityError):
        repository.save_analysis(
            _analysis(),
            [
                _perturbation(),
                _perturbation(perturbation_name=None),
            ],
        )

    with repository.connection() as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM analyses").fetchone()[0] == 0
        )
        assert (
            connection.execute("SELECT COUNT(*) FROM perturbation_results").fetchone()[
                0
            ]
            == 0
        )


def test_history_supports_date_class_decision_and_limit_filters(
    tmp_path: Path,
) -> None:
    repository = SQLiteRepository(tmp_path / "watch.db")
    repository.save_analysis(
        _analysis(
            created_at="2026-07-23T12:00:00Z",
            image_name="old.png",
            predicted_class="演示类别A",
            decision="通过",
        ),
        [],
    )
    repository.save_analysis(
        _analysis(
            created_at="2026-07-24T09:00:00Z",
            image_name="match.png",
            predicted_class="演示类别A",
            decision="通过",
        ),
        [],
    )
    repository.save_analysis(
        _analysis(
            created_at="2026-07-24T10:00:00Z",
            image_name="other.png",
            predicted_class="演示类别B",
            decision="建议人工复核",
        ),
        [],
    )

    history = repository.get_history(
        start_date="2026-07-24",
        end_date="2026-07-24",
        predicted_class="演示类别A",
        decision="通过",
        limit=1,
    )
    injection = repository.get_history(
        predicted_class="演示类别A' OR 1=1 --",
    )

    assert [row["image_name"] for row in history] == ["match.png"]
    assert injection == []
    assert len(repository.get_history()) == 3


def test_dashboard_and_distribution_queries_return_computed_counts(
    tmp_path: Path,
) -> None:
    repository = SQLiteRepository(tmp_path / "watch.db")
    repository.save_analysis(
        _analysis(
            created_at="2026-07-24T11:00:00Z",
            risk_score=80.0,
            decision="建议人工复核",
        ),
        [_perturbation(perturbation_name="高斯模糊", class_unchanged=True)],
    )
    repository.save_analysis(
        _analysis(
            created_at="2026-07-23T10:00:00Z",
            risk_score=20.0,
            decision="通过",
        ),
        [
            _perturbation(perturbation_name="高斯模糊", class_unchanged=True),
            _perturbation(perturbation_name="高斯噪声", class_unchanged=False),
        ],
    )
    repository.save_analysis(
        _analysis(
            created_at="2026-07-24T10:00:00Z",
            risk_score=40.0,
            decision="谨慎使用",
        ),
        [_perturbation(perturbation_name="高斯模糊", class_unchanged=False)],
    )

    summary = repository.get_dashboard_summary()
    decisions = repository.get_decision_distribution()
    failures = repository.get_perturbation_failure_distribution()
    filtered_summary = repository.get_dashboard_summary(start_date="2026-07-24")

    assert summary == {
        "total_count": 3,
        "high_risk_count": 1,
        "high_risk_rate": pytest.approx(1 / 3),
        "average_risk_score": pytest.approx(140 / 3),
    }
    assert decisions == [
        {"decision": "通过", "count": 1},
        {"decision": "谨慎使用", "count": 1},
        {"decision": "建议人工复核", "count": 1},
    ]
    assert failures == [
        {
            "perturbation_name": "高斯模糊",
            "failure_count": 1,
            "total_count": 3,
            "failure_rate": pytest.approx(1 / 3),
        },
        {
            "perturbation_name": "高斯噪声",
            "failure_count": 1,
            "total_count": 1,
            "failure_rate": pytest.approx(1.0),
        },
    ]
    assert filtered_summary["total_count"] == 2
    assert filtered_summary["high_risk_count"] == 1
