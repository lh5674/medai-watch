from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.container import build_container
from services.demo_cases import get_demo_cases


def main() -> None:
    container = build_container()
    before = container.database.get_dashboard_summary()["total_count"]
    sample = get_demo_cases()[0]
    result = container.agent.run(
        sample["image"],
        image_name=f"{sample['id']}.png",
        selected_tests=["gaussian_blur", "gaussian_noise", "brightness"],
    )

    assert result["record_id"] is not None, result.get("storage_error")
    assert len(result["perturbation_results"]) == 3
    assert 0.0 <= result["risk"]["risk_score"] <= 100.0
    assert all(
        0.0 <= item["comparison"]["gradcam_cosine_similarity"] <= 1.0
        and 0.0 <= item["comparison"]["gradcam_ssim"] <= 1.0
        for item in result["perturbation_results"]
    )

    stored = container.database.get_analysis(result["record_id"])
    assert stored is not None
    assert len(stored["perturbations"]) == 3
    assert Path(stored["original_image_path"]).is_file()
    assert Path(stored["gradcam_path"]).is_file()

    summary = container.database.get_dashboard_summary()
    assert summary["total_count"] == before + 1
    assert container.database.get_decision_distribution()
    assert container.database.get_perturbation_failure_distribution()

    print(
        json.dumps(
            {
                "record_id": result["record_id"],
                "decision": result["risk"]["decision"],
                "risk_score": result["risk"]["risk_score"],
                "perturbation_count": len(stored["perturbations"]),
                "dashboard_total": summary["total_count"],
                "database_path": str(container.database.database_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
