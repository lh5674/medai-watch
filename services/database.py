from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


_ANALYSIS_FIELDS = (
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
)

_PERTURBATION_FIELDS = (
    "perturbation_name",
    "changed_class",
    "changed_confidence",
    "confidence_change",
    "class_unchanged",
    "gradcam_cosine_similarity",
    "gradcam_ssim",
    "perturbed_image_path",
    "perturbed_gradcam_path",
)


def _utc_timestamp(value: str | date | datetime | None = None) -> str:
    if value is None:
        parsed = datetime.now(timezone.utc)
    elif isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime(value.year, value.month, value.day)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("created_at 不得为空。")
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("created_at 必须是有效的 ISO 8601 时间。") from exc
    else:
        raise TypeError("created_at 必须是字符串、date、datetime 或 None。")

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    parsed = parsed.astimezone(timezone.utc)
    return parsed.isoformat(timespec="seconds").replace("+00:00", "Z")


def _sqlite_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value


def _filter_value(value: str | date | datetime) -> str:
    if isinstance(value, datetime):
        return _utc_timestamp(value)
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


class SQLiteRepository:
    """SQLite persistence with atomic analysis and perturbation writes."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path).expanduser().resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(str(self.database_path))
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
        finally:
            connection.close()

    connect = connection

    def initialize(self) -> None:
        with self.connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS analyses (
                    id INTEGER PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    image_name TEXT NOT NULL,
                    predicted_class TEXT NOT NULL,
                    original_confidence REAL NOT NULL,
                    quality_score REAL NOT NULL,
                    risk_score REAL NOT NULL,
                    decision TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    report TEXT NOT NULL,
                    original_image_path TEXT NOT NULL,
                    gradcam_path TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS perturbation_results (
                    id INTEGER PRIMARY KEY,
                    analysis_id INTEGER NOT NULL,
                    perturbation_name TEXT NOT NULL,
                    changed_class TEXT NOT NULL,
                    changed_confidence REAL NOT NULL,
                    confidence_change REAL NOT NULL,
                    class_unchanged INTEGER NOT NULL CHECK (class_unchanged IN (0, 1)),
                    gradcam_cosine_similarity REAL NOT NULL,
                    gradcam_ssim REAL NOT NULL,
                    perturbed_image_path TEXT NOT NULL,
                    perturbed_gradcam_path TEXT NOT NULL,
                    FOREIGN KEY (analysis_id) REFERENCES analyses(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_analyses_created_at
                    ON analyses(created_at);
                CREATE INDEX IF NOT EXISTS idx_analyses_predicted_class
                    ON analyses(predicted_class);
                CREATE INDEX IF NOT EXISTS idx_analyses_decision
                    ON analyses(decision);
                CREATE INDEX IF NOT EXISTS idx_perturbations_analysis_id
                    ON perturbation_results(analysis_id);
                CREATE INDEX IF NOT EXISTS idx_perturbations_name
                    ON perturbation_results(perturbation_name);
                """
            )
            connection.commit()

    initialize_database = initialize

    def save_analysis(
        self,
        analysis: Mapping[str, Any],
        perturbations: Sequence[Mapping[str, Any]],
    ) -> int:
        created_at = _utc_timestamp(analysis.get("created_at"))
        analysis_values = [
            created_at,
            *(_sqlite_value(analysis[field]) for field in _ANALYSIS_FIELDS[1:]),
        ]

        with self.connection() as connection:
            try:
                connection.execute("BEGIN")
                cursor = connection.execute(
                    """
                    INSERT INTO analyses (
                        created_at,
                        image_name,
                        predicted_class,
                        original_confidence,
                        quality_score,
                        risk_score,
                        decision,
                        model_version,
                        report,
                        original_image_path,
                        gradcam_path
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    analysis_values,
                )
                analysis_id = int(cursor.lastrowid)
                for perturbation in perturbations:
                    values: list[Any] = [analysis_id]
                    for field in _PERTURBATION_FIELDS:
                        value = perturbation[field]
                        values.append(
                            int(value)
                            if field == "class_unchanged"
                            else _sqlite_value(value)
                        )
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
                        values,
                    )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return analysis_id

    insert_analysis = save_analysis

    def get_analysis(self, analysis_id: int) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM analyses WHERE id = ?",
                (int(analysis_id),),
            ).fetchone()
            if row is None:
                return None
            perturbation_rows = connection.execute(
                """
                SELECT *
                FROM perturbation_results
                WHERE analysis_id = ?
                ORDER BY id ASC
                """,
                (int(analysis_id),),
            ).fetchall()
        result = dict(row)
        result["perturbations"] = [
            self._perturbation_dict(item) for item in perturbation_rows
        ]
        return result

    get_analysis_by_id = get_analysis

    @staticmethod
    def _perturbation_dict(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["class_unchanged"] = bool(item["class_unchanged"])
        return item

    @staticmethod
    def _where(
        *,
        start_date: str | date | datetime | None = None,
        end_date: str | date | datetime | None = None,
        predicted_class: str | None = None,
        decision: str | None = None,
        alias: str = "a",
    ) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        parameters: list[Any] = []
        if start_date is not None:
            clauses.append(f"date({alias}.created_at) >= date(?)")
            parameters.append(_filter_value(start_date))
        if end_date is not None:
            clauses.append(f"date({alias}.created_at) <= date(?)")
            parameters.append(_filter_value(end_date))
        if predicted_class is not None:
            clauses.append(f"{alias}.predicted_class = ?")
            parameters.append(str(predicted_class))
        if decision is not None:
            clauses.append(f"{alias}.decision = ?")
            parameters.append(str(decision))
        return (
            f" WHERE {' AND '.join(clauses)}" if clauses else "",
            parameters,
        )

    def get_history(
        self,
        *,
        start_date: str | date | datetime | None = None,
        end_date: str | date | datetime | None = None,
        predicted_class: str | None = None,
        decision: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit 必须是正整数。")
        where, parameters = self._where(
            start_date=start_date,
            end_date=end_date,
            predicted_class=predicted_class,
            decision=decision,
        )
        parameters.append(limit)
        with self.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT a.*
                FROM analyses AS a
                {where}
                ORDER BY a.created_at DESC, a.id DESC
                LIMIT ?
                """,
                parameters,
            ).fetchall()
        return [dict(row) for row in rows]

    fetch_history = get_history
    list_analyses = get_history

    def get_dashboard_summary(
        self,
        *,
        start_date: str | date | datetime | None = None,
        end_date: str | date | datetime | None = None,
        predicted_class: str | None = None,
        decision: str | None = None,
    ) -> dict[str, int | float]:
        where, parameters = self._where(
            start_date=start_date,
            end_date=end_date,
            predicted_class=predicted_class,
            decision=decision,
        )
        with self.connection() as connection:
            row = connection.execute(
                f"""
                SELECT
                    COUNT(*) AS total_count,
                    COALESCE(
                        SUM(CASE WHEN a.decision = '建议人工复核' THEN 1 ELSE 0 END),
                        0
                    ) AS high_risk_count,
                    COALESCE(AVG(a.risk_score), 0.0) AS average_risk_score
                FROM analyses AS a
                {where}
                """,
                parameters,
            ).fetchone()
        total_count = int(row["total_count"])
        high_risk_count = int(row["high_risk_count"])
        return {
            "total_count": total_count,
            "high_risk_count": high_risk_count,
            "high_risk_rate": (
                high_risk_count / total_count if total_count else 0.0
            ),
            "average_risk_score": float(row["average_risk_score"]),
        }

    dashboard_summary = get_dashboard_summary

    def get_decision_distribution(
        self,
        *,
        start_date: str | date | datetime | None = None,
        end_date: str | date | datetime | None = None,
        predicted_class: str | None = None,
        decision: str | None = None,
    ) -> list[dict[str, Any]]:
        where, parameters = self._where(
            start_date=start_date,
            end_date=end_date,
            predicted_class=predicted_class,
            decision=decision,
        )
        parameters.extend(("通过", "谨慎使用", "建议人工复核"))
        with self.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT a.decision, COUNT(*) AS count
                FROM analyses AS a
                {where}
                GROUP BY a.decision
                ORDER BY
                    CASE a.decision
                        WHEN ? THEN 1
                        WHEN ? THEN 2
                        WHEN ? THEN 3
                        ELSE 4
                    END,
                    a.decision
                """,
                parameters,
            ).fetchall()
        return [
            {"decision": str(row["decision"]), "count": int(row["count"])}
            for row in rows
        ]

    decision_distribution = get_decision_distribution

    def get_perturbation_failure_distribution(
        self,
        *,
        start_date: str | date | datetime | None = None,
        end_date: str | date | datetime | None = None,
        predicted_class: str | None = None,
        decision: str | None = None,
    ) -> list[dict[str, Any]]:
        where, parameters = self._where(
            start_date=start_date,
            end_date=end_date,
            predicted_class=predicted_class,
            decision=decision,
        )
        with self.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    p.perturbation_name,
                    SUM(CASE WHEN p.class_unchanged = 0 THEN 1 ELSE 0 END)
                        AS failure_count,
                    COUNT(*) AS total_count
                FROM perturbation_results AS p
                JOIN analyses AS a ON a.id = p.analysis_id
                {where}
                GROUP BY p.perturbation_name
                ORDER BY MIN(p.id)
                """,
                parameters,
            ).fetchall()
        distribution: list[dict[str, Any]] = []
        for row in rows:
            failure_count = int(row["failure_count"])
            total_count = int(row["total_count"])
            distribution.append(
                {
                    "perturbation_name": str(row["perturbation_name"]),
                    "failure_count": failure_count,
                    "total_count": total_count,
                    "failure_rate": (
                        failure_count / total_count if total_count else 0.0
                    ),
                }
            )
        return distribution

    perturbation_failure_distribution = get_perturbation_failure_distribution


DatabaseRepository = SQLiteRepository
Database = SQLiteRepository
