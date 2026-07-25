from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from agent.monitoring_agent import MedicalAIWatchAgent
from config.settings import AppSettings, get_settings
from services.assets import AssetStore
from services.database import SQLiteRepository
from services.gradcam_service import GradCAMService
from services.image_quality import ImageQualityService
from services.predictor import PredictorService
from services.report_generator import ReportGenerator
from services.risk_engine import RiskEngine
from services.similarity import SimilarityService


@dataclass(frozen=True)
class ApplicationContainer:
    settings: AppSettings
    predictor: PredictorService
    image_quality: ImageQualityService
    gradcam: GradCAMService
    similarity: SimilarityService
    risk_engine: RiskEngine
    report_generator: ReportGenerator
    asset_store: AssetStore
    database: SQLiteRepository
    agent: MedicalAIWatchAgent


def build_container(settings: AppSettings | None = None) -> ApplicationContainer:
    app_settings = settings or get_settings()
    predictor = PredictorService(
        model_settings=app_settings.model,
        preprocessing=app_settings.preprocessing,
    )
    image_quality = ImageQualityService(app_settings.quality)
    gradcam = GradCAMService(
        predictor,
        target_layer=app_settings.model.target_layer,
    )
    similarity = SimilarityService()
    risk_engine = RiskEngine(app_settings.risk)
    report_generator = ReportGenerator()
    asset_store = AssetStore(app_settings.storage.asset_dir)
    database = SQLiteRepository(app_settings.storage.database_path)
    agent = MedicalAIWatchAgent(
        predictor=predictor,
        image_quality=image_quality,
        gradcam=gradcam,
        similarity=similarity,
        risk_engine=risk_engine,
        report_generator=report_generator,
        asset_store=asset_store,
        database=database,
        perturbation_settings=app_settings.perturbations,
    )
    return ApplicationContainer(
        settings=app_settings,
        predictor=predictor,
        image_quality=image_quality,
        gradcam=gradcam,
        similarity=similarity,
        risk_engine=risk_engine,
        report_generator=report_generator,
        asset_store=asset_store,
        database=database,
        agent=agent,
    )


@st.cache_resource(show_spinner=False)
def get_cached_container() -> ApplicationContainer:
    """Load the configured model and shared services once per Streamlit process."""
    return build_container()
