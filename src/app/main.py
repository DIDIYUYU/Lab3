import logging

from fastapi import FastAPI

from app.routes import root_router
from app.text_generator import TextGenerator, TransformersTextGenerator


def setup_dependencies(app: FastAPI) -> None:
    summarizer = TransformersTextGenerator.from_model_name('gpt2')
    app.dependency_overrides[TextGenerator] = lambda: summarizer


def create_app() -> FastAPI:
    logging.basicConfig(level=logging.INFO)
    app = FastAPI()
    app.include_router(root_router)
    setup_dependencies(app)
    return app
