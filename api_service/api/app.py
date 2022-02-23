from fastapi import FastAPI
from api_service.api.config import get_api_settings
from api_service.api.routes import  api, inference

def get_app() -> FastAPI:
    settings = get_api_settings()
    app = FastAPI(title='Topic Modelling API')

    # Register the routes on the main app
    app.include_router(api.router)
    app.include_router(inference.router)

    return app
