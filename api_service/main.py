from api_service.api.app import get_app
from api_service.api.config import get_api_settings
import uvicorn


settings = get_api_settings()

app = get_app()

if __name__ == '__main__':
    uvicorn.run("main:app", host='0.0.0.0', port = settings.PORT)
