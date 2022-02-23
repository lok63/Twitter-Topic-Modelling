from pydantic import BaseSettings
from functools import lru_cache
import json

class Settings(BaseSettings):
    """
    The values for those variables are automatically gathered from the .env file if not provided by default
    """
    # PROJECT_NAME: str
    PROJECT_ID: str
    PROJECT_ENV: str

    class Config:
        case_sensitive = True
        env_file = ".env"


class DefaultSettings(Settings):

    HOST: str = '0.0.0.0'
    PORT: int
    settings = Settings()


class DevSettings(DefaultSettings):
    # PORT: int = 5000
    pass
class ProdSettings(DefaultSettings):
    pass

class EnvironNotFound(Exception):
  pass


@lru_cache()
def get_api_settings() -> Settings:
    settings = Settings()
    if settings.PROJECT_ENV == 'development':
        settings = DevSettings()
    elif settings.PROJECT_ENV == 'production':
        settings = ProdSettings()
    else:
        raise EnvironNotFound(f"Tried accessing an environment variable with value : {settings.PROJECT_ENV} that does not exist")
    return settings
