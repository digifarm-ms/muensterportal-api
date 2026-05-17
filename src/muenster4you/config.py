from pathlib import Path

from pydantic_settings import BaseSettings


class AppConfig(BaseSettings):
    lancedb_fp: Path
