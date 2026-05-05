"""Central configuration — reads .env via python-dotenv."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class Settings:
    # LLM
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai").lower()
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "4096"))

    # OCR
    TESSERACT_CMD: str = os.getenv(
        "TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )

    @classmethod
    def validate(cls) -> None:
        """Raise at startup if required keys are missing."""
        if cls.LLM_PROVIDER == "openai" and not cls.OPENAI_API_KEY:
            raise EnvironmentError("OPENAI_API_KEY is not set in .env")
        if cls.LLM_PROVIDER == "gemini" and not cls.GOOGLE_API_KEY:
            raise EnvironmentError("GOOGLE_API_KEY is not set in .env")


settings = Settings()
