from __future__ import annotations

import json
import logging
import sys
from typing import Any, Dict, Optional


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if hasattr(record, "extra") and isinstance(getattr(record, "extra"), dict):
            payload.update(getattr(record, "extra"))
        return json.dumps(payload, ensure_ascii=False)


def get_logger(name: str = "agentic_video_analysis", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonLogFormatter())
        logger.addHandler(handler)

    return logger


def log_error(logger: logging.Logger, message: str, *, error: Exception | None = None, extra: Optional[Dict[str, Any]] = None) -> None:
    try:
        if error is not None:
            logger.error(message, extra={"extra": {"error": str(error)}, "extra_data": extra})
        else:
            logger.error(message, extra={"extra": extra or {}})
    except Exception:
        # Never crash the app because of logging.
        pass