"""
将根 logger 同时输出到控制台与项目根目录下的 logs/ 目录。

幂等：重复调用不会叠加多个文件 handler。
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_CONFIG_ATTR = "_graphrag_workspace_logging_configured"
_CONFIGURED_LOGS_DIR_ATTR = "_graphrag_workspace_logs_dir"

# src/config/workspace_logging.py → 项目根
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOGS_DIR = _PROJECT_ROOT / "logs"
DEFAULT_LOG_FILENAME = "graphrag_web.log"


def setup_workspace_logging(
    *,
    level: int = logging.INFO,
    logs_dir: Path | None = None,
    log_filename: str = DEFAULT_LOG_FILENAME,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    console: bool = True,
    console_level: int | None = None,
) -> Path:
    """
    配置根 logger：RotatingFileHandler（logs/） + 可选 StreamHandler（stderr）。

    Returns:
        实际使用的日志目录路径（已 ensure 存在）。
    """
    root = logging.getLogger()
    if getattr(root, _CONFIG_ATTR, False):
        cached = getattr(root, _CONFIGURED_LOGS_DIR_ATTR, None)
        return Path(cached) if cached is not None else LOGS_DIR

    logs_path = Path(logs_dir) if logs_dir is not None else LOGS_DIR
    logs_path.mkdir(parents=True, exist_ok=True)

    log_path = logs_path / log_filename

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
        delay=True,
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(fmt)

    root.setLevel(level)
    root.addHandler(file_handler)

    if console:
        ch = logging.StreamHandler(sys.stderr)
        ch.setLevel(console_level if console_level is not None else level)
        ch.setFormatter(fmt)
        root.addHandler(ch)

    setattr(root, _CONFIG_ATTR, True)
    setattr(root, _CONFIGURED_LOGS_DIR_ATTR, str(logs_path.resolve()))
    logging.getLogger(__name__).info("工作区日志已启用: %s", log_path.resolve())
    return logs_path
