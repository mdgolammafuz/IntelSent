import logging
import os
import sys
import structlog

def configure_logging(service: str = "intelsent-api") -> structlog.stdlib.BoundLogger:
    """
    Minimal, structured JSON logging to stdout.
    LOG_LEVEL env controls verbosity (default INFO).
    """
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        stream=sys.stdout,
        format="%(message)s",
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="ISO", utc=True),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    log = structlog.get_logger().bind(service=service)
    return log

# shared module-level logger
logger = configure_logging()
