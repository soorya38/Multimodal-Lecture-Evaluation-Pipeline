import logging
import sys

import structlog

def setup_logging(log_level: str | int = logging.INFO) -> None:
    """
    Configure structlog and standard logging for the application.
    
    This setup ensures that all logs are formatted consistently as JSON 
    for production log aggregators, and integrates with standard library logging.
    """
    # Configure the standard library logging framework
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,          # Merge context vars (e.g. request IDs)
            structlog.stdlib.add_log_level,                   # Add log level (e.g., info, error)
            structlog.stdlib.add_logger_name,                 # Add logger name
            structlog.processors.TimeStamper(fmt="iso"),      # Add ISO 8601 timestamp
            structlog.processors.format_exc_info,             # Format exception tracebacks nicely
            structlog.stdlib.PositionalArgumentsFormatter(),  # Support %-style formatting
            structlog.processors.JSONRenderer(),              # Output logs as JSON
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )