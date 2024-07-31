
import logging
from pathlib import Path

# ------------------------------------------------------------------------------
# Logging function & formats  --------------------------------------------------
# ------------------------------------------------------------------------------

def configure_logger(out_path = ".", log_level: str = "info"):

    # Set logging level
    if log_level == "notset":
        level = logging.NOTSET
    elif log_level == "debug":
        level = logging.DEBUG
    elif log_level == "info":
        level = logging.INFO
    elif log_level == "warn":
        level = logging.WARN
    elif log_level == "error":
        level = logging.ERROR
    elif log_level == "critical":
        level = logging.CRITICAL
    else:
        import sys
        logger = configure_logger(out_path, log_level = "error")(__name__)
        logger.error(f"Unknown log_level: {log_level}")
        logger.error( "EXIT")
        sys.exit()

    # Ensure the output directory exists
    Path(out_path).mkdir(parents=True, exist_ok=True)
    
    # Create formatters
    file_formatter = logging.Formatter('%(asctime)s|%(name)s|%(levelname)s|%(message)s')
    console_formatter = logging.Formatter('%(levelname)s|%(message)s')
    
    # Create handlers
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(f'{out_path}/multimer_mapper.log')
    
    # Set formatters for handlers
    console_handler.setFormatter(console_formatter)
    file_handler.setFormatter(file_formatter)
    
    # Configure logging
    logging.basicConfig(
        level=level,
        handlers=[
            console_handler,
            file_handler
        ]
    )
    
    # logger = logging.getLogger(__name__)
    
    # Set matplotlib logger to warning level to reduce verbosity
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
    return logging.getLogger


