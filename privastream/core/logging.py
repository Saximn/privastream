import logging
import importlib
from typing import Optional, Type, Any
from functools import wraps


class ModelError(Exception):
    """Base exception for model-related errors"""
    pass


class ModelInitializationError(ModelError):
    """Raised when model fails to initialize"""
    pass


class ModelInferenceError(ModelError):
    """Raised when model inference fails"""
    pass


class ConfigurationError(ModelError):
    """Raised when configuration is invalid"""
    pass


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Setup centralized logging for the application"""
    
    logger = logging.getLogger("privastream")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s in %(name)s: %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s in %(name)s [%(filename)s:%(lineno)d]: %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def safe_import(module_name: str, class_name: str, logger: Optional[logging.Logger] = None) -> Optional[Type]:
    """Safely import model classes with consistent error handling"""
    
    if logger is None:
        logger = logging.getLogger("privastream")
    
    try:
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        logger.info(f"Successfully imported {class_name} from {module_name}")
        return cls
    except ImportError as e:
        logger.warning(f"Failed to import module {module_name}: {e}")
        return None
    except AttributeError as e:
        logger.warning(f"Failed to find {class_name} in {module_name}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error importing {class_name} from {module_name}: {e}")
        return None


def handle_model_errors(logger: Optional[logging.Logger] = None):
    """Decorator for consistent model error handling"""
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = logging.getLogger("privastream")
            
            try:
                return func(*args, **kwargs)
            except ModelError:
                # Re-raise ModelError types as-is
                raise
            except Exception as e:
                logger.error(f"Unexpected error in {func.__name__}: {e}")
                raise ModelInferenceError(f"Failed to execute {func.__name__}: {str(e)}") from e
        
        return wrapper
    return decorator


def validate_config(config: Any, required_attrs: list, logger: Optional[logging.Logger] = None) -> bool:
    """Validate that configuration has required attributes"""
    
    if logger is None:
        logger = logging.getLogger("privastream")
    
    missing_attrs = []
    for attr in required_attrs:
        if not hasattr(config, attr):
            missing_attrs.append(attr)
    
    if missing_attrs:
        error_msg = f"Configuration missing required attributes: {missing_attrs}"
        logger.error(error_msg)
        raise ConfigurationError(error_msg)
    
    logger.info("Configuration validation passed")
    return True


def log_performance(func_name: str, duration: float, logger: Optional[logging.Logger] = None):
    """Log performance metrics"""
    
    if logger is None:
        logger = logging.getLogger("privastream")
    
    if duration > 1.0:
        logger.warning(f"{func_name} took {duration:.2f}s (slow)")
    else:
        logger.debug(f"{func_name} completed in {duration:.3f}s")


# Global logger instance
logger = setup_logging()