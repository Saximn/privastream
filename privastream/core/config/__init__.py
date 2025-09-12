"""Configuration management for Privastream."""

from .model_config import ModelConfig, WebDemoConfig, ProductionConfig, default_config, web_config, production_config

# Aliases for easier import
Config = ModelConfig
WebConfig = WebDemoConfig

__all__ = [
    "ModelConfig", "WebDemoConfig", "ProductionConfig",
    "Config", "WebConfig", 
    "default_config", "web_config", "production_config"
]