"""
Configuration settings for the Epistemic Grounding Experiment platform.
"""

import os
from typing import Dict, List


class Config:
    """Base configuration class."""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    FLASK_ENV = os.environ.get('FLASK_ENV', 'development')
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    PORT = int(os.environ.get('FLASK_PORT', 5001))
    HOST = os.environ.get('FLASK_HOST', '0.0.0.0')
    
    # API Keys (loaded from environment or frontend)
    API_KEYS = {
        'openai': os.environ.get('OPENAI_API_KEY', ''),
        'anthropic': os.environ.get('ANTHROPIC_API_KEY', ''),
        'xai': os.environ.get('XAI_API_KEY', ''),
        'groq': os.environ.get('GROQ_API_KEY', '')
    }
    
    # LLM Provider configurations
    LLM_PROVIDERS = {
        'openai': {
            'models': [
                "o4-mini", "o3", "o3-mini", "o1", "o1-pro", 
                "gpt-4.1", "gpt-4o", "chatgpt-4o-latest", 
                "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o-mini"
            ],
            'default_model': 'gpt-4o',
            'max_tokens': 2000,
            'temperature': 0.8
        },
        'anthropic': {
            'models': [
                "claude-sonnet-4-20250514", "claude-3-7-sonnet-20250219"
            ],
            'default_model': 'claude-sonnet-4-20250514',
            'max_tokens': 2000,
            'temperature': 0.8
        },
        'xai': {
            'models': ["grok-4-0709", "grok-3", "grok-3-mini", "grok-3-fast"],
            'default_model': 'grok-4-0709',
            'max_tokens': 2000,
            'temperature': 0.8,
            'api_url': 'https://api.x.ai/v1/chat/completions'
        },
        'groq': {
            'models': [
                "llama-4.5-70b-instruct", "llama-4.1-7b-instruct",
                "mixtral-8x7b-instruct", "gemma-7b-it"
            ],
            'default_model': 'llama-4.5-70b-instruct',
            'max_tokens': 2000,
            'temperature': 0.8,
            'api_url': 'https://api.groq.com/openai/v1/chat/completions'
        }
    }
    
    # Negotiation settings
    NEGOTIATION_CONFIG = {
        'max_turns': 40,
        'turn_delay': 3,  # seconds
        'max_retries': 3,
        'retry_delay': 1  # seconds, with exponential backoff
    }
    
    # Batch execution settings
    BATCH_CONFIG = {
        'max_concurrent_runs': 5,
        'default_timeout': 300,  # seconds
        'save_results': True,
        'results_directory': 'results'
    }


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    TESTING = False


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    TESTING = False
    # Override with production-specific settings
    SECRET_KEY = os.environ.get('SECRET_KEY')
    if not SECRET_KEY:
        raise ValueError("SECRET_KEY environment variable must be set in production")


class TestingConfig(Config):
    """Testing configuration."""
    DEBUG = True
    TESTING = True
    # Use mock API keys for testing
    API_KEYS = {
        'openai': 'test-openai-key',
        'anthropic': 'test-anthropic-key', 
        'xai': 'test-xai-key',
        'groq': 'test-groq-key'
    }


# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
} 