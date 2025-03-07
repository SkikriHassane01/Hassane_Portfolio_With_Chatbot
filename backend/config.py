import os 
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Base Configuration class with all common settings."""
    
    # Application settings
    DEBUG = False
    TESTING = False
    SECRET_KEY = os.environ.get('SECRET_KEY')
    BASE_DIR = Path(__file__).resolve().parent
    
    # Azure OpenAI settings
    AZURE_OPENAI_BASE_URL = os.environ.get('AZURE_OPENAI_BASE_URL', 'https://models.inference.ai.azure.com')
    AZURE_OPENAI_API_KEY = os.environ.get('AZURE_OPENAI_API_KEY')
    AZURE_OPENAI_MODEL = os.environ.get('AZURE_OPENAI_MODEL', 'gpt-4o-mini')
    
    # Model and data settings
    MODEL_PATH = os.environ.get('MODEL_PATH', 'models/chatbot_model.h5')
    INTENTS_PATH = os.environ.get('INTENTS_PATH', 'data/intents.json')
    MODEL_INFO = os.environ.get('MODEL_INFO_PATH','models/model_info.pkl')
    CLASSED_PATH = os.environ.get('CLASSES_PATH','models/classes.pkl')
    CONFIDENCE_THRESHOLD = float(os.environ.get('CONFIDENCE_THRESHOLD', '0.9'))

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    
    # Development-specific settings
    DEVELOPMENT_MODE = True
    
class TestingConfig(Config):
    """Testing configuration."""
    DEBUG = True
    TESTING = True
    
    # Test-specific settings
    WTF_CSRF_ENABLED = False

class ProductionConfig(Config):
    """Production configuration."""
    
    # Production-specific settings
    DEBUG = False
    TESTING = False
    DEVELOPMENT_MODE = False
    
    # Production security settings
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

def get_config():
    """
    Get the configuration based on the environment.
    
    Returns:
        Config: The configuration class based on FLASK_ENV environment variable.
        Defaults to development configuration if not specified.
    """
    env = os.environ.get('FLASK_ENV', 'default')
    return config.get(env, config['default'])