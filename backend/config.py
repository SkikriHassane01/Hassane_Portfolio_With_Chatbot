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
    
    # Training settings
    BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '16'))
    LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '0.001'))
    VALIDATION_SPLIT = float(os.environ.get('VALIDATION_SPLIT', '0.2'))
    CONFIDENCE_THRESHOLD = float(os.environ.get('CONFIDENCE_THRESHOLD', '0.9'))
    
    # Azure OpenAI settings
    AZURE_OPENAI_BASE_URL = os.environ.get('AZURE_OPENAI_BASE_URL', 'https://models.inference.ai.azure.com')
    AZURE_OPENAI_API_KEY = os.environ.get('AZURE_OPENAI_API_KEY')
    AZURE_OPENAI_MODEL = os.environ.get('AZURE_OPENAI_MODEL', 'gpt-4o-mini')
    
    # Model and data paths
    MODEL_PATH = os.environ.get('MODEL_PATH', 'models/chatbot_model.h5')
    INTENTS_PATH = os.environ.get('INTENTS_PATH', 'data/intents.json')
    MODEL_INFO = os.environ.get('MODEL_INFO_PATH', 'models/model_info.pkl')
    CLASSES_PATH = os.environ.get('CLASSES_PATH', 'models/classes.pkl')
    MLRUNS_DIR = os.environ.get('MLRUNS_DIR', 'mlruns')

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    DEVELOPMENT_MODE = True
        
    BATCH_SIZE = 8
    VALIDATION_SPLIT = 0.2

class TestingConfig(Config):
    """Testing configuration."""
    DEBUG = True
    TESTING = True
    
    # Test-specific settings
    WTF_CSRF_ENABLED = False
    
    BATCH_SIZE = 4
    VALIDATION_SPLIT = 0.3

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    TESTING = False
    DEVELOPMENT_MODE = False
    
    # Production security settings
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    
    # Production-specific training settings
    BATCH_SIZE = 32
    VALIDATION_SPLIT = 0.2

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