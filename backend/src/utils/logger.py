import os 
import logging
from datetime import datetime

def setup_logger(name='chatbot'):
    """
    Set up a logger that writes to both console and file
    """
    
    os.makedirs('logs', exist_ok=True)
    
    # create the logger 
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        # create a file handler 
        log_file = f"logs/{name}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # create console handler 
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # create the formatter 
        formatter = logging.Formatter(
            '%(asctime)4s | %(name)4s | %(levelname)4s | %(message)s',
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        #add handlers to the logger 
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    return logger