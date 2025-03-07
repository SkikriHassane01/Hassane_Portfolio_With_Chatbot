import numpy as np
import pickle
import os
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
import random
import sys
import json
from sentence_transformers import SentenceTransformer
from pathlib import Path
from typing import Dict
# Add the parent directory to path to import utils
backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(backend_dir)

from src.utils.logger import setup_logger
from config import get_config

config = get_config()

# Set up logger
logger = setup_logger("intent_classifier")

class IntentClassifier:
    """
    Class for intent classification using the trained model
    """
    
    def __init__(self, model_path=None, threshold=None):
        """
        Initialize the intent classifier
        
        Args:
            model_path: Path to the trained model, defaults to config MODEL_PATH
            threshold: Confidence threshold for intent prediction, defaults to config CONFIDENCE_THRESHOLD
        """
        # Settings
        self.threshold = threshold if threshold is not None else config.CONFIDENCE_THRESHOLD
        
        # Paths
        self.base_dir = config.BASE_DIR
        self.model_path = self._get_full_path(model_path or config.MODEL_PATH)
        self.classes_path = self._get_full_path(config.CLASSES_PATH)
        self.model_info_path = self._get_full_path(config.MODEL_INFO)
        self.intents_path = self._get_full_path(config.INTENTS_PATH)
        
        # Initialize resources
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        self.lemmatizer = WordNetLemmatizer()
        
        # Load the model and supporting files
        self._load_model()
        
        logger.info(">> intent classifier initialized successfully.")
        
    def _get_full_path(self, path):
        """Convert relative path to absolute path if needed."""
        if os.path.isabs(path):
            return path
        return os.path.join(self.base_dir, path)

    def _load_model(self):
        """Load the model, classes, and model information."""
        try:
            # Load the tensorflow model
            logger.info(f"Loading model from {self.model_path}")
            self.model = tf.keras.models.load_model(self.model_path)
            
            # Load the classes (intent labels)
            logger.info(f"Loading classes from {self.classes_path}")
            with open(self.classes_path, 'rb') as f:
                self.classes = pickle.load(f)
                
            # Load model info (embedding method and dimensions)
            logger.info(f"Loading model info from {self.model_info_path}")
            if self.model_info_path.endswith('.json'):
                with open(self.model_info_path, 'r') as f:
                    self.model_info = json.load(f)
            else:
                with open(self.model_info_path, 'rb') as f:
                    self.model_info = pickle.load(f)
                
            # Initialize the sentence transformer 
            self.embedding_method = self.model_info.get('embedding_method', 'sentence_transformer')
            if self.embedding_method == 'sentence_transformer':
                model_name = self.model_info.get('model_name', 'all-MiniLM-L6-v2')
                logger.info(f"Loading Sentence Transformer: {model_name}")
                self.sentence_transformer = SentenceTransformer(model_name)
                self.embedding_dim = self.model_info.get('embedding_dim', 384)
                
            # Get expected input shape from the model
            self.input_shape = self.model.input_shape or (None, self.embedding_dim)
            logger.info(f"Model expects input shape: {self.input_shape}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise ValueError(f"Failed to load model: {str(e)}")

    def predict_intent(self, message) -> Dict:
        """
        Predict the intent of the user's message.
        
        Args:
            message: User's input message
            
        Returns:
            Dictionary with intent prediction results
        """
        try:
            # Step 1: Create embedding for the message
            embedding = self.sentence_transformer.encode([message])
            
            # Step 2: Make sure the embedding has the right shape
            if embedding.shape[1] != self.input_shape[1]:
                if embedding.shape[1] > self.input_shape[1]:
                    embedding = embedding[:, :self.input_shape[1]]  # Truncate if too long
                else:
                    padding = np.zeros((1, self.input_shape[1] - embedding.shape[1]))
                    embedding = np.hstack((embedding, padding))  # Pad if too short
            
            # Step 3: Make the prediction
            prediction = self.model.predict(embedding, verbose=0)[0] # returns probability scores for each intent
            
            # Step 4: Get the best matching intent
            max_index = np.argmax(prediction)
            intent = self.classes[max_index]
            confidence = float(prediction[max_index])
            
            # Step 5: Determine if we should use fallback
            use_fallback = confidence < self.threshold
            
            logger.info(f"Intent: {intent}, Confidence: {confidence:.4f}, Fallback: {use_fallback}")
            
            return {
                "intent": intent,
                "confidence": confidence,
                "use_azure": use_fallback,
                "requires_fallback": use_fallback
            }
        except Exception as e:
            logger.error(f"Error predicting intent: {str(e)}")
            return {"intent": "unknown", "confidence": 0.0, "use_azure": True, "requires_fallback": True}

    def get_response(self, intents_json, predicted_intent):
        """
        Get a response for the predicted intent from the intents file.
        
        Args:
            intents_json: Loaded intents data
            predicted_intent: Result from predict_intent
            
        Returns:
            A response string or None if fallback is needed
        """
        # If confidence is below threshold, use Azure API
        if predicted_intent['requires_fallback']:
            logger.info("Using Azure API fallback")
            return None
        
        # Find the intent in our intents file
        tag = predicted_intent['intent']
        for intent in intents_json['intents']:
            if intent['tag'] == tag:
                # Return a random response for this intent
                response = random.choice(intent["responses"])
                return response
                
        # No matching intent found
        logger.warning(f"No response found for intent '{tag}'")
        return None

    def process_message(self, message):
        """
        Process a user message from start to finish.
        
        Args:
            message: User's input message
            
        Returns:
            Tuple of (predicted_intent, response)
        """
        try:
            # Load intents data
            intents_file = self.intents_path
                
            with open(intents_file, 'r', encoding='utf-8') as file:
                intents_json = json.load(file)
            
            # Predict intent
            predicted_intent = self.predict_intent(message)
            
            # Get response
            response = self.get_response(intents_json, predicted_intent)
            
            return predicted_intent, response
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return {"intent": "unknown", "confidence": 0.0, "use_azure": True, "requires_fallback": True}, None

if __name__ == "__main__":
    # Create intent classifier
    classifier = IntentClassifier()
    
    # Test with some example sentences
    test_sentences = [
        "Hello there",
        "what are your skills?",
        "what is health ai project",
        "what is machine learning",
        "goodby"
    ]
    
    for sentence in test_sentences:
        predicted_intent, response = classifier.process_message(sentence)
        
        print(f"\nSentence: {sentence}")
        print(f"Intent: {predicted_intent['intent']} (Confidence: {predicted_intent['confidence']:.4f})")
        print(f"Requires Azure fallback: {predicted_intent['requires_fallback']}")
        print(f"Response: {response}")