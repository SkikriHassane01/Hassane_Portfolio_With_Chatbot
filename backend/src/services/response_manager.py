import os
import json
import sys
from typing import Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import setup_logger
from src.services.azure_service import AzureOpenAIService

from config import get_config
from src.prediction.intent_classifier import IntentClassifier

# Get configuration and set up logger
config = get_config()
logger = setup_logger("response_manager")

class ResponseManager:
    """
    Manages chat responses by selecting between local model and Azure OpenAI 
    based on confidence level.
    """
    
    def __init__(self, confidence_threshold=None, intents_path=None, model_path=None):
        """
        Initialize response manager with models and configuration.
        
        Args:
            confidence_threshold: Minimum confidence to use local model (defaults to config)
            intents_path: Path to intents file (defaults to config)
            model_path: Path to local model (defaults to config)
        """
        # Use config values if parameters not provided
        self.confidence_threshold = confidence_threshold or config.CONFIDENCE_THRESHOLD
        self.intents_path = intents_path or config.INTENTS_PATH
        self.model_path = model_path or config.MODEL_PATH
        
        logger.info(f"Initializing ResponseManager (threshold: {self.confidence_threshold})")
        
        # Load intents data from file
        self._load_intents()
        
        # Initialize the intent classifier
        self._init_intent_classifier()
        
        # Initialize Azure OpenAI service
        self.azure_service = AzureOpenAIService()

    def _load_intents(self):
        """Load intents data from file"""
        try:
            intents_path = self._resolve_path(self.intents_path)
            logger.info(f"Loading intents from {intents_path}")
            with open(intents_path, 'r', encoding='utf-8') as file:
                self.intents_data = json.load(file)
        except Exception as e:
            logger.error(f"Error loading intents file: {str(e)}")
            self.intents_data = {"intents": []}

    def _init_intent_classifier(self):
        """Initialize the intent classifier model"""
        try:
            model_path = self._resolve_path(self.model_path)
            logger.info(f"Initializing intent classifier with model at {model_path}")
            self.intent_classifier = IntentClassifier(
                model_path=model_path, 
                threshold=self.confidence_threshold
            )
        except Exception as e:
            logger.error(f"Error initializing intent classifier: {str(e)}")
            self.intent_classifier = None
            
    def _resolve_path(self, path):
        """Resolve path against base directory if not absolute"""
        if os.path.isabs(path):
            return path
        return os.path.join(config.BASE_DIR, path)
        
    def get_response(self, message: str, conversation_history=None) -> Dict:
        """
        Process a user message and return an appropriate response.
        
        Args:
            message: User's message
            conversation_history: Previous conversation messages
            
        Returns:
            Response dictionary with text and metadata
        """
        # Handle empty messages
        if not message or not message.strip():
            return {"response": "I didn't receive a message. How can I help you?", 
                    "source": "default", "confidence": 0.0, "intent": None}
        
        logger.info(f"Processing message: '{message[:50]}...'")
        
        # Step 1: Try to classify the intent locally
        if self.intent_classifier:
            intent_result = self._get_local_intent(message)
            
            # If confidence is high enough, use local response
            if intent_result["confidence"] >= self.confidence_threshold:
                return self._prepare_local_response(intent_result)
            
            # Log the transition to Azure
            logger.info(f"Confidence {intent_result['confidence']:.4f} below threshold " 
                        f"{self.confidence_threshold}, using Azure")
        
        # Step 2: Fall back to Azure OpenAI
        return self._get_azure_response(message, conversation_history)
    
    def _get_local_intent(self, message):
        """Classify message intent using local model"""
        try:
            # Get prediction from classifier
            logger.info("Classifying with local model...")
            intent_data = self.intent_classifier.predict_intent(message)
            logger.info(f"Local classification: {intent_data['intent']} "
                        f"(confidence: {intent_data['confidence']:.4f})")
            return intent_data
        except Exception as e:
            logger.error(f"Error in local intent classification: {str(e)}")
            return {"intent": "unknown", "confidence": 0.0}
    
    def _prepare_local_response(self, intent_data):
        """Prepare response using local intents data"""
        # Find matching intent in intents file
        intent_tag = intent_data["intent"]
        response = None
        
        for intent in self.intents_data.get("intents", []):
            if intent.get("tag") == intent_tag and intent.get("responses"):
                # Pick a random response
                import random
                response = random.choice(intent["responses"])
                break
        
        # Handle case where no matching intent was found
        if not response:
            logger.warning(f"No response found for intent: {intent_tag}")
            response = "I understand your request, but I'm not sure how to respond to that specifically."
        
        return {
            "response": response,
            "source": "local",
            "confidence": intent_data["confidence"],
            "intent": intent_data["intent"]
        }
    
    def _get_azure_response(self, message, conversation_history):
        """Generate response using Azure OpenAI"""
        try:
            # Format conversation history for Azure
            formatted_history = self._format_conversation_for_azure(conversation_history)
            
            # Get Azure response
            logger.info("Generating response with Azure OpenAI...")
            response = self.azure_service.generate_response(
                message=message,
                conversation_history=formatted_history
            )
            
            return {
                "response": response,
                "source": "azure",
                "confidence": 1.0,  # Azure responses are considered highest confidence
                "intent": "azure_generated"
            }
        except Exception as e:
            logger.error(f"Error generating Azure response: {str(e)}")
            return {
                "response": "I'm sorry, I encountered an issue while processing your request. Please try again.",
                "source": "error",
                "confidence": 0.0,
                "intent": None
            }
    
    def _format_conversation_for_azure(self, conversation_history):
        """Format conversation history for Azure OpenAI"""
        if not conversation_history:
            return None
            
        formatted_history = []
        
        # Keep only the most recent 5 exchanges to avoid token limits
        max_history_length = 10
        if len(conversation_history) > max_history_length:
            conversation_history = conversation_history[-max_history_length:]
            
        # Convert to the format expected by Azure service
        for item in conversation_history:
            if isinstance(item, dict):
                if "role" in item and "content" in item:
                    # Already in correct format
                    formatted_history.append(item)
                elif "user" in item and item["user"]:
                    # Convert user/assistant format to role/content
                    formatted_history.append({"role": "user", "content": item["user"]})
                    if "assistant" in item and item["assistant"]:
                        formatted_history.append({"role": "assistant", "content": item["assistant"]})
            
        return formatted_history


# Example usage
if __name__ == "__main__":
    # Create response manager
    response_manager = ResponseManager()
    
    # Test with some example questions
    test_messages = [
        "Hi there, how are you?",
        "What skills does Hassane have?",
        "Tell me about Hassane's projects",
        "What is the king of morocco?"
    ]
    
    # Try each test message
    for message in test_messages:
        result = response_manager.get_response(message)
        
        print(f"\nQ: {message}")
        print(f"A: {result['response'][:100]}...")
        print(f"Source: {result['source']}")
        print(f"Intent: {result['intent']} (Confidence: {result.get('confidence', 0):.4f})")
