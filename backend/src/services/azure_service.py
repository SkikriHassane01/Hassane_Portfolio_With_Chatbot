import os
import sys
from openai import OpenAI

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import get_config
from src.utils.logger import setup_logger

# Get configuration and set up logger
config = get_config()
logger = setup_logger("azure_service")

class AzureOpenAIService:
    """
    Service for handling conversations with Azure OpenAI when intent classifier 
    confidence is below threshold.
    """
    
    def __init__(self, base_url=None, api_key=None, model=None):
        """
        Initialize Azure OpenAI client with configuration settings.
        
        Args:
            base_url: Azure OpenAI endpoint URL (defaults to config)
            api_key: Azure OpenAI API key (defaults to config)
            model: Model name to use (defaults to config)
        """
        # Get settings from config if not provided
        self.base_url = base_url or config.AZURE_OPENAI_BASE_URL
        self.api_key = api_key or config.AZURE_OPENAI_API_KEY
        self.model = model or config.AZURE_OPENAI_MODEL
        
        logger.info(f"Initializing Azure OpenAI service with model: {self.model}")
        
        try:
            # Initialize the OpenAI client with Azure configuration
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,  # Set base_url during initialization
                default_headers={
                    "api-key": self.api_key  # Azure OpenAI requires api-key in headers
                }
            )
            logger.info("Azure OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI client: {str(e)}")
            self.client = None
    
    def create_portfolio_instruction(self):
        """
        Create the instruction prompt that explains Hassane's portfolio to the AI.
        
        Returns:
            String containing portfolio information and instructions
        """
        return """You are responding as an assistant for Hassane Skikri, a Computer Science Student and Data Scientist studying at École Nationale des Sciences Appliquées de Fès (2021-2026).
                Your answers should help visitors learn about Hassane's:
                - Professional skills (Data Science, Machine Learning, Computer Vision, Python, Deep Learning)
                - Projects and their technical details
                - Educational background and certifications
                - Contact information and availability
                Keep responses friendly, professional, and technically accurate. Be concise while highlighting Hassane's expertise.
                
                Now, please respond to the user's question."""
    
    def generate_response(self, message, conversation_history=None):
        """
        Generate a response to user message using Azure OpenAI.
        
        Args:
            message: User's message
            conversation_history: Optional list of previous messages
            
        Returns:
            AI-generated response text or error message
        """
        if not self.client:
            logger.error("Cannot generate response: Azure OpenAI client not initialized")
            return "I'm sorry, I'm having trouble accessing my knowledge base right now."
        
        try:
            # Step 1: Prepare messages for the conversation
            messages = self._prepare_messages(message, conversation_history)
            
            # Step 2: Call Azure OpenAI API
            logger.info(f"Sending request to Azure OpenAI for: '{message[:50]}...'")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=500  # Using max_tokens which is the standard parameter
            )
            
            # Step 3: Extract and return the response text
            response_text = response.choices[0].message.content
            logger.info(f"Generated response ({len(response_text)} chars)")
            
            return response_text
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"I'm sorry, I couldn't generate a response at the moment. Error: {str(e)}"
    
    def _prepare_messages(self, message, conversation_history=None):
        """
        Prepare the messages array for the API request.
        
        Args:
            message: Current user message
            conversation_history: Optional previous messages
            
        Returns:
            List of message objects for the API
        """
        messages = []
        
        # If this is a new conversation, add the portfolio instruction
        if not conversation_history:
            # Add initial instruction message
            messages.append({
                "role": "user",
                "content": self.create_portfolio_instruction()
            })
            
            # Add assistant acknowledgment
            messages.append({
                "role": "assistant",
                "content": "I understand and will provide helpful information about Hassane Skikri's portfolio, skills, and background."
            })
        # Otherwise, include the conversation history
        elif conversation_history:
            # Add valid messages from history
            messages.extend([
                msg for msg in conversation_history
                if isinstance(msg, dict) and "role" in msg and "content" in msg
            ])
        
        # Add the current user message
        messages.append({
            "role": "user",
            "content": message
        })
        
        return messages


if __name__ == "__main__":
    # Create Azure OpenAI service
    azure_service = AzureOpenAIService()
    
    # Test with some example questions
    test_questions = [
        "What are Hassane's main skills?",
        "what is Hassane's educational background?",
        "what is the king of Morocco?",
    ]
    
    # Test each question
    for question in test_questions:
        print(f"\nQ: {question}")
        response = azure_service.generate_response(question)
        print(f"A: {response[:200]}...")
        