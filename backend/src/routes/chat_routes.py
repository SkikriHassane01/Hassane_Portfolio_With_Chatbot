"""
Chat API routes for the portfolio chatbot application.
Handles user messages and returns appropriate responses.
"""
from flask import Blueprint, request, jsonify
import sys
import os
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.response_manager import ResponseManager
from utils.logger import setup_logger

# Initialize logger
logger = setup_logger("chat_routes")

# Create Blueprint
chat_bp = Blueprint('chat', __name__)

# Initialize response manager (used by all routes)
response_manager = ResponseManager()

@chat_bp.route('/api/chat', methods=['POST'])
def chat():
    """
    Main chat endpoint that processes user messages and returns responses.
    
    Request JSON format:
        {
            "message": "User's message text",
            "conversation_history": [
                {"role": "user", "content": "previous message"},
                {"role": "assistant", "content": "previous response"}
            ] 
        }
    
    Response JSON format:
        {
            "response": "Assistant's response text",
            "source": "local" | "azure" | "error",
            "confidence": 0.95,
            "intent": "detected_intent_name",
            "processing_time": 0.234,
            "status": "success" | "error"
        }
    """
    # Step 1: Start timing the request
    start_time = time.time()
    
    try:
        # Step 2: Parse and validate request data
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided", "status": "error"}), 400
        
        message = data.get('message', '').strip()
        if not message:
            return jsonify({"error": "No message provided", "status": "error"}), 400
        
        # Step 3: Extract conversation history if provided
        conversation_history = data.get('conversation_history', [])
        logger.info(f"Received message: '{message[:50]}...' with {len(conversation_history)} history items")
        
        # Step 4: Process the message through the response manager
        result = response_manager.get_response(message, conversation_history)
        
        # Step 5: Calculate processing time and prepare response
        processing_time = time.time() - start_time
        
        # Step 6: Return the formatted response
        response = {
            "response": result["response"],
            "source": result["source"],
            "confidence": result.get("confidence", 0),
            "intent": result.get("intent", "unknown"),
            "processing_time": round(processing_time, 3),
            "status": "success"
        }
        
        logger.info(f"Responded in {processing_time:.3f}s using {result['source']} "
                   f"(intent: {result.get('intent', 'unknown')})")
        
        return jsonify(response), 200
        
    except Exception as e:
        # Handle any errors that occurred during processing
        logger.error(f"Error processing chat request: {str(e)}")
        processing_time = time.time() - start_time
        
        return jsonify({
            "error": "Failed to process your request",
            "details": str(e),
            "processing_time": round(processing_time, 3),
            "status": "error"
        }), 500

@chat_bp.route('/api/chat/health', methods=['GET'])
def health_check():
    """Simple health check endpoint to verify the API is running."""
    return jsonify({
        "status": "healthy",
        "service": "Portfolio Chatbot API",
    }), 200

@chat_bp.route('/api/chat/test', methods=['GET'])
def test_chat():
    """Simple test endpoint to verify the API is accessible."""
    return jsonify({
        "response": "The Portfolio Chatbot API is running correctly. Send POST requests to /api/chat to interact.",
        "source": "test",
        "status": "success"
    }), 200