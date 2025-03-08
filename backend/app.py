"""
Main Flask application entry point for the Portfolio Chatbot API.
"""
from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import sys
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.routes.chat_routes import chat_bp
from src.utils.logger import setup_logger

from config import get_config

config = get_config()

logger = setup_logger("app")

def create_app():
    """Create and configure the Flask application"""
    # Step 1: Initialize the Flask app
    app = Flask(__name__)
    
    # Step 2: Enable CORS for API endpoints
    CORS(app, resources={r"/api/*": {
        "origins": ["http://localhost:3000"]
    }})
    
    # Step 3: Register the chat blueprint
    logger.info("Registering chat routes blueprint")
    app.register_blueprint(chat_bp)
    
    # Step 4: Define the main index route
    @app.route('/')
    def index():
        """API documentation endpoint"""
        return jsonify({
            "name": "Portfolio Chatbot API",
            "status": "online",
            "version": "1.0.0",
            "endpoints": {
                "chat": "/api/chat - Send POST requests with user messages",
                "health": "/api/chat/health - Check API health status",
                "test": "/api/chat/test - Test endpoint with static response"
            }
        })
    
    # Step 5: Configure error handlers
    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 errors"""
        logger.warning(f"404 error: {request.path}")
        return jsonify({
            "error": "The requested resource was not found",
            "status": "error"
        }), 404
    
    @app.errorhandler(405)
    def method_not_allowed(error):
        """Handle 405 errors"""
        logger.warning(f"405 error: {request.method} {request.path}")
        return jsonify({
            "error": "This method is not allowed for this endpoint",
            "status": "error"
        }), 405
    
    @app.errorhandler(500)
    def server_error(error):
        """Handle 500 errors"""
        logger.error(f"Server error: {str(error)}")
        return jsonify({
            "error": "An internal server error occurred",
            "status": "error"
        }), 500
    
    # Step 6: Add request logging
    @app.before_request
    def log_request_info():
        """Log information about incoming requests"""
        # Skip logging for health check endpoints to avoid filling logs
        if not request.path.endswith('/health'):
            logger.info(f"Request: {request.method} {request.path} from {request.remote_addr}")
    
    logger.info("Flask application configured successfully")
    return app

# Create the application instance
app = create_app()

if __name__ == '__main__':
    # Get port from environment or use default
    port = int(os.environ.get('PORT', 5000))
    debug = config.DEBUG
    
    # Log startup information
    logger.info(f"Starting Portfolio Chatbot API on port {port} (debug={debug})")
    
    # Start the Flask application
    app.run(host='0.0.0.0', port=port, debug=True)