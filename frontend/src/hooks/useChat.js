// src/hooks/useChat.js
import { useState, useCallback, useEffect, useRef } from 'react';

const API_URL = 'http://localhost:5000';

export function useChat() {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [quickReplies, setQuickReplies] = useState([]);
  const conversationIdRef = useRef(null);
  
  // Generate a random conversation ID on component mount
  useEffect(() => {
    conversationIdRef.current = `chat_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }, []);
  
  // Function to send a message to the chatbot API
  const sendMessage = useCallback(async (messageInput) => {
    // Handle both string and object message formats
    const messageText = typeof messageInput === 'object' ? messageInput.content : messageInput;
    
    if (!messageText || !messageText.trim()) return;
    
    setIsLoading(true);
    setError(null);
    
    // Add user message to messages state
    const userMessage = {
      id: Date.now().toString(),
      text: messageText,
      sender: 'user',
      timestamp: new Date()
    };
    
    setMessages(prevMessages => [...prevMessages, userMessage]);
    
    try {
      // Prepare chat history
      const history = messages.map(msg => ({
        role: msg.sender === 'user' ? 'user' : 'assistant',
        content: msg.text
      })).filter(msg => msg.content);
      
      // Send request to the API
      const response = await fetch(`${API_URL}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          message: messageText,
          conversation_id: conversationIdRef.current,
          history
        })
      });
      
      // Parse response
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.error || 'Failed to get response from the chatbot');
      }
      
      // Add bot response to messages state
      const botMessage = {
        id: (Date.now() + 1).toString(),
        text: data.response,
        sender: 'assistant',
        timestamp: new Date(),
        metadata: {
          source: data.source,
          confidence: data.confidence,
          intent: data.intent,
          processing_time: data.processing_time
        }
      };
      
      setMessages(prevMessages => [...prevMessages, botMessage]);
      
      // Update quick replies if provided by the backend
      if (data.quick_replies && Array.isArray(data.quick_replies)) {
        setQuickReplies(data.quick_replies);
      } else {
        setQuickReplies([]); // Clear quick replies if none provided
      }
      
    } catch (err) {
      console.error('Error sending message:', err);
      let errorText = 'Sorry, there was a problem connecting to the chatbot.';
      
      if (err.message === 'Failed to fetch') {
        errorText = 'Unable to connect to the chatbot server. Please make sure the backend server is running on port 5000.';
      }
      
      setError(err.message || 'Failed to send message');
      
      // Add error message
      const errorMessage = {
        id: (Date.now() + 1).toString(),
        text: errorText,
        sender: 'assistant',
        timestamp: new Date(),
        isError: true
      };
      
      setMessages(prevMessages => [...prevMessages, errorMessage]);
      setQuickReplies([]); // Clear quick replies on error
      
    } finally {
      setIsLoading(false);
    }
  }, [messages]);
  
  // Function to clear conversation
  const clearConversation = useCallback(() => {
    setMessages([]);
    setQuickReplies([]); // Clear quick replies when conversation is cleared
    // Generate a new conversation ID
    conversationIdRef.current = `chat_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }, []);
  
  return {
    messages,
    isLoading,
    error,
    sendMessage,
    clearConversation,
    quickReplies
  };
}

export default useChat;