import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { MessageSquare, X, Minimize2, ChevronDown, Maximize2 } from 'lucide-react';
import ChatMessage from './ChatMessage';
import ChatInput from './ChatInput';
import QuickReplyButtons from './QuickReplyButtons';
import { useChat } from '../../hooks/useChat';

// Initial suggestions shown when chat opens
const INITIAL_SUGGESTIONS = [
  "Tell me about Hassane's skills",
  "What projects has Hassane worked on?",
  "What is Hassane's educational background?",
  "How can I contact Hassane?",
  "What technologies does Hassane work with?"
];

// Suggestion categories for contextual follow-ups
const SUGGESTION_CATEGORIES = {
  skills: [
    "Which programming languages does Hassane know?",
    "What data science skills does Hassane have?",
    "Does Hassane have experience with machine learning?",
    "What web development skills does Hassane have?"
  ],
  projects: [
    "What are Hassane's most recent projects?",
    "Has Hassane worked on any AI projects?",
    "Tell me about Hassane's computer vision projects",
    "What machine learning projects has Hassane built?"
  ],
  education: [
    "Where did Hassane study?",
    "What degree is Hassane pursuing?",
    "What certifications does Hassane have?",
    "When will Hassane graduate?"
  ],
  contact: [
    "What's Hassane's email address?",
    "Does Hassane have a LinkedIn profile?",
    "Is Hassane available for freelance work?",
    "How can I connect with Hassane on GitHub?"
  ],
  general: [
    "What are Hassane's career goals?",
    "Is Hassane available for hire?",
    "What kind of work is Hassane looking for?",
    "Where is Hassane located?"
  ]
};

const ChatWindow = () => {
  // State for UI
  const [isOpen, setIsOpen] = useState(false);
  const [isMinimized, setIsMinimized] = useState(false);
  const [isFullScreen, setIsFullScreen] = useState(false);
  const [showScrollButton, setShowScrollButton] = useState(false);
  const [currentSuggestions, setCurrentSuggestions] = useState(INITIAL_SUGGESTIONS);
  const [isInitialMessage, setIsInitialMessage] = useState(true);

  // Get chat functionality from custom hook
  const { 
    messages, 
    isLoading, 
    sendMessage
  } = useChat();

  // Refs
  const messagesEndRef = useRef(null);
  const messagesContainerRef = useRef(null);

  // Scroll to bottom of messages when new message comes in
  useEffect(() => {
    if (isOpen && !isMinimized && messages.length > 0) {
      scrollToBottom();
    }
  }, [messages, isOpen, isMinimized]);

  // Check if should show scroll to bottom button
  useEffect(() => {
    const handleScroll = () => {
      if (!messagesContainerRef.current) return;
      
      const { scrollTop, scrollHeight, clientHeight } = messagesContainerRef.current;
      const isNearBottom = scrollHeight - scrollTop - clientHeight < 100;
      setShowScrollButton(!isNearBottom && messages.length > 3);
    };

    const container = messagesContainerRef.current;
    if (container) {
      container.addEventListener('scroll', handleScroll);
    }

    return () => {
      if (container) {
        container.removeEventListener('scroll', handleScroll);
      }
    };
  }, [messages.length]);

  // Function to generate contextual suggestions based on user message
  const generateContextualSuggestions = (userMessage) => {
    // Convert message to lowercase for easier matching
    const message = userMessage.toLowerCase();
    
    // Initialize weights for each category
    let categoryWeights = {
      skills: 0,
      projects: 0,
      education: 0,
      contact: 0,
      general: 1 // General always has a minimum weight
    };
    
    // Keywords that suggest category interest
    const keywords = {
      skills: ['skill', 'ability', 'know', 'technology', 'proficiency', 'programming', 'data', 'machine learning', 'coding'],
      projects: ['project', 'portfolio', 'work', 'create', 'develop', 'build', 'application', 'program'],
      education: ['education', 'degree', 'study', 'university', 'school', 'certificate', 'certification', 'learn'],
      contact: ['contact', 'email', 'reach', 'connect', 'linkedin', 'github', 'social', 'available']
    };
    
    // Check for keyword matches
    Object.entries(keywords).forEach(([category, words]) => {
      words.forEach(word => {
        if (message.includes(word)) {
          categoryWeights[category] += 1;
        }
      });
    });
    
    // Determine the top two categories
    const sortedCategories = Object.entries(categoryWeights)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 2)
      .map(entry => entry[0]);
    
    // Get random suggestions from top categories
    let suggestions = [];
    
    sortedCategories.forEach(category => {
      const categoryQuestions = SUGGESTION_CATEGORIES[category];
      // Get 2 random questions from each top category without duplicates
      const randomQuestions = getRandomItems(categoryQuestions, 2);
      suggestions = [...suggestions, ...randomQuestions];
    });
    
    // Ensure we have at least 3 suggestions by adding from general if needed
    while (suggestions.length < 3) {
      const generalQuestion = getRandomItems(SUGGESTION_CATEGORIES.general, 1)[0];
      if (!suggestions.includes(generalQuestion)) {
        suggestions.push(generalQuestion);
      }
    }
    
    // Limit to 4 total suggestions
    return suggestions.slice(0, 4);
  };
  
  // Helper function to get random items from an array
  const getRandomItems = (array, count) => {
    const shuffled = [...array].sort(() => 0.5 - Math.random());
    return shuffled.slice(0, count);
  };

  // Auto-greet when chat first opens
  useEffect(() => {
    // Only send welcome message when first opening chat and messages are empty
    if (isOpen && messages.length === 0) {
      // Create a welcome message with proper format
      const welcomeMessage = {
        content: "Hello! I'm Hassane's portfolio chatbot. I can tell you about his skills, projects, education, and how to contact him. What would you like to know?",
        text: "Hello! I'm Hassane's portfolio chatbot. I can tell you about his skills, projects, education, and how to contact him. What would you like to know?",
        sender: 'bot',
        type: 'bot',
        role: 'assistant',
        timestamp: new Date().toISOString(),
        initial: true
      };
      
      // Send the welcome message
      sendMessage(welcomeMessage);
      setIsInitialMessage(false);
      
      // Set initial suggestions after welcome message
      setCurrentSuggestions(INITIAL_SUGGESTIONS);
    }
  }, [isOpen, messages.length, sendMessage]);

  // Update suggestions after each message exchange
  useEffect(() => {
    if (messages.length > 0) {
      // Get the last message
      const lastMessage = messages[messages.length - 1];
      
      // If the last message is from the bot (after a user's message has been responded to)
      if ((lastMessage.sender === 'bot' || lastMessage.sender === 'assistant') && !isLoading) {
        // Generate new suggestions based on the conversation content
        const userMessages = messages.filter(msg => msg.sender === 'user');
        
        if (userMessages.length > 0) {
          // Use the most recent user message to generate context
          const latestUserMessage = userMessages[userMessages.length - 1];
          
          // Generate new suggestions
          const newSuggestions = generateContextualSuggestions(
            latestUserMessage.content || latestUserMessage.text || ""
          );
          
          console.log("Updating suggestions based on:", latestUserMessage);
          console.log("New suggestions:", newSuggestions);
          
          // Update suggestions
          setCurrentSuggestions(newSuggestions);
        }
      }
    }
  }, [messages, isLoading]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const toggleChat = () => {
    if (isMinimized) {
      setIsMinimized(false);
    } else {
      setIsOpen(!isOpen);
    }
  };

  const minimizeChat = (e) => {
    e.stopPropagation();
    setIsMinimized(true);
    setIsFullScreen(false);
  };

  const closeChat = (e) => {
    e.stopPropagation();
    setIsOpen(false);
    setIsMinimized(false);
    setIsFullScreen(false);
  };
  
  const toggleFullScreen = (e) => {
    e.stopPropagation();
    setIsFullScreen(!isFullScreen);
  };

  const handleQuickReplyClick = (reply) => {
    // Send the clicked suggestion as a user message
    sendMessage({
      content: reply,
      text: reply, // Add text property for compatibility
      sender: 'user'
    });
    
    // Force update suggestions after a short delay to ensure the messages array is updated
    setTimeout(() => {
      const newSuggestions = generateContextualSuggestions(reply);
      setCurrentSuggestions(newSuggestions);
    }, 100);
  };

  // Animation variants
  const chatButtonVariants = {
    hidden: { scale: 0, opacity: 0 },
    visible: { 
      scale: 1, 
      opacity: 1,
      transition: {
        type: 'spring',
        stiffness: 260,
        damping: 20
      }
    },
    hover: { 
      scale: 1.1,
      boxShadow: '0 5px 15px rgba(0, 0, 0, 0.1)',
      transition: {
        type: 'spring',
        stiffness: 400,
        damping: 10
      }
    },
    tap: { scale: 0.9 }
  };

  const chatWindowVariants = {
    hidden: { 
      opacity: 0,
      y: 20,
      scale: 0.9
    },
    visible: { 
      opacity: 1,
      y: 0,
      scale: 1,
      transition: {
        type: 'spring',
        stiffness: 300,
        damping: 25
      }
    },
    exit: { 
      opacity: 0,
      y: 20,
      scale: 0.9,
      transition: {
        duration: 0.2
      }
    }
  };

  const minimizedVariants = {
    hidden: { 
      opacity: 0,
      x: 20,
      width: 0 
    },
    visible: { 
      opacity: 1,
      x: 0,
      width: 'auto',
      transition: {
        type: 'spring',
        stiffness: 500,
        damping: 25
      }
    },
    exit: { 
      opacity: 0,
      x: 20,
      width: 0,
      transition: {
        duration: 0.2
      }
    }
  };

  return (
    <div className={`fixed z-50 flex flex-col items-end ${
      isFullScreen 
        ? 'inset-0 bg-black bg-opacity-50' 
        : 'bottom-5 right-5'
    }`}>
      {/* Minimized Chat Window */}
      <AnimatePresence>
        {isMinimized && (
          <motion.div
            className="flex items-center mb-4 bg-white dark:bg-gray-800 rounded-full shadow-md cursor-pointer pr-2 pl-4 py-2"
            variants={minimizedVariants}
            initial="hidden"
            animate="visible"
            exit="exit"
            onClick={() => setIsMinimized(false)}
          >
            <div className="flex items-center mr-2">
              <span className="w-2 h-2 bg-green-500 rounded-full mr-2"></span>
              <span className="font-medium text-gray-800 dark:text-white">Chat with Assistant</span>
            </div>
            <motion.button
              className="p-1 rounded-full hover:bg-gray-100 dark:hover:bg-gray-700"
              onClick={(e) => {
                e.stopPropagation();
                setIsOpen(false);
                setIsMinimized(false);
              }}
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.95 }}
            >
              <X className="h-4 w-4 text-gray-500 dark:text-gray-400" />
            </motion.button>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Chat Window */}
      <AnimatePresence>
        {isOpen && !isMinimized && (
          <motion.div
            className={`${
              isFullScreen 
                ? 'w-full h-full max-w-full rounded-none m-0' 
                : 'mb-4 w-full max-w-md rounded-xl max-h-[calc(100vh-120px)] h-550px'
            } bg-white dark:bg-gray-800 shadow-lg flex flex-col overflow-hidden border border-gray-200 dark:border-gray-700`}
            variants={chatWindowVariants}
            initial="hidden"
            animate="visible"
            exit="exit"
            style={!isFullScreen ? { height: '550px' } : {}}
          >
            {/* Chat Header */}
            <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700">
              <div className="flex items-center">
                <div className="h-8 w-8 rounded-full bg-blue-100 dark:bg-blue-900/30 flex items-center justify-center mr-3">
                  <MessageSquare className="h-4 w-4 text-blue-600 dark:text-blue-400" />
                </div>
                <div>
                  <h3 className="font-medium text-gray-800 dark:text-white">Portfolio Assistant</h3>
                  <div className="flex items-center text-sm text-green-600 dark:text-green-400">
                    <span className="w-2 h-2 bg-green-500 rounded-full mr-2"></span>
                    <span>Online</span>
                  </div>
                </div>
              </div>
              <div className="flex items-center space-x-2">
                <motion.button
                  className="p-1.5 rounded-md hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-500 dark:text-gray-400"
                  onClick={toggleFullScreen}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  aria-label={isFullScreen ? "Exit full screen" : "Full screen"}
                >
                  {isFullScreen ? (
                    <Minimize2 className="h-5 w-5" />
                  ) : (
                    <Maximize2 className="h-5 w-5" />
                  )}
                </motion.button>
                {!isFullScreen && (
                  <motion.button
                    className="p-1.5 rounded-md hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-500 dark:text-gray-400"
                    onClick={minimizeChat}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    aria-label="Minimize chat"
                  >
                    <Minimize2 className="h-5 w-5" />
                  </motion.button>
                )}
                <motion.button
                  className="p-1.5 rounded-md hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-500 dark:text-gray-400"
                  onClick={closeChat}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  aria-label="Close chat"
                >
                  <X className="h-5 w-5" />
                </motion.button>
              </div>
            </div>

            {/* Main content area with flex structure to ensure proper ordering */}
            <div className="flex flex-col flex-1 overflow-hidden">
              {/* Chat Messages Area */}
              <div 
                className="flex-1 overflow-y-auto p-4 space-y-4 bg-gray-50 dark:bg-gray-900 relative"
                ref={messagesContainerRef}
              >
                {messages.map((message, index) => (
                  <ChatMessage 
                    key={index} 
                    message={message} 
                    isLastBotMessage={
                      index === messages.length - 1 && 
                      (message.sender === 'bot' || message.sender === 'assistant') &&
                      !isLoading
                    }
                  />
                ))}
                
                {/* Loading indicator */}
                {isLoading && (
                  <div className="flex items-start max-w-[85%]">
                    <div className="h-8 w-8 rounded-full bg-blue-100 dark:bg-blue-900/30 flex items-center justify-center mr-3 flex-shrink-0">
                      <MessageSquare className="h-4 w-4 text-blue-600 dark:text-blue-400" />
                    </div>
                    <div className="flex space-x-2 p-3 bg-white dark:bg-gray-800 rounded-lg rounded-tl-none shadow-sm max-w-full">
                      <div className="animate-bounce h-2 w-2 bg-gray-400 dark:bg-gray-500 rounded-full"></div>
                      <div className="animate-bounce delay-100 h-2 w-2 bg-gray-400 dark:bg-gray-500 rounded-full"></div>
                      <div className="animate-bounce delay-200 h-2 w-2 bg-gray-400 dark:bg-gray-500 rounded-full"></div>
                    </div>
                  </div>
                )}
                
                {/* Scroll to bottom button - positioned absolutely within the chat window */}
                <div className="absolute bottom-4 right-4 z-50" style={{ display: showScrollButton ? 'block' : 'none' }}>
                  <motion.button
                    className="bg-white dark:bg-gray-800 shadow-md rounded-full p-2 
                              flex items-center justify-center border border-gray-200 
                              dark:border-gray-700"
                    onClick={scrollToBottom}
                    animate={{ 
                      opacity: 1,
                      scale: 1,
                      y: 0
                    }}
                    initial={{ 
                      opacity: 0,
                      scale: 0.8,
                      y: 10
                    }}
                    transition={{ duration: 0.2 }}
                    whileHover={{ y: -2 }}
                  >
                    <ChevronDown className="h-5 w-5 text-gray-500 dark:text-gray-400" />
                  </motion.button>
                </div>
                
                {/* Invisible element to scroll to */}
                <div ref={messagesEndRef} />
              </div>
              
              {/* Quick Reply Buttons - Show suggestions */}
              {currentSuggestions.length > 0 && (
                <div className="flex-shrink-0 w-full border-t border-gray-200 dark:border-gray-700">
                  <QuickReplyButtons 
                    replies={currentSuggestions} 
                    onReplyClick={handleQuickReplyClick}
                  />
                </div>
              )}
              
              {/* Chat Input - always at the bottom */}
              <div className="flex-shrink-0 border-t border-gray-200 dark:border-gray-700 p-4 bg-white dark:bg-gray-800">
                <ChatInput 
                  onSendMessage={sendMessage} 
                  isLoading={isLoading}
                />
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Chat Button - hidden in fullscreen mode */}
      {!isFullScreen && (
        <motion.button
          className={`flex items-center justify-center w-14 h-14 rounded-full shadow-lg ${
            isOpen || isMinimized 
              ? 'bg-gray-700 dark:bg-gray-700 text-white' 
              : 'bg-blue-600 dark:bg-blue-600 text-white'
          }`}
          onClick={toggleChat}
          variants={chatButtonVariants}
          initial="hidden"
          animate="visible"
          whileHover="hover"
          whileTap="tap"
          aria-label="Toggle chat" 
        >
          {isOpen || isMinimized ? (
            <X className="h-6 w-6" />
          ) : (
            <MessageSquare className="h-6 w-6" />
          )}
        </motion.button>
      )}
    </div>
  );
};

export default ChatWindow;