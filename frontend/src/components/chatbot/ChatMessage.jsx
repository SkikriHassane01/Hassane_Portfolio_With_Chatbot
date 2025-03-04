import React from 'react';
import { motion } from 'framer-motion';
import { MessageSquare, User } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

const ChatMessage = ({ message, isLastBotMessage }) => {
  // Destructure with fallbacks to prevent errors
  const { text = '', content = '', sender = '', timestamp } = message || {};
  
  // Use either text or content property, whichever is available
  const messageContent = text || content || '';
  
  // Check if it's a bot message
  const isBot = sender === 'bot' || sender === 'assistant';
  
  // Format timestamp if available
  const formattedTime = timestamp ? new Date(timestamp).toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit'
  }) : '';
  
  // Custom animation for message entry
  const messageVariants = {
    hidden: { 
      opacity: 0,
      y: 10,
      scale: 0.95
    },
    visible: { 
      opacity: 1,
      y: 0,
      scale: 1,
      transition: {
        type: 'spring',
        damping: 25,
        stiffness: 500
      }
    }
  };

  

  return (
    <motion.div
      className={`flex items-start ${isBot ? 'justify-start' : 'justify-end'} w-full mb-4`}
      variants={messageVariants}
      initial="hidden"
      animate="visible"
    >
      {isBot && (
        <div className="h-8 w-8 rounded-full bg-blue-100 dark:bg-blue-900/30 flex items-center justify-center mr-3 flex-shrink-0">
          <MessageSquare className="h-4 w-4 text-blue-600 dark:text-blue-400" />
        </div>
      )}
      
      <div 
        className={`p-3.5 ${
          isBot 
            ? 'bg-[#3B3B37] text-white rounded-lg rounded-tl-none shadow-sm max-w-[80%]' 
            : 'bg-[#1C1B1B] text-white rounded-lg rounded-tr-none shadow-sm ml-auto max-w-[80%]'
        }`}
      >
        <div className="markdown-content prose prose-sm dark:prose-invert max-w-none prose-p:my-0 prose-headings:text-white prose-headings:mb-2 prose-headings:mt-4 prose-li:my-0 prose-ul:my-2 prose-ol:my-2">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              code: ({ node, inline, className, children, ...props }) => (
                <code
                  className={`${className} ${
                    inline
                      ? 'bg-gray-700 text-gray-100 px-1 py-0.5 rounded text-sm'
                      : 'block bg-gray-700 text-gray-100 p-2 rounded-md my-2 overflow-x-auto'
                  }`}
                  {...props}
                >
                  {children}
                </code>
              ),
              a: ({ node, className, children, ...props }) => (
                <a
                  className="text-blue-400 hover:underline hover:text-blue-300 transition-colors"
                  target="_blank"
                  rel="noopener noreferrer"
                  {...props}
                >
                  {children}
                </a>
              ),
              ul: ({ node, className, children, ...props }) => (
                <ul className="list-disc list-inside my-2 space-y-1" {...props}>
                  {children}
                </ul>
              ),
              ol: ({ node, className, children, ...props }) => (
                <ol className="list-decimal pl-5 my-2" {...props}>
                  {children}
                </ol>
              ),
              li: ({ node, className, children, ...props }) => (
                <li className="mb-1" {...props}>
                  {children}
                </li>
              ),
              p: ({ node, className, children, ...props }) => (
                <p className="mb-2 last:mb-0" {...props}>
                  {children}
                </p>
              ),
              pre: ({ node, children, ...props }) => (
                <pre className="bg-gray-700 p-2 rounded-md my-2 overflow-x-auto" {...props}>
                  {children}
                </pre>
              )
            }}
          >
            {messageContent}
          </ReactMarkdown>
        </div>
        
        {/* Show timestamp if available */}
        {timestamp && (
          <div 
            className={`text-xs mt-1.5 ${
              isBot ? 'text-gray-300' : 'text-gray-300'
            }`}
          >
            {formattedTime}
          </div>
        )}
      </div>
      
      {!isBot && (
        <div className="h-8 w-8 rounded-full bg-gray-100 dark:bg-gray-700 flex items-center justify-center ml-3 flex-shrink-0">
          <User className="h-4 w-4 text-gray-600 dark:text-gray-400" />
        </div>
      )}
    </motion.div>
  );
};

export default ChatMessage;