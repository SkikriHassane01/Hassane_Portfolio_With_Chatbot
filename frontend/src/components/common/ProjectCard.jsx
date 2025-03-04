import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Github, ExternalLink, Tag } from 'lucide-react';

const ProjectCard = ({ project }) => {
  const {
    title,
    description,
    image,
    technologies,
    category,
    githubLink,
    demoLink,
    tags = []
  } = project;

  const [showAllTechnologies, setShowAllTechnologies] = useState(false);
  
  // Generate a placeholder based on category with a gradient background
  const getCategoryColor = (category) => {
    const colorMap = {
      'End-to-End': 'from-blue-600 to-indigo-800',
      'Computer Vision': 'from-blue-500 to-indigo-600',
      'Deep Learning': 'from-purple-600 to-indigo-700',
      'Machine Learning': 'from-indigo-500 to-blue-700',
      'Web Scraping': 'from-green-500 to-emerald-700',
      'Data Analysis': 'from-blue-500 to-cyan-700',
      'Tutorial': 'from-orange-500 to-amber-700',
      'Other': 'from-gray-500 to-slate-700'
    };
    
    return colorMap[category] || 'from-gray-600 to-slate-800';
  };

  // Get an emoji for the category
  const getCategoryEmoji = (category) => {
    const emojiMap = {
      'End-to-End': '🚀',
      'Computer Vision': '👁️',
      'Deep Learning': '🧠',
      'Machine Learning': '⚙️',
      'Web Scraping': '🕸️',
      'Data Analysis': '📊',
      'Tutorial': '📚',
      'Other': '🔧'
    };
    
    return emojiMap[category] || '📁';
  };

  // Animation variants
  const cardVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { 
      opacity: 1, 
      y: 0,
      transition: { duration: 0.5 }
    },
    hover: {
      y: -5,
      transition: { duration: 0.2 }
    }
  };

  return (
    <motion.div
      className="bg-white dark:bg-gray-800 rounded-xl overflow-hidden shadow-md hover:shadow-lg transition-shadow duration-300 border border-gray-100 dark:border-gray-700 h-full flex flex-col"
      variants={cardVariants}
      initial="hidden"
      animate="visible"
      whileHover="hover"
      layoutId={`project-${title.replace(/\s+/g, '-').toLowerCase()}`}
    >
      {/* Project Image */}
      <div className="aspect-video overflow-hidden relative">
        {image ? (
          <img
            src={image}
            alt={`${title} preview`}
            className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-105"
            onError={(e) => {
              e.target.onerror = null;
              // If image fails to load, replace with div
              e.target.style.display = 'none';
              e.target.parentNode.classList.add(`bg-gradient-to-br`, getCategoryColor(category));
              const emojiSpan = document.createElement('span');
              emojiSpan.textContent = getCategoryEmoji(category);
              emojiSpan.className = 'absolute inset-0 flex items-center justify-center text-5xl';
              e.target.parentNode.appendChild(emojiSpan);
            }}
          />
        ) : (
          <div 
            className={`w-full h-full flex items-center justify-center bg-gradient-to-br ${getCategoryColor(category)}`}
            aria-label={`${title} preview placeholder`}
          >
            <span className="text-5xl">{getCategoryEmoji(category)}</span>
            <span className="absolute bottom-10 text-white text-xl font-bold">{title.slice(0, 20)}</span>
          </div>
        )}
        <div className="absolute top-3 left-3">
          <span className="px-2 py-1 text-xs font-medium bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-100 rounded-md">
            {category}
          </span>
        </div>
      </div>

      {/* Project Content */}
      <div className="p-5 space-y-4 flex-1 flex flex-col">
        <h3 className="text-xl font-bold text-gray-900 dark:text-white">{title}</h3>
        
        <p className="text-gray-600 dark:text-gray-300 text-sm line-clamp-3 flex-grow">
          {description}
        </p>

        {/* Technologies with expand/collapse functionality */}
        <div className="flex flex-col gap-2 mt-2">
          <div className="flex flex-wrap gap-2">
            {(showAllTechnologies ? technologies : technologies.slice(0, 3)).map((tech, index) => (
              <span 
                key={index}
                className="inline-flex items-center px-2 py-1 rounded-md text-xs font-medium bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-200"
              >
                <Tag className="w-3 h-3 mr-1" />
                {tech}
              </span>
            ))}
          </div>
          
          {/* Removed separate +more button */}
        </div>

        {/* Links */}
        <div className="flex justify-between items-center pt-3 border-t border-gray-200 dark:border-gray-700 mt-auto">
          <div className="flex space-x-3">
            {githubLink && (
              <a
                href={githubLink}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center text-sm text-gray-700 hover:text-blue-600 dark:text-gray-300 dark:hover:text-blue-400 transition-colors"
                aria-label={`View ${title} source code on GitHub`}
              >
                <Github className="w-4 h-4 mr-1" />
                <span>Code</span>
              </a>
            )}
            
            {demoLink && (
              <a
                href={demoLink}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center text-sm text-gray-700 hover:text-blue-600 dark:text-gray-300 dark:hover:text-blue-400 transition-colors"
                aria-label={`View ${title} live demo`}
              >
                <ExternalLink className="w-4 h-4 mr-1" />
                <span>Live Demo</span>
              </a>
            )}
          </div>
          
          <motion.button
            onClick={() => setShowAllTechnologies(!showAllTechnologies)}
            className="text-sm font-medium text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 transition-colors"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            {showAllTechnologies ? 'Less' : 'Details'}
          </motion.button>
        </div>
      </div>
    </motion.div>
  );
};

export default ProjectCard;