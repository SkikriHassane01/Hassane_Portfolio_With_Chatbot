import React from 'react';
import { motion } from 'framer-motion';
import { Github, Linkedin, Mail, Globe } from 'lucide-react';

const AboutSection = () => {
  const socialLinks = [
    {
      icon: <Github className="w-5 h-5" />,
      href: "https://github.com/SkikriHassane01",
      label: "GitHub"
    },
    {
      icon: <Linkedin className="w-5 h-5" />,
      href: "https://www.linkedin.com/in/hassane-skikri-8b25b9308/",
      label: "LinkedIn"
    },
    {
      icon: <Mail className="w-5 h-5" />,
      href: "mailto:hassaneskikri@gmail.com",
      label: "Email"
    },
    {
      icon: <Globe className="w-5 h-5" />,
      href: "https://hassaneskikri.me/",
      label: "Website"
    }
  ];

  // Animation variants for the text
  const titleVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0 }
  };

  const nameVariants = {
    hidden: { opacity: 0, x: -20 },
    visible: { opacity: 1, x: 0 }
  };

  const textVariants = {
    hidden: { opacity: 0 },
    visible: { 
      opacity: 1,
      transition: {
        staggerChildren: 0.05
      }
    }
  };

  const letterVariants = {
    hidden: { opacity: 0, y: 50 },
    visible: { 
      opacity: 1, 
      y: 0,
      transition: { type: "spring", stiffness: 100 }
    }
  };

  const text = "Hello, I'm ";
  const name = "Hassane SKIKRI";

  return (
    <section id="about" className="pt-48 pb-48 bg-gray-50 dark:bg-gray-950">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col lg:flex-row lg:items-center lg:space-x-12">
          {/* Image Column - aligned with navbar */}
          <div className="lg:w-1/3">
            <motion.div 
              className="relative mx-auto lg:mx-0"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5 }}
            >
              <div className="aspect-square w-72 h-72 overflow-hidden rounded-full border-4 border-blue-100 dark:border-blue-900/30">
                <img
                  src="../assets/images/About_photo.png"
                  alt="Hassane SKIKRI"
                  className="w-full h-full object-cover object-center"
                />
              </div>
              {/* Decorative Elements - positioned for left alignment */}
              <motion.div 
                className="absolute -bottom-2 -right-2 w-48 h-48 bg-blue-100 dark:bg-blue-900/20 rounded-full -z-10 blur-3xl"
                animate={{ 
                  scale: [1, 1.1, 1],
                  opacity: [0.3, 0.5, 0.3]
                }}
                transition={{ 
                  duration: 4,
                  repeat: Infinity,
                  ease: "easeInOut"
                }}
              />
              <motion.div 
                className="absolute -top-2 -left-2 w-48 h-48 bg-purple-100 dark:bg-purple-900/20 rounded-full -z-10 blur-3xl"
                animate={{ 
                  scale: [1, 1.2, 1],
                  opacity: [0.3, 0.5, 0.3]
                }}
                transition={{ 
                  duration: 5,
                  repeat: Infinity,
                  ease: "easeInOut"
                }}
              />
            </motion.div>
          </div>

          {/* Content Column - to the right */}
          <div className="lg:w-2/3 mt-12 lg:mt-0 space-y-8">
            <div className="space-y-4">
              <motion.h1 
                className="text-4xl font-bold text-gray-900 dark:text-white flex flex-wrap"
                variants={textVariants}
                initial="hidden"
                animate="visible"
              >
                {text.split("").map((char, index) => (
                  <motion.span
                    key={index}
                    variants={letterVariants}
                    className="inline-block"
                  >
                    {char}
                  </motion.span>
                ))}
                <motion.span
                  variants={nameVariants}
                  initial="hidden"
                  animate="visible"
                  transition={{ delay: 0.5, duration: 0.5 }}
                  className="text-blue-600 dark:text-blue-400 ml-2"
                >
                  {name}
                </motion.span>
              </motion.h1>
              
              <motion.h2 
                className="text-xl font-semibold text-gray-600 dark:text-gray-300"
                variants={titleVariants}
                initial="hidden"
                animate="visible"
                transition={{ delay: 1 }}
              >
                Computer Science Student
              </motion.h2>
            </div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 1.2 }}
              className="p-6 rounded-lg bg-white/50 dark:bg-gray-900/50 backdrop-blur-sm shadow-sm border border-gray-100 dark:border-gray-800"
            >
              <p className="text-gray-600 dark:text-gray-300 leading-relaxed">
                I'm a data science enthusiast and computer Science student at École Nationale des Sciences 
                Appliquées de Fès with a deep passion for data science and machine learning. I excel at 
                transforming complex data into actionable insights, driving impactful solutions and 
                continuous improvement.
              </p>
            </motion.div>

            {/* Social Links */}
            <motion.div 
              className="flex gap-5 pt-2"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 1.4 }}
            >
              {socialLinks.map((link, index) => (
                <motion.a
                  key={link.label}
                  href={link.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="p-3.5 rounded-lg bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-300
                    hover:bg-gray-200 dark:hover:bg-gray-700 hover:text-blue-600 dark:hover:text-blue-400 transition-colors duration-200 shadow-sm"
                  aria-label={link.label}
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.95 }}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 1.4 + index * 0.1 }}
                >
                  {link.icon}
                </motion.a>
              ))}
            </motion.div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default AboutSection;