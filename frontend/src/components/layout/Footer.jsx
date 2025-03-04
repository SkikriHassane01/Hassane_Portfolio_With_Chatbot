import React from 'react';
import { motion } from 'framer-motion';
import { Github, Linkedin, Mail, Heart, ArrowUp } from 'lucide-react';

const Footer = () => {
  const currentYear = new Date().getFullYear();
  
  // Navigation links for footer
  const footerLinks = [
    { name: 'About', href: '#about' },
    { name: 'Projects', href: '#projects' },
    { name: 'Skills', href: '#skills' },
    { name: 'Certificates', href: '#certificates' },
    { name: 'Contact', href: '#contact' },
  ];
  
  // Social links
  const socialLinks = [
    {
      icon: <Github className="h-5 w-5" />,
      href: "https://github.com/SkikriHassane01",
      label: "GitHub"
    },
    {
      icon: <Linkedin className="h-5 w-5" />,
      href: "https://www.linkedin.com/in/hassane-skikri-8b25b9308/",
      label: "LinkedIn"
    },
    {
      icon: <Mail className="h-5 w-5" />,
      href: "mailto:hassaneskikri@gmail.com",
      label: "Email"
    }
  ];

  // Scroll to top function
  const scrollToTop = () => {
    window.scrollTo({
      top: 0,
      behavior: 'smooth'
    });
  };

  return (
    <footer className="bg-white dark:bg-gray-900 border-t border-gray-200 dark:border-gray-800">
      {/* Wave Pattern Divider */}
      <div className="relative h-16 overflow-hidden">
        <svg
          className="absolute bottom-0 w-full h-16 text-white dark:text-gray-900 fill-current"
          viewBox="0 0 1200 120"
          preserveAspectRatio="none"
        >
          <path
            d="M0,0V46.29c47.79,22.2,103.59,32.17,158,28,70.36-5.37,136.33-33.31,206.8-37.5C438.64,32.43,512.34,53.67,583,72.05c69.27,18,138.3,24.88,209.4,13.08,36.15-6,69.85-17.84,104.45-29.34C989.49,25,1113-14.29,1200,52.47V0Z"
            opacity=".25"
          ></path>
          <path
            d="M0,0V15.81C13,36.92,27.64,56.86,47.69,72.05,99.41,111.27,165,111,224.58,91.58c31.15-10.15,60.09-26.07,89.67-39.8,40.92-19,84.73-46,130.83-49.67,36.26-2.85,70.9,9.42,98.6,31.56,31.77,25.39,62.32,62,103.63,73,40.44,10.79,81.35-6.69,119.13-24.28s75.16-39,116.92-43.05c59.73-5.85,113.28,22.88,168.9,38.84,30.2,8.66,59,6.17,87.09-7.5,22.43-10.89,48-26.93,60.65-49.24V0Z"
            opacity=".5"
          ></path>
          <path
            d="M0,0V5.63C149.93,59,314.09,71.32,475.83,42.57c43-7.64,84.23-20.12,127.61-26.46,59-8.63,112.48,12.24,165.56,35.4C827.93,77.22,886,95.24,951.2,90c86.53-7,172.46-45.71,248.8-84.81V0Z"
          ></path>
        </svg>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 md:grid-cols-12 gap-8 mb-8">
          {/* Logo and Description */}
          <div className="md:col-span-5 space-y-4">
            <div className="flex items-center space-x-2">
              <span className="font-bold text-xl text-gray-900 dark:text-white">Hassane Skikri</span>
            </div>
            <p className="text-gray-600 dark:text-gray-300 text-sm">
              Computer Science student with a passion for data science and machine learning.
              Building innovative solutions and continuously learning new technologies.
            </p>
            
            {/* Social Links */}
            <div className="flex space-x-3 pt-2">
              {socialLinks.map((social, index) => (
                <motion.a
                  key={index}
                  href={social.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="p-2 rounded-full bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 
                           hover:bg-blue-100 dark:hover:bg-blue-900/30 hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
                  aria-label={social.label}
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.95 }}
                >
                  {social.icon}
                </motion.a>
              ))}
            </div>
          </div>
          
          {/* Quick Links */}
          <div className="md:col-span-3">
            <h3 className="text-sm font-semibold text-gray-900 dark:text-white uppercase tracking-wider mb-4">
              Navigation
            </h3>
            <ul className="space-y-2">
              {footerLinks.map((link) => (
                <li key={link.name}>
                  <a
                    href={link.href}
                    className="text-gray-600 hover:text-blue-600 dark:text-gray-300 dark:hover:text-blue-400 text-sm transition-colors"
                    onClick={(e) => {
                      e.preventDefault();
                      document.querySelector(link.href).scrollIntoView({
                        behavior: 'smooth'
                      });
                    }}
                  >
                    {link.name}
                  </a>
                </li>
              ))}
            </ul>
          </div>
          
          {/* Contact */}
          <div className="md:col-span-4">
            <h3 className="text-sm font-semibold text-gray-900 dark:text-white uppercase tracking-wider mb-4">
              Contact Me
            </h3>
            <ul className="space-y-3">
              <li className="flex items-start">
                <Mail className="h-5 w-5 text-gray-500 dark:text-gray-400 mr-3 mt-0.5" />
                <a
                  href="mailto:hassaneskikri@gmail.com"
                  className="text-gray-600 hover:text-blue-600 dark:text-gray-300 dark:hover:text-blue-400 text-sm transition-colors"
                >
                  hassaneskikri@gmail.com
                </a>
              </li>
              <li className="flex items-start">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-gray-500 dark:text-gray-400 mr-3 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z" />
                </svg>
                <a
                  href="tel:+212635607145"
                  className="text-gray-600 hover:text-blue-600 dark:text-gray-300 dark:hover:text-blue-400 text-sm transition-colors"
                >
                  +212 635 607 145
                </a>
              </li>
              <li className="flex items-start">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-gray-500 dark:text-gray-400 mr-3 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
                <span className="text-gray-600 dark:text-gray-300 text-sm">
                  Fes-Meknes, Morocco
                </span>
              </li>
            </ul>

            {/* Newsletter/Availability */}
            <div className="mt-6 bg-gray-50 dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                <span className="inline-block h-2 w-2 rounded-full bg-green-500 mr-2"></span>
                Currently available for new opportunities and collaborations
              </p>
              <a 
                href="#contact"
                onClick={(e) => {
                  e.preventDefault();
                  document.querySelector('#contact').scrollIntoView({
                    behavior: 'smooth'
                  });
                }}
                className="inline-block mt-2 text-sm font-medium text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300"
              >
                Get in touch →
              </a>
            </div>
          </div>
        </div>
        
        {/* Divider */}
        <div className="border-t border-gray-200 dark:border-gray-800 py-8">
          <div className="flex flex-col md:flex-row justify-between items-center space-y-4 md:space-y-0">
            {/* Copyright */}
            <div className="flex items-center space-x-1 text-gray-500 dark:text-gray-400 text-sm">
              <span>© {currentYear}</span>
              <span>Hassane Skikri.</span>
              <span>All rights reserved.</span>
              <span className="inline-flex items-center">
                Made with <Heart className="h-3 w-3 mx-1 text-red-500" fill="currentColor" /> in Morocco
              </span>
            </div>
            
            {/* Back to top button */}
            <motion.button
              onClick={scrollToTop}
              className="flex items-center space-x-1 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 text-sm group transition-colors"
              whileHover={{ y: -2 }}
              whileTap={{ y: 0 }}
            >
              <span>Back to top</span>
              <ArrowUp className="h-4 w-4 group-hover:translate-y-[-2px] transition-transform" />
            </motion.button>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;