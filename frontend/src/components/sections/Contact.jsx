import React, { useState, useRef } from 'react';
import { motion } from 'framer-motion';
import { 
  Send, 
  Mail, 
  Phone, 
  MapPin, 
  Github, 
  Linkedin, 
  Globe,
  ArrowRight,
  CheckCircle,
  AlertCircle,
  Loader
} from 'lucide-react';

import resumePDF from '../../assets/docs/Skikri_Hassane_CV.pdf';
const ContactSection = () => {
  // Form state
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    subject: '',
    message: ''
  });
  
  const [formStatus, setFormStatus] = useState({
    submitted: false,
    submitting: false,
    error: null,
    success: false
  });

  const [validationErrors, setValidationErrors] = useState({});
  const formRef = useRef(null);

  // Contact information
  const contactInfo = [
    {
      icon: <Mail className="h-6 w-6" />,
      title: 'Email',
      value: 'hassaneskikri@gmail.com',
      link: 'mailto:hassaneskikri@gmail.com'
    },
    {
      icon: <Phone className="h-6 w-6" />,
      title: 'Phone',
      value: '+212 635607145',
      link: 'tel:+212635607145'
    },
    {
      icon: <MapPin className="h-6 w-6" />,
      title: 'Location',
      value: 'Fez, Morocco',
      link: 'https://maps.google.com/?q=Fez,Morocco'
    }
  ];

  // Social links
  const socialLinks = [
    {
      icon: <Github className="h-5 w-5" />,
      label: 'GitHub',
      href: 'https://github.com/SkikriHassane01',
      color: 'hover:bg-gray-800 hover:text-white'
    },
    {
      icon: <Linkedin className="h-5 w-5" />,
      label: 'LinkedIn',
      href: 'https://www.linkedin.com/in/hassane-skikri-8b25b9308/',
      color: 'hover:bg-blue-600 hover:text-white'
    },
    {
      icon: <Globe className="h-5 w-5" />,
      label: 'Website',
      href: 'https://hassaneskikri.me/',
      color: 'hover:bg-green-600 hover:text-white'
    },
    {
      icon: <Mail className="h-5 w-5" />,
      label: 'WhatsApp',
      href: 'https://wa.me/212635607145',
      color: 'hover:bg-green-500 hover:text-white'
    }
  ];

  // Animation variants
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.5 }
    }
  };

  // Form validation
  const validateForm = () => {
    const errors = {};
    
    if (!formData.name.trim()) {
      errors.name = 'Name is required';
    }
    
    if (!formData.email.trim()) {
      errors.email = 'Email is required';
    } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(formData.email)) {
      errors.email = 'Email is invalid';
    }
    
    if (!formData.subject.trim()) {
      errors.subject = 'Subject is required';
    }
    
    if (!formData.message.trim()) {
      errors.message = 'Message is required';
    } else if (formData.message.trim().length < 20) {
      errors.message = 'Message must be at least 20 characters';
    }
    
    setValidationErrors(errors);
    return Object.keys(errors).length === 0;
  };

  // Form handlers
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
    
    // Clear validation error when typing
    if (validationErrors[name]) {
      setValidationErrors(prev => ({
        ...prev,
        [name]: null
      }));
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!validateForm()) {
      return;
    }
    
    setFormStatus({
      ...formStatus,
      submitting: true
    });
    
    try {
      // Fallback to form submission API like Formspree
      const response = await fetch('https://formspree.io/f/mzzdyavy', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });
      
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
      
      setFormStatus({
        submitted: true,
        submitting: false,
        error: null,
        success: true
      });
      
      // Reset form after successful submission
      setFormData({
        name: '',
        email: '',
        subject: '',
        message: ''
      });
      
      // After 5 seconds, reset the success state
      setTimeout(() => {
        setFormStatus(prev => ({
          ...prev,
          submitted: false,
          success: false
        }));
      }, 5000);
      
    } catch (error) {
      console.error('Contact form error:', error);
      setFormStatus({
        submitted: false,
        submitting: false,
        error: "Failed to send message. Please try again later.",
        success: false
      });
    }
  };

  return (
    <section id="contact" className="min-h-screen py-20 bg-gray-50 dark:bg-gray-950">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="text-center mb-16"
        >
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">
            Get In Touch
          </h2>
          <p className="text-lg text-gray-600 dark:text-gray-300 max-w-2xl mx-auto">
            Have a question or want to work together? Feel free to contact me through the form below 
            or directly via email or social media.
          </p>
        </motion.div>

        <div className="grid md:grid-cols-3 gap-12">
          {/* Contact Info */}
          <motion.div
            className="space-y-8 md:col-span-1"
            variants={containerVariants}
            initial="hidden"
            animate="visible"
          >
            <div className="space-y-6">
              <motion.h3 
                className="text-xl font-semibold text-gray-900 dark:text-white"
                variants={itemVariants}
              >
                Contact Information
              </motion.h3>
              
              {contactInfo.map((item, index) => (
                <motion.a
                  key={index}
                  href={item.link}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-start p-4 space-x-4 rounded-lg bg-white dark:bg-gray-800 shadow-sm hover:shadow-md border border-gray-100 dark:border-gray-700 transition-all duration-300"
                  variants={itemVariants}
                  whileHover={{ scale: 1.02, y: -3 }}
                >
                  <div className="p-2.5 bg-blue-50 dark:bg-blue-900/20 rounded-full text-blue-600 dark:text-blue-400">
                    {item.icon}
                  </div>
                  <div>
                    <p className="font-medium text-gray-900 dark:text-white mb-1">{item.title}</p>
                    <p className="text-gray-600 dark:text-gray-300 break-all">{item.value}</p>
                  </div>
                </motion.a>
              ))}
            </div>

            <motion.div 
              className="space-y-4"
              variants={itemVariants}
            >
              <h3 className="text-xl font-semibold text-gray-900 dark:text-white">
                Connect With Me
              </h3>
              <div className="flex flex-wrap gap-3">
                {socialLinks.map((social, index) => (
                  <motion.a
                    key={index}
                    href={social.href}
                    target="_blank"
                    rel="noopener noreferrer"
                    className={`p-3 rounded-full bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 transition-colors duration-300 ${social.color}`}
                    aria-label={social.label}
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    {social.icon}
                  </motion.a>
                ))}
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-4">
                Available for freelance opportunities, collaborations, and remote positions.
              </p>
            </motion.div>
          </motion.div>

          {/* Contact Form */}
          <motion.div
            className="md:col-span-2"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-md p-6 sm:p-8 border border-gray-100 dark:border-gray-700">
              <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-6">
                Send Me a Message
              </h3>

              {formStatus.success ? (
                <motion.div
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg p-4 flex items-start space-x-3 mb-6"
                >
                  <CheckCircle className="h-6 w-6 text-green-600 dark:text-green-400 flex-shrink-0 mt-0.5" />
                  <div>
                    <h4 className="font-medium text-green-800 dark:text-green-300">Message sent successfully!</h4>
                    <p className="text-green-700 dark:text-green-400 mt-1 text-sm">
                      Thank you for reaching out. I'll get back to you as soon as possible.
                    </p>
                  </div>
                </motion.div>
              ) : null}

              {formStatus.error ? (
                <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4 flex items-start space-x-3 mb-6">
                  <AlertCircle className="h-6 w-6 text-red-600 dark:text-red-400 flex-shrink-0 mt-0.5" />
                  <div>
                    <h4 className="font-medium text-red-800 dark:text-red-300">Something went wrong</h4>
                    <p className="text-red-700 dark:text-red-400 mt-1 text-sm">{formStatus.error}</p>
                  </div>
                </div>
              ) : null}

              <form ref={formRef} onSubmit={handleSubmit} className="space-y-5">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Name field */}
                  <div>
                    <label 
                      htmlFor="name" 
                      className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1"
                    >
                      Your Name
                    </label>
                    <input
                      type="text"
                      id="name"
                      name="name"
                      value={formData.name}
                      onChange={handleInputChange}
                      className={`w-full px-4 py-2.5 rounded-lg border ${
                        validationErrors.name 
                          ? 'border-red-300 dark:border-red-700 bg-red-50 dark:bg-red-900/10' 
                          : 'border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700'
                      } text-gray-900 dark:text-white placeholder-gray-400 dark:placeholder-gray-500 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition`}
                      placeholder="John Doe"
                      disabled={formStatus.submitting}
                    />
                    {validationErrors.name && (
                      <p className="mt-1.5 text-sm text-red-600 dark:text-red-400">
                        {validationErrors.name}
                      </p>
                    )}
                  </div>

                  {/* Email field */}
                  <div>
                    <label 
                      htmlFor="email" 
                      className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1"
                    >
                      Your Email
                    </label>
                    <input
                      type="email"
                      id="email"
                      name="email"
                      value={formData.email}
                      onChange={handleInputChange}
                      className={`w-full px-4 py-2.5 rounded-lg border ${
                        validationErrors.email 
                          ? 'border-red-300 dark:border-red-700 bg-red-50 dark:bg-red-900/10' 
                          : 'border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700'
                      } text-gray-900 dark:text-white placeholder-gray-400 dark:placeholder-gray-500 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition`}
                      placeholder="johndoe@example.com"
                      disabled={formStatus.submitting}
                    />
                    {validationErrors.email && (
                      <p className="mt-1.5 text-sm text-red-600 dark:text-red-400">
                        {validationErrors.email}
                      </p>
                    )}
                  </div>
                </div>

                {/* Subject field */}
                <div>
                  <label 
                    htmlFor="subject" 
                    className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1"
                  >
                    Subject
                  </label>
                  <input
                    type="text"
                    id="subject"
                    name="subject"
                    value={formData.subject}
                    onChange={handleInputChange}
                    className={`w-full px-4 py-2.5 rounded-lg border ${
                      validationErrors.subject 
                        ? 'border-red-300 dark:border-red-700 bg-red-50 dark:bg-red-900/10' 
                        : 'border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700'
                    } text-gray-900 dark:text-white placeholder-gray-400 dark:placeholder-gray-500 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition`}
                    placeholder="Project inquiry, job opportunity, etc."
                    disabled={formStatus.submitting}
                  />
                  {validationErrors.subject && (
                    <p className="mt-1.5 text-sm text-red-600 dark:text-red-400">
                      {validationErrors.subject}
                    </p>
                  )}
                </div>

                {/* Message field */}
                <div>
                  <label 
                    htmlFor="message" 
                    className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1"
                  >
                    Your Message
                  </label>
                  <textarea
                    id="message"
                    name="message"
                    value={formData.message}
                    onChange={handleInputChange}
                    rows={5}
                    className={`w-full px-4 py-2.5 rounded-lg border ${
                      validationErrors.message 
                        ? 'border-red-300 dark:border-red-700 bg-red-50 dark:bg-red-900/10' 
                        : 'border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700'
                    } text-gray-900 dark:text-white placeholder-gray-400 dark:placeholder-gray-500 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition resize-none`}
                    placeholder="Let me know how I can help you..."
                    disabled={formStatus.submitting}
                  />
                  {validationErrors.message && (
                    <p className="mt-1.5 text-sm text-red-600 dark:text-red-400">
                      {validationErrors.message}
                    </p>
                  )}
                </div>

                {/* Submit button */}
                <div className="pt-4">
                  <motion.button
                    type="submit"
                    className={`w-full sm:w-auto px-6 py-3 flex items-center justify-center space-x-2 rounded-lg ${
                      formStatus.submitting 
                        ? 'bg-blue-400 dark:bg-blue-700 cursor-not-allowed' 
                        : 'bg-blue-600 hover:bg-blue-700 dark:bg-blue-600 dark:hover:bg-blue-500'
                    } text-white font-medium transition-colors shadow-sm`}
                    disabled={formStatus.submitting}
                    whileHover={{ scale: formStatus.submitting ? 1 : 1.02 }}
                    whileTap={{ scale: formStatus.submitting ? 1 : 0.98 }}
                  >
                    {formStatus.submitting ? (
                      <>
                        <Loader className="h-5 w-5 animate-spin" />
                        <span>Sending...</span>
                      </>
                    ) : (
                      <>
                        <Send className="h-5 w-5" />
                        <span>Send Message</span>
                      </>
                    )}
                  </motion.button>
                </div>
              </form>
            </div>
          </motion.div>
        </div>

        {/* Availability Status */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.4 }}
          className="mt-16 text-center"
        >
          <div className="inline-flex items-center space-x-2 px-4 py-2 rounded-full bg-green-100 dark:bg-green-900/20 text-green-800 dark:text-green-300">
            <span className="relative flex h-3 w-3">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-3 w-3 bg-green-500"></span>
            </span>
            <span className="text-sm font-medium">Available for new opportunities</span>
          </div>

          <div className="mt-6">
            <motion.a
              href={resumePDF}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center space-x-2 text-blue-600 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300 font-medium transition-colors"
              whileHover={{ x: 5 }}
            >
              <span>Download my resume</span>
              <ArrowRight className="h-4 w-4" />
            </motion.a>
          </div>
        </motion.div>
      </div>
    </section>
  );
};

export default ContactSection;