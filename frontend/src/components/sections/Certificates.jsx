import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Calendar, Award, ExternalLink, X } from 'lucide-react';
import certificates from '../../utils/certificates';
const CertificatesSection = () => {
  const [selectedCertificate, setSelectedCertificate] = useState(null);
  
  // Pagination state
  const [currentPage, setCurrentPage] = useState(1);
  const certificatesPerPage = 4;
  


  // Animation variants
  const sectionVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        when: "beforeChildren",
        staggerChildren: 0.2
      }
    }
  };

  const cardVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { 
        duration: 0.4,
        ease: "easeOut"
      }
    },
    hover: {
      y: -5,
      boxShadow: "0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)",
      transition: {
        duration: 0.2
      }
    }
  };

  const modalVariants = {
    hidden: { opacity: 0, scale: 0.8 },
    visible: { 
      opacity: 1, 
      scale: 1,
      transition: {
        duration: 0.3,
        ease: "easeOut"
      }
    },
    exit: {
      opacity: 0,
      scale: 0.8,
      transition: {
        duration: 0.2
      }
    }
  };

  const backdropVariants = {
    hidden: { opacity: 0 },
    visible: { opacity: 1 },
    exit: { opacity: 0 }
  };

  const getRandomPlaceholderImage = (seed, color) => {
    // Generate a placeholder gradient based on certificate color without using btoa
    const colors = {
      blue: 'linear-gradient(135deg, #3182ce 0%, #63b3ed 100%)',
      purple: 'linear-gradient(135deg, #6b46c1 0%, #9f7aea 100%)',
      green: 'linear-gradient(135deg, #38a169 0%, #9ae6b4 100%)',
      orange: 'linear-gradient(135deg, #dd6b20 0%, #fbd38d 100%)',
    };
    
    const gradient = colors[color] || colors.blue;
    const firstLetters = seed.split(' ')
      .map(word => word[0])
      .join('')
      .slice(0, 2)
      .toUpperCase();
      
    // Simply return gradient style
    return gradient;
  };

  // Function to get image source (with fallback to placeholder)
  const getImageSource = (certificate) => {
    // Try to use actual image, fallback to generated placeholder
    if (certificate.image) {
      return `url(${certificate.image})`;
    } else {
      return getRandomPlaceholderImage(certificate.issuer, certificate.color);
    }
  };
  
  // Calculate pagination
  const indexOfLastCertificate = currentPage * certificatesPerPage;
  const indexOfFirstCertificate = indexOfLastCertificate - certificatesPerPage;
  const currentCertificates = certificates.slice(indexOfFirstCertificate, indexOfLastCertificate);
  const totalPages = Math.ceil(certificates.length / certificatesPerPage);

  return (
    <section id="certificates" className="min-h-screen py-20 bg-white dark:bg-gray-900">
      <motion.div 
        className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8"
        variants={sectionVariants}
        initial="hidden"
        animate="visible"
        viewport={{ once: true }}
      >
        {/* Section Header */}
        <motion.div
          className="text-center mb-16"
          variants={cardVariants}
        >
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">
            Certificates & Credentials
          </h2>
          <p className="text-lg text-gray-600 dark:text-gray-300 max-w-2xl mx-auto">
            Professional certifications that demonstrate my expertise in various technology domains,
            from data science and machine learning to database management and business analytics.
          </p>
        </motion.div>

        {/* Certificates Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-2 gap-8">
          {currentCertificates.map((certificate) => (
            <motion.div
              key={certificate.id}
              className="bg-white dark:bg-gray-800 rounded-xl overflow-hidden shadow-md hover:shadow-lg 
                        transition-all border border-gray-100 dark:border-gray-700 h-full cursor-pointer"
              variants={cardVariants}
              whileHover="hover"
              onClick={() => setSelectedCertificate(certificate)}
              layout
            >
              <div className="flex flex-col h-full">
                {/* Certificate Image */}
                <div className="aspect-[16/9] overflow-hidden relative group">
                  <div 
                    className="w-full h-full bg-contain bg-center flex items-center justify-center"
                    style={{ 
                      backgroundImage: getImageSource(certificate),
                      backgroundColor: '#f0f4f8'
                    }}
                    aria-label={`${certificate.title} certificate`}
                  >
                    {!certificate.image && (
                      <div className="text-white text-5xl font-bold opacity-30">
                        {certificate.issuer.slice(0, 2).toUpperCase()}
                      </div>
                    )}
                    {/* Purple overlay with hover effect - more subtle */}
                    <div className="absolute inset-0 bg-purple-600/30 mix-blend-overlay transition-opacity duration-300 group-hover:opacity-40"></div>
                  </div>
                  <div className="absolute bottom-0 left-0 w-full h-1/4 bg-gradient-to-t from-purple-900/40 to-transparent"></div>
                </div>

                {/* Certificate Content */}
                <div className="p-6 flex flex-col flex-grow">
                  <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-2 line-clamp-2">
                    {certificate.title}
                  </h3>
                  
                  <div className="flex items-center text-sm text-gray-500 dark:text-gray-400 mb-4">
                    <span className="font-medium">{certificate.issuer}</span>
                    <span className="mx-2">•</span>
                    <span className="flex items-center">
                      <Calendar className="w-4 h-4 mr-1" />
                      {certificate.date}
                    </span>
                  </div>
                  
                  <p className="text-gray-600 dark:text-gray-300 text-sm mb-4 line-clamp-3">
                    {certificate.description}
                  </p>
                  
                  {/* Certificate Skills - first 4 with counter for rest */}
                  <div className="flex flex-wrap gap-2 mt-auto">
                    {certificate.skills.slice(0, 4).map((skill, index) => (
                      <span 
                        key={index}
                        className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-200"
                      >
                        {skill}
                      </span>
                    ))}
                    {certificate.skills.length > 4 && (
                      <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-50 dark:bg-gray-800 text-gray-500 dark:text-gray-400">
                        +{certificate.skills.length - 4} more
                      </span>
                    )}
                  </div>
                </div>

                {/* View Details Button */}
                <div className="px-6 py-4 border-t border-gray-100 dark:border-gray-700">
                  <button
                    className="w-full flex justify-center items-center py-2 px-4 bg-blue-50 dark:bg-blue-900/20 
                              text-blue-600 dark:text-blue-400 rounded-lg hover:bg-blue-100 dark:hover:bg-blue-900/30 
                              transition-colors font-medium text-sm"
                    onClick={(e) => {
                      e.stopPropagation();
                      setSelectedCertificate(certificate);
                    }}
                  >
                    <Award className="w-4 h-4 mr-2" />
                    View Details
                  </button>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
        
        {/* Pagination Controls */}
        {totalPages > 1 && (
          <div className="flex justify-center mt-12 space-x-4">
            <button
              onClick={() => setCurrentPage(prev => Math.max(prev - 1, 1))}
              disabled={currentPage === 1}
              className={`px-4 py-2 rounded-lg border ${
                currentPage === 1 
                  ? 'bg-gray-100 dark:bg-gray-800 text-gray-400 dark:text-gray-600 cursor-not-allowed' 
                  : 'bg-white dark:bg-gray-800 border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700'
              }`}
              aria-label="Previous page"
            >
              Previous
            </button>
            
            <span className="flex items-center justify-center px-4 py-2 bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 rounded-lg border border-gray-300 dark:border-gray-600">
              Page {currentPage} of {totalPages}
            </span>
            
            <button
              onClick={() => setCurrentPage(prev => Math.min(prev + 1, totalPages))}
              disabled={currentPage === totalPages}
              className={`px-4 py-2 rounded-lg border ${
                currentPage === totalPages 
                  ? 'bg-gray-100 dark:bg-gray-800 text-gray-400 dark:text-gray-600 cursor-not-allowed' 
                  : 'bg-white dark:bg-gray-800 border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700'
              }`}
              aria-label="Next page"
            >
              Next
            </button>
          </div>
        )}
        
        {/* Certificate Detail Modal */}
        {selectedCertificate && (
          <motion.div
            className="fixed inset-0 z-50 flex items-center justify-center p-4"
            initial="hidden"
            animate="visible"
            exit="exit"
            variants={backdropVariants}
          >
            <motion.div 
              className="absolute inset-0 bg-black/50 dark:bg-black/70 backdrop-blur-sm"
              onClick={() => setSelectedCertificate(null)}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            />
            
            <motion.div
              className="bg-white dark:bg-gray-800 rounded-xl shadow-xl overflow-hidden max-w-4xl w-full max-h-[90vh] relative z-10"
              variants={modalVariants}
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex flex-col md:flex-row h-full max-h-[90vh]">
                {/* Left side - Issuer and Date instead of Image */}
                <div className="w-full md:w-2/5 relative overflow-hidden bg-gray-100 dark:bg-gray-900 flex items-center justify-center">
                  <div 
                    className="w-full h-full flex items-center justify-center"
                    style={{
                      background: getRandomPlaceholderImage(selectedCertificate.issuer, selectedCertificate.color),
                      minHeight: '200px',
                      height: '100%',
                      position: 'relative'
                    }}
                    aria-label={`${selectedCertificate.title} certificate`}
                  >
                    {selectedCertificate.image && (
                      <div 
                        className="absolute inset-0 opacity-20 bg-center bg-contain bg-no-repeat" 
                        style={{ backgroundImage: `url(${selectedCertificate.image})` }}
                      />
                    )}
                    <div className="flex flex-col items-center justify-center text-white p-4 relative z-10">
                      <Award className="w-16 h-16 mb-4 opacity-80" />
                      <div className="text-4xl font-bold text-center">
                        {selectedCertificate.issuer}
                      </div>
                      <div className="mt-2 opacity-80">
                        {selectedCertificate.date}
                      </div>
                    </div>
                  </div>
                </div>
                
                {/* Right side - Content */}
                <div className="w-full md:w-3/5 p-6 overflow-y-auto">
                  <div className="flex justify-between items-start mb-4">
                    <h3 className="text-2xl font-bold text-gray-900 dark:text-white">
                      {selectedCertificate.title}
                    </h3>
                    <button
                      className="p-1 rounded-full hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
                      onClick={() => setSelectedCertificate(null)}
                      aria-label="Close modal"
                    >
                      <X className="w-5 h-5 text-gray-500 dark:text-gray-400" />
                    </button>
                  </div>
                  
                  <div className="flex items-center text-sm text-gray-500 dark:text-gray-400 mb-6">
                    <span className="font-medium">{selectedCertificate.issuer}</span>
                    <span className="mx-2">•</span>
                    <span className="flex items-center">
                      <Calendar className="w-4 h-4 mr-1" />
                      {selectedCertificate.date}
                    </span>
                  </div>
                  
                  <div className="mb-6">
                    <h4 className="text-sm font-semibold uppercase text-gray-500 dark:text-gray-400 mb-2">
                      Description
                    </h4>
                    <p className="text-gray-700 dark:text-gray-300">
                      {selectedCertificate.description}
                    </p>
                  </div>
                  
                  <div className="mb-6">
                    <h4 className="text-sm font-semibold uppercase text-gray-500 dark:text-gray-400 mb-2">
                      Skills
                    </h4>
                    <div className="flex flex-wrap gap-2">
                      {selectedCertificate.skills.map((skill, index) => (
                        <span 
                          key={index}
                          className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-200"
                        >
                          {skill}
                        </span>
                      ))}
                    </div>
                  </div>
                  
                  <div className="pt-4 border-t border-gray-200 dark:border-gray-700">
                    <a
                      href={selectedCertificate.verificationLink}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center px-4 py-2 bg-blue-600 hover:bg-blue-700 
                                dark:bg-blue-700 dark:hover:bg-blue-600 text-white rounded-lg 
                                transition-colors duration-200 font-medium"
                    >
                      <ExternalLink className="w-4 h-4 mr-2" />
                      Verify Certificate
                    </a>
                  </div>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
        
        {/* Call to Action removed */}
        <motion.div 
          className="mt-16 text-center"
          variants={cardVariants}
        >
          <p className="text-gray-600 dark:text-gray-300">
            I'm continuously learning and adding new skills to my repertoire.
            Check back regularly for updates on my latest certifications.
          </p>
        </motion.div>
      </motion.div>
    </section>
  );
};

export default CertificatesSection;