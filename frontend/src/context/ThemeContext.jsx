import React, { createContext, useContext, useState, useEffect } from 'react';

const ThemeContext = createContext();

export function ThemeProvider({ children }) {
  const [isDark, setIsDark] = useState(() => {
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem('theme');
      
      // Handle different possible stored values
      if (saved) {
        // Handle string values like "dark" or "light"
        if (saved === "dark") {
          return true;
        } else if (saved === "light") {
          return false;
        }
        
        // Try to parse as JSON for boolean values
        try {
          return JSON.parse(saved);
        } catch (e) {
          // If parsing fails, use system preference
          console.warn('Invalid theme value in localStorage');
        }
      }
      
      // Default to system preference if no valid saved value
      return window.matchMedia('(prefers-color-scheme: dark)').matches;
    }
    return false;
  });

  useEffect(() => {
    if (typeof window !== 'undefined') {
      // Store as a simple string instead of JSON
      localStorage.setItem('theme', isDark ? "dark" : "light");
      document.documentElement.classList.toggle('dark', isDark);
    }
  }, [isDark]);

  return (
    <ThemeContext.Provider value={{ isDark, setIsDark }}>
      {children}
    </ThemeContext.Provider>
  );
}

export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
};