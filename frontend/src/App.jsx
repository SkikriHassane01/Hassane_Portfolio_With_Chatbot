import { useState } from 'react'
import { ThemeProvider } from './context/ThemeContext'
import Navbar from './components/layout/Navbar'
import Footer from './components/layout/Footer'
import About from './components/sections/About'
import TechSphere from './components/sections/TechSphere'
import Projects from './components/sections/Projects'
import Certificates from './components/sections/Certificates'
import Skills from './components/sections/Skills'
function App() {
  return (
    <ThemeProvider>
      <div className="min-h-screen bg-white dark:bg-gray-900">
        <Navbar />
        <main>
          <About />
          <TechSphere />
          <Projects />
          <Certificates />
          <Skills />
        </main>
        <Footer />
      </div>
    </ThemeProvider>
  )
}

export default App
