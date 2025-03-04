import { useState } from 'react'
import { ThemeProvider } from './context/ThemeContext'
import Navbar from './components/layout/Navbar'
import Footer from './components/layout/Footer'
import About from './components/sections/About'
function App() {
  return (
    <ThemeProvider>
      <div className="min-h-screen bg-white dark:bg-gray-900">
        <Navbar />
        <main>
          <About />
        </main>
        <Footer />
      </div>
    </ThemeProvider>
  )
}

export default App
