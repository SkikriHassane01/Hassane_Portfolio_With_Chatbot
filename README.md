# Portfolio Website with AI Chatbot

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![License](https://img.shields.io/badge/license-Boost_Software_License-green)
![React](https://img.shields.io/badge/React-18.2.0-61DAFB?logo=react)
![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-3.3.2-38B2AC?logo=tailwind-css)
![Flask](https://img.shields.io/badge/Flask-2.3.3-000000?logo=flask)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-FF6F00?logo=tensorflow)

A modern, responsive portfolio website with an integrated AI-powered chatbot built using React, Tailwind CSS, and Flask. The chatbot utilizes a hybrid approach combining TensorFlow for intent classification and Azure OpenAI for advanced responses.

![Portfolio Website Preview](https://via.placeholder.com/800x450.png?text=Portfolio+Website+Preview)

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Technology Stack](#-technology-stack)
- [Installation](#-installation)
- [Development](#-development)
- [Deployment](#-deployment)
- [Folder Structure](#-folder-structure)
- [License](#-license)
- [Contact](#-contact)

## ✨ Features

### Portfolio Website
- 🌓 Light/Dark theme toggle with system preference detection
- 📱 Fully responsive design that works on all devices
- 🔄 Interactive 3D technology sphere built with Three.js
- 🗃️ Project showcase with filtering and search functionality
- 📊 Visualized skills and expertise
- 🏆 Certificate showcase with modal details
- 📬 Contact form with validation
- 🎬 Smooth animations using Framer Motion

### AI Chatbot
- 🧠 Hybrid response system using TensorFlow and Azure OpenAI
- 🔍 Intent classification for faster responses to common questions
- 💬 Contextual conversation capabilities
- 🎯 Quick reply suggestions based on conversation context
- 📱 Mobile-friendly interface with minimize/maximize options
- 🔄 Conversation history persistence within session

## 🏗 Architecture

```
┌─────────────────────────────────────┐      ┌─────────────────────────────────────┐
│                                     │      │                                     │
│           React Frontend            │      │            Flask Backend            │
│                                     │      │                                     │
│  ┌─────────────┐    ┌────────────┐  │      │  ┌─────────────┐    ┌────────────┐  │
│  │  Portfolio  │    │  Chatbot   │  │      │  │  Intent     │    │  Azure     │  │
│  │  Sections   │    │  Interface │  │      │  │  Classifier │    │  OpenAI    │  │
│  └─────────────┘    └────────────┘  │      │  └─────────────┘    └────────────┘  │
│          │               │          │      │          │               │          │
│          └───────┬───────┘          │      │          └───────┬───────┘          │
│                  │                  │      │                  │                  │
│          ┌───────┴───────┐          │      │          ┌───────┴───────┐          │
│          │  API Client   │          │      │          │  Response     │          │
│          └───────────────┘          │      │          │  Manager      │          │
│                  │                  │      │          └───────────────┘          │
└──────────────────┼──────────────────┘      └──────────────────┼──────────────────┘
                   │                                            │                   
                   └────────────────────┬─────────────────────┘                    
                                        │                                          
                                 ┌──────┴───────┐                                  
                                 │   RESTful    │                                  
                                 │     API      │                                  
                                 └──────────────┘                                  
```

## Front End Architecture

### Portfolio Sections

Individual sections like About, Projects, Skills, etc.

### Chatbot Interface

Built with several components:

- `ChatWindow.jsx`: Main container managing state and UI
- `ChatMessage.jsx`: Renders individual messages
- `ChatInput.jsx`: Handles user input
- `QuickReplyButtons.jsx`: Displays contextual suggestions

### API Client

`useChat.js:` Custom React hook that:

- Manages local chat state
- Handles API calls to the backend
- Processes responses and updates UI

## Backend architecture

### Intent Classifier

- Uses TensorFlow to classify user messages into intents
- Built with a neural network trained on `intents.json` data
- Implemented in `intent_classifier.py`
- Returns intent predictions with confidence scores

### Azure OpenAI Integration

- Connects to Azure OpenAI API for advanced responses
- Implemented in `azure_service.py`
- Handles API authentication and request formatting
- Processes responses from the language model

### API Endpoints

- REST API implemented with Flask
- Main endpoint: /api/chat for message processing
- Additional endpoints for health checks and testing
- Configured with CORS for cross-origin requests

## Response Manager

- Central orchestration layer `response_manager.py`
- Decides whether to use local responses or Azure OpenAI

## 🔧 Technology Stack

### Frontend
- **React.js** - Frontend framework
- **Vite** - Build tool and development server
- **Tailwind CSS** - Utility-first CSS framework
- **Framer Motion** - Animation library
- **Three.js** - 3D graphics library (for TechSphere)
- **React Three Fiber** - React renderer for Three.js
- **Lucide React** - Icon library
- **EmailJS** - Client-side email sending

### Backend
- **Flask** - Python web framework
- **TensorFlow** - Machine learning framework
- **NLTK** - Natural Language Toolkit
- **Azure OpenAI API** - Advanced language model integration
- **Gunicorn** - WSGI HTTP Server for production

## 🚀 Installation

### Clone the Repository

```bash
git clone https://github.com/SkikriHassane01/Hassane_Portfolio_With_Chatbot.git
cd portfolio-website
```

### Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install
```

### Backend Setup

```bash
# Navigate to backend directory from the project root
cd backend

# Create and activate virtual environment
python -m venv venv

venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create a .env file for environment variables
cp .env.example .env

# Update the .env file with your Azure OpenAI credentials
```

## 💻 Development

### Running Frontend Development Server

```bash
# From the frontend directory
npm run dev

# This will start the development server on http://localhost:3000
```

### Running Backend Development Server

```bash
# From the backend directory with virtual environment activated
python app.py

# This will start the Flask server on http://localhost:5000
```

### Building for Production

```bash
# Frontend build
cd frontend
npm run build

# The build artifacts will be stored in the frontend/dist/ directory
```

## 🌐 Deployment

### Docker Deployment

We use Docker for containerization, making deployment consistent across environments.

```bash
# From the project root
docker-compose up --build

# This will build and start both frontend and backend containers
```

### Digital Ocean Deployment

For detailed instructions on deploying to Digital Ocean App Platform, see the [deployment guide](docs/deployment.md).

### Domain Configuration

For instructions on configuring a custom domain with Namecheap, see the [domain setup guide](docs/domain-setup.md).


## 📁 Folder Structure

```
portfolio-website/
├── frontend/                  # React frontend application
│   ├── public/                # Static files
│   ├── src/
│   │   ├── components/        # React components
│   │   │   ├── chatbot/       # Chatbot components
│   │   │   ├── common/        # Reusable components
│   │   │   ├── layout/        # Layout components
│   │   │   └── sections/      # Portfolio section components
│   │   ├── context/           # React context providers
│   │   ├── hooks/             # Custom React hooks
│   │   ├── styles/            # CSS files
│   │   ├── utils/             # Utility functions
│   │   ├── App.jsx            # Main application component
│   │   └── main.jsx           # Application entry point
│   ├── index.html             # HTML template
│   ├── package.json           # Frontend dependencies
│   ├── tailwind.config.js     # Tailwind CSS configuration
│   └── vite.config.js         # Vite configuration
│
├── backend/                   # Flask backend application
│   ├── data/                  # Data files
│   │   └── intents.json       # Chatbot training data
│   ├── models/                # Trained ML models
│   ├── src/
│   │   ├── prediction/        # Intent classification
│   │   ├── routes/            # API endpoints
│   │   ├── services/          # Business logic
│   │   └── utils/             # Utility functions
│   ├── app.py                 # Flask application
│   ├── requirements.txt       # Python dependencies
│   └── wsgi.py                # WSGI entry point
│
├── docker-compose.yml         # Docker Compose configuration
├── .gitignore                 # Git ignore file
├── LICENSE                    # License file
└── README.md                  # Project documentation
```

## 📜 License

This project is licensed under the MIT license - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

Email: `hassaneskikri@gmail.com`

Project Link: [Here](https://github.com/SkikriHassane01/Hassane_Portfolio_With_Chatbot)

---

Built with ❤️ by Hassane Skikir