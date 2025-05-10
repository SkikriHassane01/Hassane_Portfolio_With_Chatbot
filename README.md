# [Portfolio Website with AI Chatbot](https://skikrihassane01.github.io/Hassane_Portfolio_With_Chatbot/)

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![React](https://img.shields.io/badge/React-18.2.0-61DAFB?logo=react)
![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-3.3.2-38B2AC?logo=tailwind-css)
![Flask](https://img.shields.io/badge/Flask-2.3.3-000000?logo=flask)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-FF6F00?logo=tensorflow)

A modern, responsive portfolio website with an integrated AI-powered chatbot built using React, Tailwind CSS, and Flask. The chatbot utilizes a hybrid approach combining TensorFlow for intent classification and Azure OpenAI for advanced responses.

![Portfolio Website Preview](https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExbWIyYzI4aXZkNWJmbGJtZWxkc3Rxb2hoNWVoNHBpdWFqenk3eXBmbyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/vokeEQ13s71BdYpySM/giphy.gif)

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
cd frontend

npm install
```

### Backend Setup

```bash
cd backend

python -m venv venv

venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

## 🌐 Deployment

For detailed instructions on deployment, see the [deployment guide](docs/deployment.md).

## 📁 Folder Structure

```
portfolio-website/
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── components/        # React components
│   │   │   ├── chatbot/       # Chatbot components
│   │   │   ├── common/        # Reusable components
│   │   │   ├── layout/        # Layout components
│   │   │   └── sections/      # Portfolio section components
│   │   ├── context/
│   │   ├── hooks/
│   │   ├── styles/
│   │   ├── utils/
│   │   ├── App.jsx
│   │   └── main.jsx 
│   ├── package.json
│   ├── tailwind.config.js
│   └── vite.config.js
│
├── backend/
│   ├── data/ 
│   │   └── intents.json 
│   ├── models/ 
│   ├── src/
│   │   ├── prediction/   
│   │   ├── routes/        
│   │   ├── services/    
│   │   └── utils/          
│   ├── app.py              
│   ├── requirements.txt   
│
├── .github/workflows/         # GitHub Actions workflows
│   ├── deploy-frontend.yml    # Frontend deployment workflow
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
