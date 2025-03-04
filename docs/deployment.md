# Deployment Process 

This guide provides straightforward instructions for deploying your portfolio website using Docker containers to GitHub Pages (frontend) and Azure (backend).

## Overview

1. Build Docker images for both frontend and backend
2. Deploy frontend to GitHub Pages using GitHub Actions
3. Deploy backend to Azure App Service (Docker container)
4. Connect the frontend and backend

## Prerequisites

- Docker installed on your local machine
- GitHub account with GitHub Actions enabled
- Azure account

## Frontend Deployment (GitHub Pages with Docker)

### 1. Create Frontend Dockerfile

Create `frontend/Dockerfile`:

```dockerfile
# Build stage
FROM node:18-alpine as build

WORKDIR /app

# Copy package files and install dependencies
COPY package.json package-lock.json ./
RUN npm ci

# Copy source code
COPY . .

# Build the app
RUN npm run build

# Serve stage - use nginx to serve static files
FROM nginx:alpine

# Copy built files from build stage
COPY --from=build /app/dist /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
```

### 2. Create Nginx Configuration

Create `frontend/nginx.conf`:

```nginx
server {
    listen 80;
    
    location / {
        root /usr/share/nginx/html;
        index index.html index.htm;
        try_files $uri $uri/ /index.html;
    }
}
```

### 3. Configure API URL

Create `.env.production` in your frontend directory:

```
VITE_API_URL=https://your-backend-name.azurewebsites.net
```

### 4. Set Up GitHub Actions for Frontend

Create `.github/workflows/deploy-frontend.yml`:

```yaml
name: Deploy Frontend

on:
  push:
    branches: [ main ]
    paths:
      - 'frontend/**'

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v3

      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'

      - name: Build Frontend
        run: |
          cd frontend
          npm ci
          npm run build
        env:
          VITE_API_URL: ${{ secrets.AZURE_BACKEND_URL }}

      - name: Deploy to GitHub Pages
        uses: JamesIves/github-pages-deploy-action@v4
        with:
          folder: frontend/dist
          branch: gh-pages
```

## Backend Deployment (Azure with Docker)

### 1. Create Backend Dockerfile

Create `backend/Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4')"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production

# Expose port
EXPOSE 5000

# Run with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "wsgi:app"]
```

### 2. Update CORS Settings

In your backend Flask app, update CORS settings to allow GitHub Pages:

```python
from flask_cors import CORS

# Configure CORS
CORS(app, resources={r"/api/*": {"origins": [
    "http://localhost:3000",  # Development
    "https://yourusername.github.io"  # Production
]}})
```

### 3. Deploy Backend to Azure (Portal Method)

1. **Login to Azure Portal**
   - Go to [portal.azure.com](https://portal.azure.com)

2. **Create App Service**
   - Click "Create a resource"
   - Search for "Web App for Containers"
   - Click "Create"

3. **Configure App Service**
   - **Basics**:
     - Resource Group: Create new
     - Name: your-backend-name
     - Publish: Docker Container
     - Operating System: Linux
     - Region: Select nearest region
   - **Docker**:
     - Options: Single Container
     - Source: Quickstart
     - Image and tag: Will update later
   - Click "Review + create", then "Create"

4. **Configure Container Settings**
   - Once created, go to your App Service
   - Navigate to "Settings > Configuration"
   - Add these Application settings:
     - `AZURE_OPENAI_BASE_URL`: Your Azure OpenAI URL
     - `AZURE_OPENAI_API_KEY`: Your Azure OpenAI key
     - `FLASK_ENV`: production
     - `SECRET_KEY`: Random string
     - `CONFIDENCE_THRESHOLD`: 0.7
   - Click "Save"

5. **Deploy Container (CLI Method)**
   - Build and push your Docker image:

```bash
# Login to Azure
az login

# Create Azure Container Registry (ACR)
az acr create --name yourRegistry --resource-group your-resource-group --sku Basic --admin-enabled true

# Login to ACR
az acr login --name yourRegistry

# Build and tag your image
docker build -t yourRegistry.azurecr.io/portfolio-backend:latest ./backend

# Push image to ACR
docker push yourRegistry.azurecr.io/portfolio-backend:latest

# Configure App Service to use your container
az webapp config container set --name your-backend-name --resource-group your-resource-group --docker-custom-image-name yourRegistry.azurecr.io/portfolio-backend:latest --docker-registry-server-url https://yourRegistry.azurecr.io --docker-registry-server-user yourRegistry --docker-registry-server-password $(az acr credential show --name yourRegistry --query passwords[0].value -o tsv)
```

### 4. GitHub Actions for Backend (Alternative)

Create `.github/workflows/deploy-backend.yml`:

```yaml
name: Deploy Backend to Azure

on:
  push:
    branches: [ main ]
    paths:
      - 'backend/**'

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: 'Login to Azure'
        uses: azure/login@v1
        with:
          creds: ${{ secrets.AZURE_CREDENTIALS }}
          
      - name: 'Build and Push Docker Image'
        uses: azure/docker-login@v1
        with:
          login-server: ${{ secrets.REGISTRY_LOGIN_SERVER }}
          username: ${{ secrets.REGISTRY_USERNAME }}
          password: ${{ secrets.REGISTRY_PASSWORD }}
      
      - run: |
          docker build -t ${{ secrets.REGISTRY_LOGIN_SERVER }}/portfolio-backend:${{ github.sha }} ./backend
          docker push ${{ secrets.REGISTRY_LOGIN_SERVER }}/portfolio-backend:${{ github.sha }}
      
      - name: 'Deploy to Azure App Service'
        uses: azure/webapps-deploy@v2
        with:
          app-name: 'your-backend-name'
          images: '${{ secrets.REGISTRY_LOGIN_SERVER }}/portfolio-backend:${{ github.sha }}'
```

## Connecting Frontend and Backend

### 1. Update API References in Frontend

In `useChat.js` or wherever you make API calls:

```javascript
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

// When making API calls
fetch(`${API_URL}/api/chat`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ message: messageText })
})
```

### 2. Testing the Connection

1. Deploy both frontend and backend
2. Access your frontend at `https://yourusername.github.io/portfolio-website`
3. Test the chatbot functionality to ensure it connects to the backend