# Deployment Guide

This guide provides step-by-step instructions for deploying the portfolio website with AI chatbot using Docker containers, Azure App Service, and GitHub Pages.


## Backend Deployment to Azure App Service

### Step 1: Containerize the Backend

1. Create/update the `Dockerfile` in the backend directory:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

....
```

2. Build and test the Docker container locally:

```bash
cd backend
docker build -t portfolio-backend:latest .
docker run -p 8000:8000 -e AZURE_OPENAI_KEY=your-key portfolio-backend:latest
```

3. Verify it works by accessing `http://localhost:8000`

### Step 2: Create Azure Container Registry (ACR)

```bash
# Login to Azure
az login

# Create resource group if needed
az group create --name portfolio-resource-group --location eastus

# Create Azure Container Registry
az acr create --resource-group portfolio-resource-group --name portfolioacr --sku Basic --admin-enabled true

# Get ACR credentials
az acr credential show --name portfolioacr
```

Make note of the username and one of the passwords shown.

### Step 3: Push the Container to ACR

```bash
# Login to your ACR
docker login portfolioacr.azurecr.io --username portfolioacr --password <password>

# Tag your local image
docker tag portfolio-backend:latest portfolioacr.azurecr.io/portfolio-backend:latest

# Push the image to ACR
docker push portfolioacr.azurecr.io/portfolio-backend:latest
```

### Step 4: Create and Configure the Azure App Service

```bash
# Create App Service Plan (Linux)
az appservice plan create --resource-group portfolio-resource-group --name portfolio-service-plan --is-linux --sku B1

# Create Web App for Containers
az webapp create --resource-group portfolio-resource-group \
  --plan portfolio-service-plan \
  --name your-backend-app-name \
  --deployment-container-image-name portfolioacr.azurecr.io/portfolio-backend:latest

# Configure the container registry settings
az webapp config container set \
  --resource-group portfolio-resource-group \
  --name your-backend-app-name \
  --docker-custom-image-name portfolioacr.azurecr.io/portfolio-backend:latest \
  --docker-registry-server-url https://portfolioacr.azurecr.io \
  --docker-registry-server-user portfolioacr \
  --docker-registry-server-password <password>
```

### Step 5: Configure Environment Variables

```bash
# Set environment variables for the app service
az webapp config appsettings set --resource-group portfolio-resource-group --name your-backend-app-name --settings \
  AZURE_OPENAI_KEY=your-openai-key \
  AZURE_OPENAI_ENDPOINT=your-openai-endpoint \
  AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name \
  FLASK_ENV=production \
  WEBSITES_PORT=8000
```

### Step 6: Enable CORS for GitHub Pages

```bash
# Set CORS for GitHub Pages domain
az webapp cors add --resource-group portfolio-resource-group --name your-backend-app-name \
  --allowed-origins "https://your-github-username.github.io"
```

Also update your Flask app to handle CORS:

```python
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": [
    "http://localhost:3000",
    "https://your-github-username.github.io"
]}})
```

### Step 7: Restart the App Service

```bash
# Restart the app service to apply changes
az webapp restart --resource-group portfolio-resource-group --name your-backend-app-name
```

Your backend API should now be accessible at: `https://your-backend-app-name.azurewebsites.net/api`

## Frontend Deployment to GitHub Pages

### Step 1: Update API URL in Frontend

Update your `.env` file in the frontend directory:

```
VITE_API_URL=https://your-backend-app-name.azurewebsites.net/api
```

### Step 2: Configure GitHub Repository

1. Ensure your frontend builds correctly with the API URL:

```bash
cd frontend
npm install
npm run build
```

2. Create a `.github/workflows` directory if it doesn't exist:

```bash
mkdir -p .github/workflows
```

### Step 3: Create GitHub Actions Workflow File

Create `.github/workflows/deploy-frontend.yml`:

```yaml
name: Deploy Frontend to GitHub Pages

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
          cache: 'npm'
          cache-dependency-path: frontend/package-lock.json

      - name: Install and Build
        working-directory: ./frontend
        run: |
          npm ci
          npm run build
        env:
          VITE_API_URL: ${{ secrets.VITE_API_URL }}

      - name: Deploy to GitHub Pages
        uses: JamesIves/github-pages-deploy-action@v4
        with:
          folder: frontend/dist
          branch: gh-pages
```

### Step 4: Configure GitHub Repository Secrets

1. Go to your GitHub repository settings
2. Navigate to "Secrets and variables" > "Actions"
3. Add a new repository secret:
   - Name: `VITE_API_URL`
   - Value: `https://your-backend-app-name.azurewebsites.net/api`

### Step 5: Enable GitHub Pages

1. Go to your GitHub repository settings
2. Navigate to "Pages"
3. Under "Source", select "Deploy from a branch"
4. Select the "gh-pages" branch and "/ (root)" folder
5. Click "Save"

### Step 6: Push Changes and Trigger Deployment

```bash
# Commit your changes
git add .
git commit -m "Setup deployment workflow"

# Push to GitHub
git push origin main
```

This will trigger the GitHub Action to build and deploy your frontend.

### Step 7: Configure Custom Domain (Optional)

1. In your GitHub repository, go to "Settings" > "Pages"
2. Under "Custom domain", enter your domain name
3. Click "Save"
4. Update your DNS settings with your domain provider:
   - Add an A record pointing to GitHub Pages IPs
   - Or add a CNAME record pointing to `your-github-username.github.io`

## Verifying the Deployment

1. Frontend: Visit `https://your-github-username.github.io/your-repo-name` or your custom domain
2. Backend API: Test with a tool like Postman at `https://your-backend-app-name.azurewebsites.net/api/health`
3. Test the chatbot on your deployed frontend to ensure it connects to the backend