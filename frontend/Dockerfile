FROM node:20-alpine as build

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

#Copy the source code
COPY . .

# Build the app
RUN npm run build

# Serve stages -use nginx to serve static files
FROM nginx:alpine as production

# copy built files from build stages
COPY --from=build /app/dist /usr/share/nginx/html

# copy nginx configuration 
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
