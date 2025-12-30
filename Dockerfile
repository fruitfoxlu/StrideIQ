# Simple static site container
FROM nginx:alpine

# Copy static files
COPY index.html /usr/share/nginx/html/index.html
COPY style.css /usr/share/nginx/html/style.css
COPY app.js /usr/share/nginx/html/app.js
COPY README.md /usr/share/nginx/html/README.md

# Optional: basic security headers (minimal)
RUN printf '%s
' 'server {' '  listen 80;' '  server_name _;' '  root /usr/share/nginx/html;' '  index index.html;' '  add_header X-Content-Type-Options nosniff always;' '  add_header X-Frame-Options DENY always;' '  add_header Referrer-Policy no-referrer always;' '  add_header Permissions-Policy "camera=(), microphone=(), geolocation=()" always;' '  location / { try_files $uri $uri/ =404; }' '}' > /etc/nginx/conf.d/default.conf

EXPOSE 80
