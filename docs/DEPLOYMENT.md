# PrivaStream Deployment Guide

## Deployment Options

PrivaStream can be deployed in various configurations depending on your requirements:

- **Single Machine**: All services on one server
- **Microservices**: Distributed services across multiple servers
- **Cloud**: AWS, GCP, Azure deployments
- **Edge**: On-premise or edge computing deployments

## Docker Deployment

### Single Machine Deployment

#### Prerequisites

```bash
# Install Docker and Docker Compose
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

#### Quick Deploy

```bash
# Clone repository
git clone https://github.com/privastream/tiktok-techjam-2025.git
cd privastream

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Deploy with Docker Compose
docker-compose up -d
```

#### Services Overview

The deployment includes:

- **Frontend**: React app on port 3000
- **Backend**: Flask API on port 5000
- **Mediasoup**: WebRTC SFU on port 3001
- **Redis**: Caching and session storage (optional)
- **Nginx**: Reverse proxy and SSL termination

### Production Deployment

#### docker-compose.prod.yml

```yaml
version: "3.8"
services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - backend
      - frontend
    restart: unless-stopped

  backend:
    build: .
    environment:
      - FLASK_ENV=production
      - CUDA_VISIBLE_DEVICES=0
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  frontend:
    build:
      context: ./src/web/frontend
      dockerfile: Dockerfile
    restart: unless-stopped

  mediasoup:
    build:
      context: ./src/web/mediasoup
      dockerfile: Dockerfile
    environment:
      - NODE_ENV=production
    restart: unless-stopped

  redis:
    image: redis:alpine
    volumes:
      - redis-data:/data
    restart: unless-stopped

volumes:
  redis-data:
```

#### Deploy to Production

```bash
# Deploy with production config
docker-compose -f docker-compose.prod.yml up -d

# Monitor logs
docker-compose logs -f
```

## Kubernetes Deployment

### Prerequisites

```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl
sudo mv kubectl /usr/local/bin/

# Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
```

### Kubernetes Manifests

#### Namespace

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: privastream
```

#### Backend Deployment

```yaml
# k8s/backend-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: privastream-backend
  namespace: privastream
spec:
  replicas: 3
  selector:
    matchLabels:
      app: privastream-backend
  template:
    metadata:
      labels:
        app: privastream-backend
    spec:
      containers:
        - name: backend
          image: privastream/backend:latest
          ports:
            - containerPort: 5000
          env:
            - name: FLASK_ENV
              value: "production"
          resources:
            requests:
              nvidia.com/gpu: 1
              memory: "4Gi"
              cpu: "1"
            limits:
              nvidia.com/gpu: 1
              memory: "8Gi"
              cpu: "2"
```

#### Service

```yaml
# k8s/backend-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: privastream-backend-service
  namespace: privastream
spec:
  selector:
    app: privastream-backend
  ports:
    - port: 5000
      targetPort: 5000
  type: LoadBalancer
```

#### Deploy to Kubernetes

```bash
# Apply manifests
kubectl apply -f k8s/

# Check status
kubectl get pods -n privastream
kubectl get services -n privastream
```

## Cloud Deployments

### AWS Deployment

#### Using ECS with Fargate

```yaml
# aws/ecs-task-definition.json
{
  "family": "privastream-backend",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "containerDefinitions":
    [
      {
        "name": "privastream-backend",
        "image": "privastream/backend:latest",
        "portMappings": [{ "containerPort": 5000, "protocol": "tcp" }],
        "environment": [{ "name": "FLASK_ENV", "value": "production" }],
      },
    ],
}
```

#### Deploy with AWS CLI

```bash
# Create ECS cluster
aws ecs create-cluster --cluster-name privastream-cluster

# Register task definition
aws ecs register-task-definition --cli-input-json file://ecs-task-definition.json

# Create service
aws ecs create-service \
  --cluster privastream-cluster \
  --service-name privastream-backend \
  --task-definition privastream-backend:1 \
  --desired-count 2 \
  --launch-type FARGATE
```

### Google Cloud Platform

#### Using Cloud Run

```bash
# Build and push image
gcloud builds submit --tag gcr.io/PROJECT_ID/privastream-backend

# Deploy to Cloud Run
gcloud run deploy privastream-backend \
  --image gcr.io/PROJECT_ID/privastream-backend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Azure Container Instances

```bash
# Create resource group
az group create --name privastream-rg --location eastus

# Deploy container
az container create \
  --resource-group privastream-rg \
  --name privastream-backend \
  --image privastream/backend:latest \
  --cpu 2 \
  --memory 4 \
  --ports 5000
```

## Configuration

### Environment Variables

#### Production Environment

```bash
# .env.production
FLASK_ENV=production
FLASK_DEBUG=False

# Database
DATABASE_URL=postgresql://user:pass@localhost/privastream

# Redis
REDIS_URL=redis://localhost:6379

# GPU Configuration
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Security
SECRET_KEY=your-secret-key-here
JWT_SECRET_KEY=your-jwt-secret-here

# API Configuration
API_RATE_LIMIT=1000
MAX_UPLOAD_SIZE=1073741824  # 1GB

# Monitoring
SENTRY_DSN=your-sentry-dsn
LOG_LEVEL=INFO
```

### Nginx Configuration

```nginx
# nginx.conf
upstream backend {
    server backend:5000;
}

upstream frontend {
    server frontend:3000;
}

upstream mediasoup {
    server mediasoup:3001;
}

server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;

    # Frontend
    location / {
        proxy_pass http://frontend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # API
    location /api/ {
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # WebSocket
    location /socket.io/ {
        proxy_pass http://backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    # Mediasoup WebRTC
    location /mediasoup/ {
        proxy_pass http://mediasoup;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    # File uploads
    client_max_body_size 1G;
}
```

## SSL/TLS Configuration

### Let's Encrypt with Certbot

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
echo "0 12 * * * /usr/bin/certbot renew --quiet" | sudo crontab -
```

## Monitoring and Logging

### Prometheus Metrics

```yaml
# docker-compose.monitoring.yml
version: "3.8"
services:
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana

volumes:
  grafana-storage:
```

### Log Aggregation with ELK Stack

```yaml
# docker-compose.logging.yml
version: "3.8"
services:
  elasticsearch:
    image: elasticsearch:7.17.0
    environment:
      - discovery.type=single-node
    ports:
      - "9200:9200"

  logstash:
    image: logstash:7.17.0
    volumes:
      - ./logstash.conf:/usr/share/logstash/pipeline/logstash.conf

  kibana:
    image: kibana:7.17.0
    ports:
      - "5601:5601"
    depends_on:
      - elasticsearch
```

## Performance Optimization

### GPU Optimization

```bash
# GPU memory management
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256"
```

### Load Balancing

```nginx
# nginx load balancer
upstream backend_pool {
    least_conn;
    server backend1:5000 weight=3;
    server backend2:5000 weight=2;
    server backend3:5000 weight=1;
}
```

### Caching

```python
# Redis caching configuration
CACHE_TYPE = "RedisCache"
CACHE_REDIS_URL = "redis://redis:6379"
CACHE_DEFAULT_TIMEOUT = 300
```

## Security

### Firewall Configuration

```bash
# UFW rules
sudo ufw allow 22/tcp      # SSH
sudo ufw allow 80/tcp      # HTTP
sudo ufw allow 443/tcp     # HTTPS
sudo ufw enable
```

### Container Security

```dockerfile
# Security-focused Dockerfile
FROM python:3.10-slim

# Create non-root user
RUN useradd -m -u 1000 privastream

# Set security headers
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Switch to non-root user
USER privastream
```

## Backup and Recovery

### Database Backup

```bash
# PostgreSQL backup
pg_dump -h localhost -U user privastream > backup.sql

# Restore
psql -h localhost -U user -d privastream < backup.sql
```

### Model Backup

```bash
# Backup models
tar -czf models-backup.tar.gz models/

# Restore models
tar -xzf models-backup.tar.gz
```

## Health Checks

### Docker Health Check

```dockerfile
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5000/health || exit 1
```

### Kubernetes Health Check

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 5000
  initialDelaySeconds: 30
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /ready
    port: 5000
  initialDelaySeconds: 5
  periodSeconds: 5
```

## Troubleshooting

### Common Issues

#### Out of Memory

```bash
# Check memory usage
docker stats

# Reduce model batch size
export BATCH_SIZE=1
```

#### GPU Issues

```bash
# Check GPU availability
nvidia-smi

# Docker GPU support
docker run --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
```

#### Port Conflicts

```bash
# Check port usage
netstat -tulpn | grep :5000

# Change ports in docker-compose.yml
ports:
  - "5001:5000"
```

## Support

For deployment issues:

- **Documentation**: Check deployment guides
- **Issues**: [GitHub Issues](https://github.com/privastream/tiktok-techjam-2025/issues)
- **Enterprise Support**: Contact support@privastream.ai
