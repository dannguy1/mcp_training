# MCP Training Service Deployment Guide

This guide covers various deployment options for the MCP Training Service, from simple Docker deployments to production Kubernetes clusters.

## 🐳 Docker Deployment

### Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- 4GB+ RAM available
- 10GB+ disk space

### Quick Start

1. **Clone and navigate to the project**
   ```bash
   git clone <repository-url>
   cd mcp_training
   ```

2. **Start the service**
   ```bash
   # Production deployment
   docker-compose up -d
   
   # Development with hot reload
   docker-compose --profile dev up -d
   
   # With monitoring stack
   docker-compose --profile monitoring up -d
   ```

3. **Verify deployment**
   ```bash
   # Check service status
   docker-compose ps
   
   # Check logs
   docker-compose logs mcp-training
   
   # Test health endpoint
   curl http://localhost:8000/health
   ```

### Production Configuration

1. **Environment variables**
   ```bash
   # Create production environment file
   cp env.example .env.prod
   
   # Edit with production values
   nano .env.prod
   ```

2. **Production docker-compose**
   ```yaml
   # docker-compose.prod.yml
   version: '3.8'
   services:
     mcp-training:
       build: .
       environment:
         - LOG_LEVEL=WARNING
         - DEBUG=false
         - API_KEY=${API_KEY}
         - REQUIRE_AUTH=true
       volumes:
         - /data/models:/app/models
         - /data/exports:/app/exports
         - /data/logs:/app/logs
       restart: unless-stopped
       deploy:
         resources:
           limits:
             memory: 4G
             cpus: '2.0'
   ```

3. **Deploy with production config**
   ```bash
   docker-compose -f docker-compose.prod.yml up -d
   ```

## 🐙 Docker Swarm Deployment

### Initialize Swarm

```bash
# Initialize swarm (on manager node)
docker swarm init

# Join worker nodes
docker swarm join --token <token> <manager-ip>:2377
```

### Deploy Stack

```bash
# Deploy the stack
docker stack deploy -c docker-compose.yml mcp-training

# Check stack status
docker stack services mcp-training

# Scale service
docker service scale mcp-training_mcp-training=3
```

### Swarm Configuration

```yaml
# docker-compose.swarm.yml
version: '3.8'
services:
  mcp-training:
    image: mcp-training:latest
    deploy:
      replicas: 3
      update_config:
        parallelism: 1
        delay: 10s
        failure_action: rollback
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
        reservations:
          memory: 2G
          cpus: '1.0'
    volumes:
      - mcp_models:/app/models
      - mcp_exports:/app/exports
      - mcp_logs:/app/logs
    networks:
      - mcp_network

volumes:
  mcp_models:
    driver: local
  mcp_exports:
    driver: local
  mcp_logs:
    driver: local

networks:
  mcp_network:
    driver: overlay
```

## ☸️ Kubernetes Deployment

### Prerequisites

- Kubernetes cluster 1.20+
- kubectl configured
- Helm 3.0+ (optional)

### Manual Deployment

1. **Create namespace**
   ```bash
   kubectl create namespace mcp-training
   kubectl config set-context --current --namespace=mcp-training
   ```

2. **Create ConfigMap**
   ```yaml
   # configmap.yaml
   apiVersion: v1
   kind: ConfigMap
   metadata:
     name: mcp-training-config
   data:
     model_config.yaml: |
       model:
         type: isolation_forest
         contamination: 0.1
       features:
         numeric:
           - signal_strength
           - bitrate
     training_config.yaml: |
       service:
         name: mcp-training
       api:
         host: 0.0.0.0
         port: 8000
   ```

3. **Create Secret**
   ```yaml
   # secret.yaml
   apiVersion: v1
   kind: Secret
   metadata:
     name: mcp-training-secret
   type: Opaque
   data:
     api-key: <base64-encoded-api-key>
   ```

4. **Create Deployment**
   ```yaml
   # deployment.yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: mcp-training
   spec:
     replicas: 3
     selector:
       matchLabels:
         app: mcp-training
     template:
       metadata:
         labels:
           app: mcp-training
       spec:
         containers:
         - name: mcp-training
           image: mcp-training:latest
           ports:
           - containerPort: 8000
           env:
           - name: API_KEY
             valueFrom:
               secretKeyRef:
                 name: mcp-training-secret
                 key: api-key
           - name: LOG_LEVEL
             value: "INFO"
           volumeMounts:
           - name: config
             mountPath: /app/config
           - name: models
             mountPath: /app/models
           - name: exports
             mountPath: /app/exports
           - name: logs
             mountPath: /app/logs
           resources:
             requests:
               memory: "2Gi"
               cpu: "1"
             limits:
               memory: "4Gi"
               cpu: "2"
           livenessProbe:
             httpGet:
               path: /health
               port: 8000
             initialDelaySeconds: 30
             periodSeconds: 10
           readinessProbe:
             httpGet:
               path: /health
               port: 8000
             initialDelaySeconds: 5
             periodSeconds: 5
         volumes:
         - name: config
           configMap:
             name: mcp-training-config
         - name: models
           persistentVolumeClaim:
             claimName: mcp-models-pvc
         - name: exports
           persistentVolumeClaim:
             claimName: mcp-exports-pvc
         - name: logs
           persistentVolumeClaim:
             claimName: mcp-logs-pvc
   ```

5. **Create Service**
   ```yaml
   # service.yaml
   apiVersion: v1
   kind: Service
   metadata:
     name: mcp-training-service
   spec:
     selector:
       app: mcp-training
     ports:
     - protocol: TCP
       port: 80
       targetPort: 8000
     type: LoadBalancer
   ```

6. **Create Persistent Volumes**
   ```yaml
   # pvc.yaml
   apiVersion: v1
   kind: PersistentVolumeClaim
   metadata:
     name: mcp-models-pvc
   spec:
     accessModes:
       - ReadWriteMany
     resources:
       requests:
         storage: 10Gi
   ---
   apiVersion: v1
   kind: PersistentVolumeClaim
   metadata:
     name: mcp-exports-pvc
   spec:
     accessModes:
       - ReadWriteMany
     resources:
       requests:
         storage: 20Gi
   ---
   apiVersion: v1
   kind: PersistentVolumeClaim
   metadata:
     name: mcp-logs-pvc
   spec:
     accessModes:
       - ReadWriteMany
     resources:
       requests:
         storage: 5Gi
   ```

7. **Deploy to Kubernetes**
   ```bash
   kubectl apply -f configmap.yaml
   kubectl apply -f secret.yaml
   kubectl apply -f pvc.yaml
   kubectl apply -f deployment.yaml
   kubectl apply -f service.yaml
   ```

### Helm Deployment

1. **Create Helm chart**
   ```bash
   helm create mcp-training
   cd mcp-training
   ```

2. **Customize values.yaml**
   ```yaml
   # values.yaml
   replicaCount: 3
   
   image:
     repository: mcp-training
     tag: latest
     pullPolicy: IfNotPresent
   
   service:
     type: LoadBalancer
     port: 80
   
   resources:
     limits:
       cpu: 2
       memory: 4Gi
     requests:
       cpu: 1
       memory: 2Gi
   
   persistence:
     models:
       enabled: true
       size: 10Gi
     exports:
       enabled: true
       size: 20Gi
     logs:
       enabled: true
       size: 5Gi
   
   config:
     apiKey: ""
     logLevel: INFO
   ```

3. **Deploy with Helm**
   ```bash
   helm install mcp-training . -n mcp-training
   ```

## 🔧 Configuration Management

### Environment-Specific Configs

1. **Development**
   ```bash
   # .env.dev
   DEBUG=true
   LOG_LEVEL=DEBUG
   API_HOST=0.0.0.0
   API_PORT=8000
   REQUIRE_AUTH=false
   ```

2. **Staging**
   ```bash
   # .env.staging
   DEBUG=false
   LOG_LEVEL=INFO
   API_HOST=0.0.0.0
   API_PORT=8000
   REQUIRE_AUTH=true
   API_KEY=staging-key
   ```

3. **Production**
   ```bash
   # .env.prod
   DEBUG=false
   LOG_LEVEL=WARNING
   API_HOST=0.0.0.0
   API_PORT=8000
   REQUIRE_AUTH=true
   API_KEY=production-key
   ```

### Configuration Validation

```bash
# Validate configuration
python -c "from mcp_training.core.config import get_config; print(get_config())"

# Test configuration loading
python -m mcp_training.cli validate-config
```

## 📊 Monitoring Setup

### Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'mcp-training'
    static_configs:
      - targets: ['mcp-training:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s
```

### Grafana Dashboards

1. **Training Dashboard**
   - Training job status
   - Model performance metrics
   - Feature extraction statistics

2. **System Dashboard**
   - CPU and memory usage
   - Disk I/O
   - Network traffic

3. **API Dashboard**
   - Request rates
   - Response times
   - Error rates

### Alerting Rules

```yaml
# monitoring/alerts/training_alerts.yml
groups:
  - name: mcp-training
    rules:
      - alert: TrainingJobFailed
        expr: mcp_training_job_status{status="failed"} > 0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Training job failed"
          
      - alert: HighMemoryUsage
        expr: container_memory_usage_bytes{container="mcp-training"} > 3e9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage"
```

## 🔒 Security Configuration

### Network Security

1. **Firewall Rules**
   ```bash
   # Allow only necessary ports
   ufw allow 8000/tcp  # API
   ufw allow 22/tcp    # SSH
   ufw deny 8000/tcp from 0.0.0.0/0  # Restrict API access
   ```

2. **API Authentication**
   ```bash
   # Generate secure API key
   openssl rand -hex 32
   
   # Set in environment
   export API_KEY="generated-key"
   export REQUIRE_AUTH=true
   ```

3. **TLS/SSL Configuration**
   ```nginx
   # nginx.conf
   server {
       listen 443 ssl;
       server_name your-domain.com;
       
       ssl_certificate /path/to/cert.pem;
       ssl_certificate_key /path/to/key.pem;
       
       location / {
           proxy_pass http://localhost:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

### Data Security

1. **Encrypted Storage**
   ```bash
   # Encrypt volumes
   cryptsetup luksFormat /dev/sdb
   cryptsetup luksOpen /dev/sdb mcp_data
   mkfs.ext4 /dev/mapper/mcp_data
   ```

2. **Backup Strategy**
   ```bash
   # Backup script
   #!/bin/bash
   DATE=$(date +%Y%m%d_%H%M%S)
   tar -czf backup_$DATE.tar.gz models/ exports/
   gpg --encrypt --recipient your-email backup_$DATE.tar.gz
   ```

## 🚀 Deployment Checklist

### Pre-Deployment

- [ ] Environment variables configured
- [ ] Configuration files validated
- [ ] Dependencies installed
- [ ] Storage volumes created
- [ ] Network access configured
- [ ] Monitoring setup ready

### Deployment

- [ ] Service deployed successfully
- [ ] Health checks passing
- [ ] API endpoints accessible
- [ ] Logs being generated
- [ ] Metrics being collected
- [ ] Alerts configured

### Post-Deployment

- [ ] Load testing completed
- [ ] Performance benchmarks met
- [ ] Security scan passed
- [ ] Documentation updated
- [ ] Team notified
- [ ] Monitoring verified

## 🔄 Rolling Updates

### Docker Swarm

```bash
# Update service
docker service update --image mcp-training:new-version mcp-training

# Rollback if needed
docker service rollback mcp-training
```

### Kubernetes

```bash
# Update deployment
kubectl set image deployment/mcp-training mcp-training=mcp-training:new-version

# Rollback
kubectl rollout undo deployment/mcp-training
```

## 🆘 Troubleshooting

### Common Issues

1. **Service won't start**
   ```bash
   # Check logs
   docker-compose logs mcp-training
   
   # Check resource usage
   docker stats
   ```

2. **API not accessible**
   ```bash
   # Check port binding
   netstat -tlnp | grep 8000
   
   # Test connectivity
   curl -v http://localhost:8000/health
   ```

3. **Storage issues**
   ```bash
   # Check volume mounts
   docker volume ls
   
   # Check permissions
   ls -la models/ exports/ logs/
   ```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
export DEBUG=true

# Restart service
docker-compose restart mcp-training
```

## 📈 Scaling

### Horizontal Scaling

```bash
# Docker Swarm
docker service scale mcp-training=5

# Kubernetes
kubectl scale deployment mcp-training --replicas=5
```

### Vertical Scaling

```yaml
# Increase resources
resources:
  limits:
    memory: 8Gi
    cpu: '4.0'
  requests:
    memory: 4Gi
    cpu: '2.0'
```

This deployment guide provides comprehensive instructions for deploying the MCP Training Service in various environments, from simple Docker setups to production Kubernetes clusters. 