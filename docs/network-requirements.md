# Network Requirements for Remote GPU Workers

## Required Network Access

### From GPU Worker to Main Server
- **Redis**: Port 6379 (TCP) - Task queue communication
- **Qdrant**: Port 6333 (TCP) - Vector database access
- **HTTP API**: Port 8000 (TCP) - File transfer (if using HTTP mode)

### From Main Server to GPU Worker
- **Health Check**: Port 8001 (TCP) - Optional worker health monitoring

## Firewall Configuration

### Main Server Firewall Rules
```bash
# Allow Redis access from GPU workers
sudo ufw allow from GPU_WORKER_IP to any port 6379

# Allow Qdrant access from GPU workers  
sudo ufw allow from GPU_WORKER_IP to any port 6333

# Allow HTTP API access from GPU workers (if using HTTP file transfer)
sudo ufw allow from GPU_WORKER_IP to any port 8000
```

### GPU Worker Firewall Rules
```bash
# Allow health check from main server (optional)
sudo ufw allow from MAIN_SERVER_IP to any port 8001

# Allow outbound connections to main server
sudo ufw allow out to MAIN_SERVER_IP port 6379
sudo ufw allow out to MAIN_SERVER_IP port 6333
sudo ufw allow out to MAIN_SERVER_IP port 8000
```

## File Sharing Options

### Option A: Network File System (Recommended)
- **NFS**: Linux-to-Linux file sharing
- **SMB/CIFS**: Cross-platform file sharing
- **Cloud Storage**: S3, Azure Blob, Google Cloud Storage

### Option B: HTTP File Transfer
- Files transferred via HTTP API endpoints
- Automatic cleanup after processing
- Higher network overhead but simpler setup

## Security Considerations

### Network Security
- Use VPN or private networks for production deployments
- Restrict firewall rules to specific IP addresses
- Consider using SSL/TLS for HTTP file transfers
- Monitor network traffic for unusual activity

### File Access Security
- Validate file paths to prevent directory traversal
- Restrict file access to designated directories only
- Implement authentication for file transfer endpoints
- Use temporary files with automatic cleanup

## Performance Considerations

### Network Bandwidth
- Audio files: 1-50 MB typical
- Video files: 100 MB - 10 GB typical
- Consider network capacity when planning concurrent workers

### Latency Impact
- NFS: Low latency for file operations
- HTTP: Higher latency but more reliable over WAN
- Local storage preferred for temporary files

## Troubleshooting

### Common Network Issues
1. **Connection refused**: Check firewall rules and service status
2. **Permission denied**: Verify file permissions and user access
3. **Timeout errors**: Check network connectivity and bandwidth
4. **DNS resolution**: Use IP addresses if DNS is unreliable

### Diagnostic Commands
```bash
# Test Redis connectivity
redis-cli -h MAIN_SERVER_IP -p 6379 ping

# Test Qdrant connectivity
curl http://MAIN_SERVER_IP:6333/collections

# Test HTTP API connectivity
curl http://MAIN_SERVER_IP:8000/health

# Test network connectivity
ping MAIN_SERVER_IP
telnet MAIN_SERVER_IP 6379
```
