"""Simple webhook receiver for testing webhook functionality."""

from fastapi import FastAPI, Request
import uvicorn
import json
from datetime import datetime

app = FastAPI(title="Webhook Receiver", description="Simple webhook receiver for testing")

# Store received webhooks for inspection
received_webhooks = []

@app.post("/webhook")
async def receive_webhook(request: Request):
    """Receive and log webhook notifications."""
    try:
        payload = await request.json()
        
        # Add timestamp and log
        webhook_data = {
            "received_at": datetime.utcnow().isoformat(),
            "payload": payload,
            "headers": dict(request.headers)
        }
        
        received_webhooks.append(webhook_data)
        
        print(f"\n{'='*50}")
        print(f"WEBHOOK RECEIVED at {webhook_data['received_at']}")
        print(f"{'='*50}")
        print(f"Event Type: {payload.get('event_type', 'unknown')}")
        print(f"Timestamp: {payload.get('timestamp', 'unknown')}")
        
        if 'data' in payload:
            data = payload['data']
            print(f"Task ID: {data.get('task_id', 'unknown')}")
            print(f"Status: {data.get('status', 'unknown')}")
            
            if 'progress' in data:
                print(f"Progress: {data['progress']*100:.1f}%")
            
            if 'message' in data:
                print(f"Message: {data['message']}")
            
            if 'error' in data:
                print(f"Error: {data['error']}")
            
            if 'result' in data:
                print(f"Result: {json.dumps(data['result'], indent=2)}")
        
        print(f"{'='*50}\n")
        
        return {"status": "received", "webhook_id": len(received_webhooks)}
        
    except Exception as e:
        print(f"Error processing webhook: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/webhooks")
async def list_webhooks():
    """List all received webhooks."""
    return {
        "total_webhooks": len(received_webhooks),
        "webhooks": received_webhooks
    }

@app.get("/webhooks/latest")
async def get_latest_webhook():
    """Get the latest received webhook."""
    if received_webhooks:
        return received_webhooks[-1]
    return {"message": "No webhooks received yet"}

@app.delete("/webhooks")
async def clear_webhooks():
    """Clear all received webhooks."""
    global received_webhooks
    count = len(received_webhooks)
    received_webhooks = []
    return {"message": f"Cleared {count} webhooks"}

@app.get("/")
async def root():
    """Root endpoint with basic info."""
    return {
        "message": "Webhook Receiver is running",
        "endpoints": {
            "receive_webhook": "POST /webhook",
            "list_webhooks": "GET /webhooks",
            "latest_webhook": "GET /webhooks/latest",
            "clear_webhooks": "DELETE /webhooks"
        },
        "total_received": len(received_webhooks)
    }

if __name__ == "__main__":
    print("Starting Webhook Receiver...")
    print("Webhook endpoint: http://localhost:8001/webhook")
    print("Management interface: http://localhost:8001/webhooks")
    print("Press Ctrl+C to stop")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8001,
        log_level="info"
    )
