"""
WebSocket routes for real-time updates in MCP Training Service.
"""

import json
import asyncio
from typing import List, Dict, Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from fastapi.responses import JSONResponse

from ...utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


class ConnectionManager:
    """Manages WebSocket connections and message broadcasting."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_types: Dict[WebSocket, str] = {}  # Track connection types (general, logs)
    
    async def connect(self, websocket: WebSocket, connection_type: str = "general"):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_types[websocket] = connection_type
        logger.info(f"WebSocket connected: {connection_type} (total: {len(self.active_connections)})")
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            connection_type = self.connection_types.pop(websocket, "unknown")
            logger.info(f"WebSocket disconnected: {connection_type} (total: {len(self.active_connections)})")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send a message to a specific WebSocket connection."""
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Failed to send personal message: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: str, connection_type: str = "general"):
        """Broadcast a message to all connections of a specific type."""
        disconnected = []
        
        for connection in self.active_connections:
            if self.connection_types.get(connection) == connection_type:
                try:
                    await connection.send_text(message)
                except Exception as e:
                    logger.error(f"Failed to broadcast message: {e}")
                    disconnected.append(connection)
        
        # Remove dead connections
        for connection in disconnected:
            self.disconnect(connection)
    
    async def broadcast_to_all(self, message: str):
        """Broadcast a message to all active connections."""
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Failed to broadcast to all: {e}")
                disconnected.append(connection)
        
        # Remove dead connections
        for connection in disconnected:
            self.disconnect(connection)


# Global connection manager instance
manager = ConnectionManager()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """General WebSocket endpoint for real-time updates."""
    await manager.connect(websocket, "general")
    try:
        # Send initial system status
        try:
            from .health import health_status
            from fastapi import Request
            from starlette.datastructures import State
            
            # Create a minimal mock request
            mock_request = Request(scope={
                'type': 'http',
                'method': 'GET',
                'path': '/api/health/status',
                'headers': [],
                'state': State()
            })
            
            status = await health_status(mock_request)
            await broadcast_system_status(status)
        except Exception as e:
            logger.warning(f"Failed to send initial system status: {e}")
        
        while True:
            # Keep connection alive and handle any incoming messages
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                logger.debug(f"Received WebSocket message: {message}")
                # Handle any client messages if needed
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON received: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


@router.websocket("/ws/logs")
async def logs_websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint specifically for log streaming."""
    await manager.connect(websocket, "logs")
    try:
        while True:
            # Keep connection alive and handle any incoming messages
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                logger.debug(f"Received logs WebSocket message: {message}")
                # Handle any client messages if needed
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON received in logs WebSocket: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"Logs WebSocket error: {e}")
        manager.disconnect(websocket)


# Utility functions for broadcasting messages
async def broadcast_training_update(job_id: str, progress: float, status: str, **kwargs):
    """Broadcast training progress update to all connected clients."""
    message = {
        "type": "training_update",
        "data": {
            "job_id": job_id,
            "progress": progress,
            "status": status,
            **kwargs
        }
    }
    await manager.broadcast(json.dumps(message), "general")


async def broadcast_model_ready(model_id: str, **kwargs):
    """Broadcast model ready notification to all connected clients."""
    message = {
        "type": "model_ready",
        "data": {
            "model_id": model_id,
            **kwargs
        }
    }
    await manager.broadcast(json.dumps(message), "general")


async def broadcast_system_status(status_data: Dict[str, Any]):
    """Broadcast system status update to all connected clients."""
    message = {
        "type": "system_status",
        "data": status_data
    }
    await manager.broadcast(json.dumps(message), "general")


async def broadcast_log_entry(log_data: Dict[str, Any]):
    """Broadcast log entry to logs WebSocket clients."""
    message = {
        "type": "log",
        "data": log_data
    }
    await manager.broadcast(json.dumps(message), "logs")


async def broadcast_error(error_message: str, error_type: str = "system"):
    """Broadcast error message to all connected clients."""
    message = {
        "type": "error",
        "data": {
            "message": error_message,
            "error_type": error_type
        }
    }
    await manager.broadcast_to_all(json.dumps(message))


@router.get("/ws/status")
async def websocket_status():
    """Get WebSocket connection status."""
    return {
        "active_connections": len(manager.active_connections),
        "connection_types": dict(manager.connection_types),
        "enabled": True
    } 