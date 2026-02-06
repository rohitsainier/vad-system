# src/handlers/webhook_handler.py
"""
Webhook handler for sending VAD events to external services
"""
import asyncio
import aiohttp
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
import json
import time
import hashlib
import hmac
import structlog

from ..core.state_machine import SpeechSegment
from .speech_handler import ProcessedSpeech

logger = structlog.get_logger(__name__)


@dataclass
class WebhookPayload:
    """Webhook payload structure"""
    event_type: str
    timestamp: float
    data: Dict[str, Any]
    signature: Optional[str] = None


class WebhookHandler:
    """
    Webhook handler for sending events to external services
    
    Features:
    - Multiple endpoint support
    - Retry with exponential backoff
    - Request signing
    - Batch sending
    - Rate limiting
    """
    
    def __init__(
        self,
        endpoints: List[str] = None,
        secret_key: Optional[str] = None,
        timeout: float = 10.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        batch_size: int = 10,
        batch_timeout: float = 5.0
    ):
        self.endpoints = endpoints or []
        self.secret_key = secret_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        
        self._batch_queue: List[WebhookPayload] = []
        self._batch_lock = asyncio.Lock()
        self._last_batch_time = time.time()
        
        self._success_count = 0
        self._failure_count = 0
    
    def add_endpoint(self, endpoint: str):
        """Add a webhook endpoint"""
        if endpoint not in self.endpoints:
            self.endpoints.append(endpoint)
    
    def remove_endpoint(self, endpoint: str):
        """Remove a webhook endpoint"""
        if endpoint in self.endpoints:
            self.endpoints.remove(endpoint)
    
    async def send_speech_start(self, segment: SpeechSegment):
        """Send speech start event"""
        payload = WebhookPayload(
            event_type="speech_start",
            timestamp=time.time(),
            data={
                "start_time_ms": segment.start_time_ms,
                "confidence": segment.confidence
            }
        )
        await self._queue_or_send(payload)
    
    async def send_speech_end(self, segment: SpeechSegment, include_audio: bool = False):
        """Send speech end event"""
        data = {
            "start_time_ms": segment.start_time_ms,
            "end_time_ms": segment.end_time_ms,
            "duration_ms": segment.duration_ms,
            "confidence": segment.confidence
        }
        
        if include_audio and segment.audio_data is not None:
            import base64
            # Convert to int16 and base64 encode
            audio_int16 = (segment.audio_data * 32767).astype('int16')
            data["audio_base64"] = base64.b64encode(audio_int16.tobytes()).decode()
        
        payload = WebhookPayload(
            event_type="speech_end",
            timestamp=time.time(),
            data=data
        )
        await self._queue_or_send(payload)
    
    async def send_custom_event(self, event_type: str, data: Dict[str, Any]):
        """Send custom event"""
        payload = WebhookPayload(
            event_type=event_type,
            timestamp=time.time(),
            data=data
        )
        await self._queue_or_send(payload)
    
    async def _queue_or_send(self, payload: WebhookPayload):
        """Queue payload for batching or send immediately"""
        async with self._batch_lock:
            self._batch_queue.append(payload)
            
            # Check if we should send batch
            should_send = (
                len(self._batch_queue) >= self.batch_size or
                time.time() - self._last_batch_time >= self.batch_timeout
            )
            
            if should_send:
                await self._send_batch()
    
    async def _send_batch(self):
        """Send batched payloads"""
        if not self._batch_queue:
            return
        
        batch = self._batch_queue.copy()
        self._batch_queue = []
        self._last_batch_time = time.time()
        
        for endpoint in self.endpoints:
            await self._send_to_endpoint(endpoint, batch)
    
    async def _send_to_endpoint(self, endpoint: str, payloads: List[WebhookPayload]):
        """Send payloads to a single endpoint"""
        body = {
            "payloads": [asdict(p) for p in payloads],
            "batch_id": hashlib.md5(f"{time.time()}".encode()).hexdigest()[:8]
        }
        
        # Sign request if secret key is configured
        if self.secret_key:
            body_bytes = json.dumps(body).encode()
            signature = hmac.new(
                self.secret_key.encode(),
                body_bytes,
                hashlib.sha256
            ).hexdigest()
        else:
            signature = None
        
        headers = {
            "Content-Type": "application/json",
            "X-VAD-Signature": signature or ""
        }
        
        for attempt in range(self.max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        endpoint,
                        json=body,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=self.timeout)
                    ) as response:
                        if response.status < 400:
                            self._success_count += len(payloads)
                            logger.debug(
                                "Webhook sent",
                                endpoint=endpoint,
                                status=response.status,
                                count=len(payloads)
                            )
                            return
                        else:
                            logger.warning(
                                "Webhook failed",
                                endpoint=endpoint,
                                status=response.status
                            )
                            
            except asyncio.TimeoutError:
                logger.warning(
                    "Webhook timeout",
                    endpoint=endpoint,
                    attempt=attempt + 1
                )
            except Exception as e:
                logger.error(
                    "Webhook error",
                    endpoint=endpoint,
                    error=str(e)
                )
            
            # Wait before retry
            if attempt < self.max_retries - 1:
                await asyncio.sleep(self.retry_delay * (2 ** attempt))
        
        self._failure_count += len(payloads)
        logger.error(
            "Webhook failed after retries",
            endpoint=endpoint,
            payloads=len(payloads)
        )
    
    async def flush(self):
        """Flush any remaining batched payloads"""
        async with self._batch_lock:
            await self._send_batch()
    
    @property
    def stats(self) -> Dict[str, int]:
        """Get webhook statistics"""
        return {
            "success_count": self._success_count,
            "failure_count": self._failure_count,
            "pending_count": len(self._batch_queue)
        }