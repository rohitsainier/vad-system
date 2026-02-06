# src/core/event_emitter.py
"""
Event emitter for pub/sub pattern in VAD system
"""
import asyncio
from typing import Dict, List, Callable, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import threading
import structlog
from collections import defaultdict

logger = structlog.get_logger(__name__)


class EventType(Enum):
    """Standard VAD event types"""
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"
    SPEECH_PROBABILITY = "speech_probability"
    STATE_CHANGE = "state_change"
    ERROR = "error"
    AUDIO_LEVEL = "audio_level"


@dataclass
class Event:
    """Generic event container"""
    type: Union[EventType, str]
    data: Dict[str, Any]
    timestamp: float
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class EventEmitter:
    """
    Thread-safe event emitter with sync and async support
    
    Features:
    - Multiple listeners per event
    - Wildcard listeners (listen to all events)
    - Once listeners (auto-remove after first call)
    - Async listener support
    - Thread-safe operations
    """
    
    def __init__(self):
        self._listeners: Dict[str, List[Callable]] = defaultdict(list)
        self._once_listeners: Dict[str, List[Callable]] = defaultdict(list)
        self._wildcard_listeners: List[Callable] = []
        self._lock = threading.RLock()
        self._async_queue: Optional[asyncio.Queue] = None
    
    def on(self, event_type: Union[EventType, str], callback: Callable) -> 'EventEmitter':
        """
        Register a listener for an event type
        
        Args:
            event_type: Type of event to listen for
            callback: Function to call when event occurs
            
        Returns:
            self for chaining
        """
        event_name = event_type.value if isinstance(event_type, EventType) else event_type
        
        with self._lock:
            if callback not in self._listeners[event_name]:
                self._listeners[event_name].append(callback)
                logger.debug("Listener registered", event=event_name)
        
        return self
    
    def once(self, event_type: Union[EventType, str], callback: Callable) -> 'EventEmitter':
        """
        Register a one-time listener (removed after first call)
        
        Args:
            event_type: Type of event to listen for
            callback: Function to call when event occurs
            
        Returns:
            self for chaining
        """
        event_name = event_type.value if isinstance(event_type, EventType) else event_type
        
        with self._lock:
            self._once_listeners[event_name].append(callback)
        
        return self
    
    def on_any(self, callback: Callable) -> 'EventEmitter':
        """
        Register a wildcard listener (called for all events)
        
        Args:
            callback: Function to call for any event
            
        Returns:
            self for chaining
        """
        with self._lock:
            if callback not in self._wildcard_listeners:
                self._wildcard_listeners.append(callback)
        
        return self
    
    def off(self, event_type: Union[EventType, str], callback: Callable) -> 'EventEmitter':
        """
        Remove a listener
        
        Args:
            event_type: Type of event
            callback: Function to remove
            
        Returns:
            self for chaining
        """
        event_name = event_type.value if isinstance(event_type, EventType) else event_type
        
        with self._lock:
            if callback in self._listeners[event_name]:
                self._listeners[event_name].remove(callback)
            if callback in self._once_listeners[event_name]:
                self._once_listeners[event_name].remove(callback)
        
        return self
    
    def off_all(self, event_type: Optional[Union[EventType, str]] = None) -> 'EventEmitter':
        """
        Remove all listeners for an event type (or all events)
        
        Args:
            event_type: Optional event type, or None for all
            
        Returns:
            self for chaining
        """
        with self._lock:
            if event_type is None:
                self._listeners.clear()
                self._once_listeners.clear()
                self._wildcard_listeners.clear()
            else:
                event_name = event_type.value if isinstance(event_type, EventType) else event_type
                self._listeners[event_name] = []
                self._once_listeners[event_name] = []
        
        return self
    
    def emit(self, event_type: Union[EventType, str], data: Dict[str, Any] = None, **kwargs) -> bool:
        """
        Emit an event synchronously
        
        Args:
            event_type: Type of event
            data: Event data
            **kwargs: Additional event data
            
        Returns:
            True if any listeners were called
        """
        import time
        
        event_name = event_type.value if isinstance(event_type, EventType) else event_type
        event_data = {**(data or {}), **kwargs}
        
        event = Event(
            type=event_type,
            data=event_data,
            timestamp=time.time() * 1000
        )
        
        listeners_called = 0
        
        with self._lock:
            # Get all applicable listeners
            regular = list(self._listeners.get(event_name, []))
            once = list(self._once_listeners.get(event_name, []))
            wildcards = list(self._wildcard_listeners)
            
            # Clear once listeners
            self._once_listeners[event_name] = []
        
        # Call listeners outside lock
        for callback in regular + once:
            try:
                if asyncio.iscoroutinefunction(callback):
                    # Schedule async callback
                    asyncio.create_task(callback(event))
                else:
                    callback(event)
                listeners_called += 1
            except Exception as e:
                logger.error("Listener error", event=event_name, error=str(e))
        
        # Call wildcard listeners
        for callback in wildcards:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(event))
                else:
                    callback(event)
                listeners_called += 1
            except Exception as e:
                logger.error("Wildcard listener error", error=str(e))
        
        return listeners_called > 0
    
    async def emit_async(self, event_type: Union[EventType, str], data: Dict[str, Any] = None, **kwargs) -> bool:
        """
        Emit an event asynchronously
        
        Args:
            event_type: Type of event
            data: Event data
            **kwargs: Additional event data
            
        Returns:
            True if any listeners were called
        """
        import time
        
        event_name = event_type.value if isinstance(event_type, EventType) else event_type
        event_data = {**(data or {}), **kwargs}
        
        event = Event(
            type=event_type,
            data=event_data,
            timestamp=time.time() * 1000
        )
        
        tasks = []
        
        with self._lock:
            regular = list(self._listeners.get(event_name, []))
            once = list(self._once_listeners.get(event_name, []))
            wildcards = list(self._wildcard_listeners)
            self._once_listeners[event_name] = []
        
        for callback in regular + once + wildcards:
            try:
                if asyncio.iscoroutinefunction(callback):
                    tasks.append(callback(event))
                else:
                    callback(event)
            except Exception as e:
                logger.error("Listener error", event=event_name, error=str(e))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        return len(regular) + len(once) + len(wildcards) > 0
    
    def listener_count(self, event_type: Union[EventType, str] = None) -> int:
        """Get number of listeners for an event type"""
        with self._lock:
            if event_type is None:
                return sum(len(l) for l in self._listeners.values()) + len(self._wildcard_listeners)
            
            event_name = event_type.value if isinstance(event_type, EventType) else event_type
            return len(self._listeners.get(event_name, [])) + len(self._once_listeners.get(event_name, []))


class AsyncEventEmitter(EventEmitter):
    """
    Async-first event emitter with queue support
    """
    
    def __init__(self, queue_size: int = 1000):
        super().__init__()
        self._queue = asyncio.Queue(maxsize=queue_size)
        self._running = False
    
    async def start_processing(self):
        """Start processing events from queue"""
        self._running = True
        
        while self._running:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=0.1)
                await self.emit_async(event.type, event.data)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error("Event processing error", error=str(e))
    
    def stop_processing(self):
        """Stop event processing"""
        self._running = False
    
    async def queue_event(self, event_type: Union[EventType, str], data: Dict[str, Any] = None):
        """Queue an event for processing"""
        import time
        
        event = Event(
            type=event_type,
            data=data or {},
            timestamp=time.time() * 1000
        )
        
        try:
            await self._queue.put(event)
        except asyncio.QueueFull:
            logger.warning("Event queue full, dropping event", event=str(event_type))