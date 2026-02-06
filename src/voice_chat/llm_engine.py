# src/voice_chat/llm_engine.py
"""
LLM Engine using Ollama for local inference
FIXED: Auto-detect Ollama host for WSL2 compatibility
"""
import time
import os
from typing import Optional, List, Dict, AsyncIterator
from dataclasses import dataclass, field
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class LLMResponse:
    """Response from LLM"""
    text: str
    model: str
    processing_time_ms: float
    tokens_generated: int = 0
    tokens_per_second: float = 0.0


@dataclass
class ChatMessage:
    """A single chat message"""
    role: str  # "system", "user", "assistant"
    content: str


class LLMEngine:
    """
    LLM Engine using Ollama
    
    Features:
    - Local inference (no API calls)
    - Auto-detect Ollama host (works in WSL2)
    - Streaming responses
    - Conversation history
    - System prompt customization
    """
    
    DEFAULT_SYSTEM_PROMPT = """You are a helpful, friendly voice assistant. 
Keep your responses concise and conversational since they will be spoken aloud.
Aim for 1-3 sentences unless the user asks for more detail.
Do not use markdown formatting, bullet points, or special characters.
Speak naturally as if having a conversation."""
    
    def __init__(
        self,
        model: str = "llama3.1:8b",
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 256,
        ollama_host: Optional[str] = None
    ):
        self.model = model
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Auto-detect Ollama host
        if ollama_host:
            self.ollama_host = ollama_host
        else:
            self.ollama_host = os.environ.get(
                'OLLAMA_HOST',
                self._detect_ollama_host()
            )
        
        self._history: List[ChatMessage] = []
        self._is_initialized = False
    
    @staticmethod
    def _detect_ollama_host() -> str:
        """
        Auto-detect Ollama host
        Handles WSL2 where Ollama might run on Windows side
        """
        import urllib.request
        
        candidates = [
            "http://localhost:11434",
            "http://127.0.0.1:11434",
            "http://172.29.160.1:11434"
        ]
        
        # In WSL2, Ollama might be on the Windows host
        try:
            with open('/etc/resolv.conf', 'r') as f:
                for line in f:
                    if line.strip().startswith('nameserver'):
                        windows_ip = line.strip().split()[1]
                        if windows_ip not in ('127.0.0.1', 'localhost'):
                            candidates.append(f"http://{windows_ip}:11434")
                            break
        except (FileNotFoundError, IndexError):
            pass
        
        # Also check OLLAMA_HOST env var
        env_host = os.environ.get('OLLAMA_HOST')
        if env_host:
            if not env_host.startswith('http'):
                env_host = f"http://{env_host}"
            candidates.insert(0, env_host)
        
        for host in candidates:
            try:
                req = urllib.request.Request(
                    f"{host}/api/tags",
                    method='GET'
                )
                urllib.request.urlopen(req, timeout=3)
                logger.info("Ollama found", host=host)
                return host
            except Exception:
                continue
        
        logger.warning("Ollama not found on any host, using default")
        return "http://localhost:11434"
    
    def initialize(self):
        """Verify Ollama connection and model availability"""
        if self._is_initialized:
            return
        
        try:
            import ollama
        except ImportError:
            raise RuntimeError(
                "ollama package not installed. Run: pip install ollama"
            )
        
        try:
            client = ollama.Client(host=self.ollama_host)
            
            # Test connection
            models_response = client.list()
            available = []
            
            if hasattr(models_response, 'models'):
                available = [m.model for m in models_response.models]
            elif isinstance(models_response, dict):
                available = [m.get('name', m.get('model', '')) 
                           for m in models_response.get('models', [])]
            
            logger.info("Ollama connected",
                       host=self.ollama_host,
                       available_models=available[:5])
            
            # Check if model is available
            model_base = self.model.split(':')[0]
            model_found = any(model_base in m for m in available)
            
            if not model_found and available:
                logger.warning(
                    f"Model {self.model} not found. Available: {available}"
                )
                # Try to pull
                try:
                    logger.info(f"Pulling model {self.model}...")
                    client.pull(self.model)
                except Exception as e:
                    raise RuntimeError(
                        f"Model {self.model} not available and pull failed: {e}\n"
                        f"Available models: {available}\n"
                        f"Run: ollama pull {self.model}"
                    )
            
            self._is_initialized = True
            logger.info("LLM Engine initialized", 
                       model=self.model, 
                       host=self.ollama_host)
            
        except Exception as e:
            error_msg = str(e)
            if "Connection refused" in error_msg or "urlopen error" in error_msg:
                raise RuntimeError(
                    f"Cannot connect to Ollama at {self.ollama_host}\n"
                    f"Make sure Ollama is running:\n"
                    f"  - Check: curl {self.ollama_host}/api/tags\n"
                    f"  - Start: ollama serve\n"
                    f"  - WSL2: Ollama might be on Windows side.\n"
                    f"    Set: export OLLAMA_HOST=http://<windows-ip>:11434"
                )
            raise
    
    def chat(self, user_message: str) -> LLMResponse:
        """Send a message and get a response"""
        if not self._is_initialized:
            self.initialize()
        
        import ollama
        
        start_time = time.time()
        
        self._history.append(ChatMessage(role="user", content=user_message))
        
        # Build messages
        messages = [{"role": "system", "content": self.system_prompt}]
        for msg in self._history[-20:]:
            messages.append({"role": msg.role, "content": msg.content})
        
        client = ollama.Client(host=self.ollama_host)
        
        try:
            response = client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                }
            )
        except Exception as e:
            logger.error("LLM chat error", error=str(e))
            # Remove failed user message from history
            self._history.pop()
            raise
        
        # Extract response text
        if isinstance(response, dict):
            assistant_text = response.get('message', {}).get('content', '').strip()
            eval_count = response.get('eval_count', 0)
            eval_duration = response.get('eval_duration', 0)
        else:
            assistant_text = response.message.content.strip()
            eval_count = getattr(response, 'eval_count', 0)
            eval_duration = getattr(response, 'eval_duration', 0)
        
        if not assistant_text:
            assistant_text = "I'm sorry, I didn't generate a response. Could you try again?"
        
        self._history.append(ChatMessage(role="assistant", content=assistant_text))
        
        processing_time = (time.time() - start_time) * 1000
        
        # Calculate tokens per second
        if eval_duration and eval_duration > 0:
            tps = eval_count / (eval_duration / 1e9)
        else:
            tps = eval_count / (processing_time / 1000) if processing_time > 0 else 0
        
        result = LLMResponse(
            text=assistant_text,
            model=self.model,
            processing_time_ms=processing_time,
            tokens_generated=eval_count,
            tokens_per_second=tps
        )
        
        logger.info(
            "LLM response",
            text=assistant_text[:80],
            tokens=eval_count,
            time_ms=f"{processing_time:.0f}",
            tps=f"{tps:.1f}"
        )
        
        return result
    
    async def chat_stream(self, user_message: str) -> AsyncIterator[str]:
        """Stream response token by token"""
        if not self._is_initialized:
            self.initialize()
        
        import ollama
        
        self._history.append(ChatMessage(role="user", content=user_message))
        
        messages = [{"role": "system", "content": self.system_prompt}]
        for msg in self._history[-20:]:
            messages.append({"role": msg.role, "content": msg.content})
        
        client = ollama.Client(host=self.ollama_host)
        
        full_response = []
        
        stream = client.chat(
            model=self.model,
            messages=messages,
            options={
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
            stream=True
        )
        
        for chunk in stream:
            if isinstance(chunk, dict):
                token = chunk.get('message', {}).get('content', '')
            else:
                token = chunk.message.content
            
            full_response.append(token)
            yield token
        
        complete_text = "".join(full_response).strip()
        self._history.append(ChatMessage(role="assistant", content=complete_text))
    
    def clear_history(self):
        """Clear conversation history"""
        self._history.clear()
        logger.info("Conversation history cleared")
    
    @property
    def history(self) -> List[ChatMessage]:
        return self._history.copy()
    
    def cleanup(self):
        """Cleanup resources"""
        self._history.clear()
        self._is_initialized = False