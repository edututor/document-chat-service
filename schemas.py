from pydantic import BaseModel
from typing import List, Literal

# Chat message structure for TTS
class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class TextToSpeechRequest(BaseModel):
    chat_history: List[ChatMessage]
    language: str = "en"

class ChatGPTResponse(BaseModel):
    answer: str

class DocumentChatRequest(BaseModel):
    document_name: str
    query: str
