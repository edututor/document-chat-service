from pydantic import BaseModel

# Analysis generation
class ChatGPTResponse(BaseModel):
    answer: str

class TextToSpeechRequest(BaseModel):
    text: str
    language: str = "en"

class DocumentChatRequest(BaseModel):
    document_name: str
    query: str