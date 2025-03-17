from pydantic import BaseModel

# Analysis generation
class ChatGPTResponse(BaseModel):
    answer: str

class TextToSpeechRequest(BaseModel):
    text: str
    language: str = "en"