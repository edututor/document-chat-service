from pydantic import BaseModel

# Analysis generation
class ChatGPTResponse(BaseModel):
    answer: str
