from loguru import logger
from fastapi import FastAPI, HTTPException, Query, Depends
from gtts import gTTS
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from agents import document_chat_agent
from openai_client import OpenAiClient
from schemas import ChatGPTResponse, TextToSpeechRequest
from vector_manager import VectorManager
from pinecone import Pinecone
from config import settings
import os
from typing import List


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this for production (e.g., ["https://your-frontend.com"])
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/document-chat")
def generate_analysis(document_name: str = Query(...), query: str = Query(...)):
    try:    
        # Initialize required components
        client = OpenAiClient()
        vector_manager = VectorManager()  

        document_name = document_name.lower()
        query_embeddings = vector_manager.vectorize(query)         

        # Access Pinecone indexes
        logger.info("Accessing Pinecone indexes...")
        pc = Pinecone(api_key=settings.pinecone_api_key)
        text_index = pc.Index("document-text-embeddings")
        try:
            text_results = text_index.query(
                        vector=query_embeddings,
                        top_k=15,
                        filter={"company_name": {"$eq": document_name}},
                        include_metadata=True
                    )["matches"]
        except Exception as e:
            logger.error(f"An error occured when querying Pinecone: {e}")
            return JSONResponse(
                status_code=500,
                content={"message": f"An error occured when querying Pinecone: {e}"},
            )
        metadata = text_results.metadata

        logger.info(f"Prompting ChatGPT...")        
        chunks = [chunk for chunk in metadata["chunk"]]
        chunks_header = "**DOCUMENT CHUNKS:**"
        user_question = f"**USER QUESTION: \n\n{query}**"
        combined_input = chunks_header + "\n\n" + "\n\n".join(chunks) + "\n\n" + user_question
        prompt = document_chat_agent.prompt(combined_input)
        response = client.query_gpt(prompt, ChatGPTResponse)

        # Return the response of the chat-bot
        return JSONResponse(
            status_code=200,
            content={
                "message": "Analysis and report generated successfully.",
                "response": response.answer
            }
        )

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred while processing the request.")


@app.post("/api/text-to-speech")
async def text_to_speech(request: TextToSpeechRequest):
    return tts_pipeline(chat_history=request.chat_history, language=request.language)



def tts_pipeline(chat_history: List[dict], language: str = "en") -> FileResponse:
    """
    Pipeline to process chat history and return speech audio.
    Looks for the last assistant or user message and converts it to speech.
    """
    # Step 1: Find the last message that can be converted
    for message in reversed(chat_history):
        if message["role"] in ["user", "assistant"] and message["content"].strip():
            text_to_convert = message["content"]
            break
    else:
        raise HTTPException(status_code=400, detail="No valid message found for TTS conversion.")

    # Step 2: Generate speech
    tts = gTTS(text=text_to_convert, lang=language, slow=False)
    file_path = "speech_output.mp3"
    tts.save(file_path)

    # Step 3: Return the file
    return FileResponse(file_path, media_type="audio/mpeg", filename="speech.mp3")



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)

