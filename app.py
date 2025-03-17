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
    """
    Converts input text into speech and returns an MP3 file.
    """
    try:
        # Generate speech from text
        tts = gTTS(text=request.text, lang=request.language, slow=False)

        # Save the generated speech to a temporary file
        file_path = "output.mp3"
        tts.save(file_path)

        # Return the audio file as a response
        return FileResponse(file_path, media_type="audio/mpeg", filename="speech.mp3")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating speech: {str(e)}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)

