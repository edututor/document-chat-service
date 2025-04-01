from loguru import logger
from fastapi import FastAPI, HTTPException
from gtts import gTTS
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from openai_client import OpenAiClient
from schemas import TextToSpeechRequest, DocumentChatRequest
from functions.document_chat_function import DocumentChat
from functions.generate_quiz_function import QuizGenerator
from functions.tutoring_function import TutoringRouter
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
def handle_chat(request: DocumentChatRequest):
    document_name = request.document_name
    query = request.query

    try:    
        # Initialize required components
        client = OpenAiClient()
        function_to_call = client.router(query, document_name)

        if function_to_call["function_name"] == "doc_qa":
            document_chat_handler = DocumentChat()
            final_response = document_chat_handler.answer_document_question(query, client, document_name)
            return JSONResponse(
                status_code=200,
                content={"final_response": final_response.answer},
            )
        
        elif function_to_call["function_name"] == "generate_quiz":
            logger.info(f"Querying to create a new quiz for {query}")
            quiz_generator = QuizGenerator()

            return quiz_generator.generate_quiz(document_name, query)

        elif function_to_call["function_name"] == "tutoring":
            logger.info(f"Routing tutoring request for query: {query}")
            tutoring_router = TutoringRouter()
            final_response = tutoring_router.route_subject(query, document_name)
            return JSONResponse(
                status_code=200,
                content={"final_response": final_response},
            )

        else:
            return JSONResponse(
                status_code=400,
                content={"message": f"The router agent didn't call a function"},
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

