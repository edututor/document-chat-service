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

        else:
            return JSONResponse(
                status_code=400,
                content={"message": f"The router agent didn't call a function"},
            )

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred while processing the request.")

@app.post("/api/tutoring")
def handle_tutoring(request: DocumentChatRequest):
    document_name = request.document_name
    user_query = request.query

    try:
        tutoring_router = TutoringRouter()
        final_response = tutoring_router.route_subject(user_query, document_name)

        return JSONResponse(
            status_code=200,
            content={"final_response": final_response},
        )

    except Exception as e:
        logger.error(f"Error in tutoring pipeline: {e}")
        raise HTTPException(status_code=500, detail="Internal error in tutoring service.")


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

