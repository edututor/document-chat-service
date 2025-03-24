import requests
from fastapi.responses import JSONResponse
from fastapi import HTTPException
from config import settings


class QuizGenerator:

    def __init__(self):
        self.quiz_service_url = settings.quiz_service_url

    def generate_quiz(self, document_name, user_query):

        try:
            # Build the payload that matches the QuizRequest model
            payload = {
                "document_name": document_name,
                "user_query": user_query
            }

            response = requests.post(self.quiz_service_url, json=payload)
            response.raise_for_status()  # raises HTTPError if not 200-299

            quiz_response_data = response.json()  

            return JSONResponse(
                status_code=200,
                content=quiz_response_data
            )

        except requests.HTTPError as http_err:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Quiz generator service error: {str(http_err)}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected error while calling quiz service: {str(e)}"
            )