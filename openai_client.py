from openai import OpenAI
from config import settings
from functions.function_definitions import generate_quiz_function, answer_doc_function , tutoring_function

import json


class OpenAiClient:
    def __init__(self) -> None:
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model

    def query_gpt(self, messages, response_format):
        completion = self.client.beta.chat.completions.parse(
            model=self.model, messages=messages, response_format=response_format
        )
        return completion.choices[0].message.parsed
    
    def router(self, user_query: str, document_name: str):
        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model, 
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a routing agent. Your ONLY job is to decide whether "
                            "the user needs the 'doc_qa' function or the 'generate_quiz' "
                            "function. Then provide the correct arguments. DO NOT return any text "
                            "outside of the function call. Always follow the function schema exactly."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Selected document: {document_name}\nUser query: {user_query}"
                    }
                ],
                functions=[answer_doc_function, generate_quiz_function],
                function_call="auto"
            )
            
            top_message = response.choices[0].message

            # We expect that the model calls exactly one of the two functions:
            if top_message.function_call:
                func_name = top_message.function_call.name
                raw_args = top_message.function_call.arguments
                args = json.loads(raw_args)  # skip validation for now
                return {
                    "function_name": func_name,
                    "arguments": args
                }
            else:
                # If the model doesn't call a function (unlikely if instructions are clear),
                # we can handle or log it:
                return {
                    "function_name": None,
                    "arguments": {},
                    "error": "No function call was made by the model."
                }
            
        except Exception as e:
            return f"An error occured while using the router: {e}"
    
    def generate_embeddings(self, text_list):
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-large", input=text_list
            )
            #embeddings = [item['embedding'] for item in response['data']]
            return response
        except Exception as e:
            print(f"An error occurred: {e}")
            return []
    