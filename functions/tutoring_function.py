from openai_client import OpenAiClient
from loguru import logger
from schemas import ChatGPTResponse
from agents.tutoring_router_agent import tutoring_router_agent

class TutoringRouter:
    def __init__(self):
        self.client = OpenAiClient()

    def route_subject(self, user_query, document_name):
        prompt = tutoring_router_agent.prompt(user_query)
        subject_response = self.client.query_gpt(prompt, ChatGPTResponse)

        if subject_response is None:
            logger.error("Subject routing failed. GPT returned None.")
            return "Error: Could not route the subject."

        subject_response_clean = subject_response.answer.strip().lower()

        logger.info(f"Subject routed to: {subject_response_clean}")

        if subject_response_clean == "math":
            return MathPipeline().handle_math_tutoring(user_query, document_name)
        elif subject_response_clean == "english":
            return EnglishPipeline().handle_english_tutoring(user_query, document_name)
        else:
            logger.warning(f"Unexpected subject response: {subject_response_clean}")
            return "Subject not supported yet. Please choose Math or English."

class MathPipeline:
    def __init__(self):
        self.client = OpenAiClient()

    def handle_math_tutoring(self, user_query, document_name):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a math tutor delivering a structured lecture. Provide comprehensive explanations, clear definitions, "
                    "step-by-step solutions, and illustrative examples to enhance understanding. "
                    "Format your response as a clear, educational, and engaging lecture."
                )
            },
            {"role": "user", "content": user_query}
        ]

        math_response = self.client.query_gpt(messages, ChatGPTResponse)

        if math_response is None:
            logger.error("Math tutoring failed. GPT returned None.")
            return "Error: Could not generate a math tutoring response."

        return math_response.answer

class EnglishPipeline:
    def __init__(self):
        self.client = OpenAiClient()

    def handle_english_tutoring(self, user_query, document_name):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an English tutor delivering a structured lecture. Provide comprehensive explanations, clear definitions, "
                    "grammar explanations, literature analysis, and illustrative examples. "
                    "Format your response as a clear, educational, and engaging lecture."
                )
            },
            {"role": "user", "content": user_query}
        ]

        english_response = self.client.query_gpt(messages, ChatGPTResponse)

        if english_response is None:
            logger.error("English tutoring failed. GPT returned None.")
            return "Error: Could not generate an English tutoring response."

        return english_response.answer
