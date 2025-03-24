from loguru import logger
from vector_manager import VectorManager
from pinecone import Pinecone
from agents.document_chat_agent import document_chat_agent
from config import settings
from schemas import ChatGPTResponse

class DocumentChat:

    def __init__(self):
        self.vector_manager = VectorManager()

    def answer_document_question(self, query, client, document_name):

        document_name = document_name.lower()
        query_embeddings = self.vector_manager.vectorize(client, query)         

        # Access Pinecone indexes
        logger.info("Accessing Pinecone indexes...")
        pc = Pinecone(api_key=settings.pinecone_api_key)
        text_index = pc.Index("document-text-index")
        try:
            text_results = text_index.query(
                        vector=query_embeddings,
                        top_k=15,
                        filter={"document_name": {"$eq": document_name}},
                        include_metadata=True
                    )["matches"]
        except Exception as e:
            logger.error(f"An error occured when querying Pinecone: {e}")
            return e   
        
        joined_chunks = ""
        for chunk in text_results:
            metadata = chunk.metadata
            joined_chunks += metadata["chunk"] + "\n\n"

        logger.info(f"Prompting ChatGPT...")        
        chunks_header = "**DOCUMENT CHUNKS:**"
        user_question = f"**USER QUESTION: \n\n{query}**"
        combined_input = chunks_header + "\n\n" + joined_chunks + "\n\n" + user_question
        prompt = document_chat_agent.prompt(combined_input)
        response = client.query_gpt(prompt, ChatGPTResponse)

        # Return the response of the chat-bot
        return response