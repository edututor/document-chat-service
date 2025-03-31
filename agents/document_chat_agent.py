class Agent:
    def __init__(self, name, role, function, query='None') -> None:
        self.role: str = role
        self.name: str = name
        self.function: str = function
        self.query: str = query

    def set_max_tokens(self, summary_token_target):
        self.function.format(summary_token_target=summary_token_target)

    def prompt(self, input_prompt):
        system_prompt = f"You are a: {self.name}. Your role is: {self.role}. Your function is: {self.function}. Based on your role and function, do the task you are given. Do not give me anything else other than the given task"

        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": input_prompt}]

# Quiz generator agent:
document_chat_agent = Agent(
    name="Document Chat Agent",
    role=(
        "You are a conversational document analysis agent that uses provided document chunks to answer user queries. "
        "Your task is to help users understand and engage with the content by delivering clear, concise, and evidence-based responses."
    ),
    function=(
        "Your function is to analyze the given document chunks, which have been selected based on their relevance to the user's query, "
        "and to generate an answer that directly addresses the query. Your responses should: \n"
        "1) Directly reference and base your answer on the provided text chunks. \n"
        "2) Clearly explain how the information from the text supports your answer. \n"
        "3) Avoid introducing external or unrelated information, ensuring that your response is strictly grounded in the provided content."
    )
)

