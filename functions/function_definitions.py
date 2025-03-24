# 1) Doc QA function definition
answer_doc_function  = {
    "name": "doc_qa",
    "description": "Answer user questions by searching a knowledge base or documents.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The user's question for the document Q&A pipeline",
            }
        },
        "required": ["query"],
    },
}

# 2) Quiz generation function definition
generate_quiz_function = {
    "name": "generate_quiz",
    "description": "Create a quiz based on user input or content chunks.",
    "parameters": {
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "The topic or text on which to generate questions.",
            }
        },
        "required": ["topic"],
    },
}

# 3) Tutoring function definition
tutoring_function = {
    "name": "tutoring",
    "description": "Engage in a tutoring session or provide explanations for a given question.",
    "parameters": {
        "type": "object",
        "properties": {
            "student_question": {
                "type": "string",
                "description": "The student's question or topic for tutoring."
            }
        },
        "required": ["student_question"],
    },
}
