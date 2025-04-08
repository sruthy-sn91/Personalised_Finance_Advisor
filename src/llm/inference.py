from config.config import config
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

class LLMInference:
    def __init__(self):
        self.api_key = config.GROQ_API_KEY
        self.model_name = config.MODEL_NAME  # e.g., "llama3-8b-8192"
        self.groq_chat = ChatGroq(groq_api_key=self.api_key, model_name=self.model_name)
        self.memory = []
        self.max_memory_length = 10

    def add_to_memory(self, role, content):
        if isinstance(content, (SystemMessage, HumanMessage, AIMessage)):
            msg = content
        else:
            if role.lower() == "user":
                msg = HumanMessage(content=content)
            elif role.lower() == "assistant":
                msg = AIMessage(content=content)
            else:
                msg = SystemMessage(content=content)
        self.memory.append(msg)
        if len(self.memory) > self.max_memory_length:
            self.memory = self.memory[-self.max_memory_length:]

    def trim_memory_if_needed(self, token_limit=3000):
        combined = " ".join([msg.content for msg in self.memory])
        if len(combined) > token_limit:
            self.memory = []

    def generate_answer(self, user_query, context_docs, user_profile=None):
        self.trim_memory_if_needed(token_limit=3000)

        system_instruction = (
            "You are a financial assistant. Answer queries concisely in 100 words or less. "
            "Use short sentences or 3-4 bullet points if helpful. Avoid repeating the question."
        )

        messages = [SystemMessage(content=system_instruction)]
        messages.extend(self.memory)

        if context_docs:
            context_text = "\n".join(context_docs)[:1000]
            messages.append(SystemMessage(content="Context: " + context_text))

        messages.append(HumanMessage(content=user_query))

        response_msg = self.groq_chat(messages)

        if isinstance(response_msg, AIMessage):
            final_text = response_msg.content
        else:
            final_text = str(response_msg)

        self.add_to_memory("user", user_query)
        self.add_to_memory("assistant", final_text)

        return final_text
