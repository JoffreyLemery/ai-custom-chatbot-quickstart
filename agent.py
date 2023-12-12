from langchain.agents import Tool, ConversationalChatAgent, AgentExecutor
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory

from config import CHAT_MODEL
from tools.tool import create_vector_db_tool
from utils import is_answer_formatted_in_json, output_response, transform_to_json


class Agent:

    def __init__(self):
        self.llm = ChatOpenAI(temperature=0, model_name=CHAT_MODEL)
        self.agent_executor = self.create_agent_executor()

    def create_agent_executor(self):
        q_and_a_tool = create_vector_db_tool(llm=self.llm)
        tools = [
            Tool(
                name="Expert Amadeus",
                return_direct=True,
                func=lambda query: _parse_source_docs(q_and_a_tool, query),
                description="useful for when you need to answer questions related to Amadeus. To use only for question related to Amadeus"
            ),

            Tool(
                name="Out of scope",
                return_direct=True, 
                func=_unknown_response,
                description="For questions unrelated to Amadeus"
                )
            
        ]
        memory = ConversationBufferWindowMemory(llm=self.llm, k=10, memory_key="chat_history", return_messages=True,
                                                human_prefix="user", ai_prefix="Amadeus assistant", input_key="input")
        custom_agent = ConversationalChatAgent.from_llm_and_tools(llm=self.llm,
                                                                  tools=tools,
                                                                  verbose=True,
                                                                  max_iterations=3,
                                                                  handle_parsing_errors=True,
                                                                  memory=memory,
                                                                  input_variables=["input", "chat_history",
                                                                                   "agent_scratchpad"],
                                                                  system_message=
                                                                  f"""
                                                                  Have a conversation with a human, answering the 
                                                                  following as best you can and try to use a tool to help.
                                                                  Be generous in your answers ans provide exemples.
                                                                  Use bullets points as much as possible.
                                                                  You have access to the following tools: 
                                                                  ** Tools **
                                                                  Expert Amadeus - Useful for when you need to answer questions related Amadeus
                                                                  Out of scope - For questions unrelated to Amadeus.
                                                                  """
                                                                  )
        return AgentExecutor.from_agent_and_tools(agent=custom_agent, tools=tools, memory=memory,
                                                  verbose=True)

    def query_axsgent(self, user_input):
        try:
            response = self.agent_executor.run(input=user_input)
            if is_answer_formatted_in_json(response):
                return response
            return f"""
            {{
                "result": "{response}"
            }}"""

        except ValueError as e:
            response = str(e)
            response_prefix = "Could not parse LLM output: `\nAI: "
            if not response.startswith(response_prefix):
                raise e
            response_suffix = "`"
            if response.startswith(response_prefix):
                response = response[len(response_prefix):]
            if response.endswith(response_suffix):
                response = response[:-len(response_suffix)]
            output_response(response)
            return response


def _parse_source_docs(q_and_a_tool: RetrievalQA, query: str):
    result = q_and_a_tool({"question": query})
    return transform_to_json(result)

def _unknown_response(query):
  return "I'm sorry, but i have be design to answer to question to Amadeus. Could we stay in the scope ?"
