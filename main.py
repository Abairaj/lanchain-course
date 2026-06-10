from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from pydantic import Field,BaseModel
from tavily import TavilyClient

client = TavilyClient()

class ResponseFormat(BaseModel):
    
    answer:str = Field(description='the response of the query')



def main():
    load_dotenv()

    @tool
    def search_web(query: str) -> str:
        """
        Searches the internet for information and returns
        relevant results.

        Use this tool for:
        - Weather and climate questions
        - Current events and news
        - Information about cities, countries, and locations
        - Facts that may change over time

        Do not use this tool for:
        - Simple arithmetic
        - General reasoning
        - Questions that can be answered from the conversation context

        Args:
            query: Search query to execute.

        Returns:
            Search result text.
        """
        print(f"querying the web with query: {query}")
        return client.search(query)
    
    llm = ChatOllama(
    model="qwen3:8b",
    temperature=0,
)
    tools = [search_web]
    
    agent = create_agent(model=llm,tools=tools,response_format=ResponseFormat)
    
    response = agent.invoke({"messages":HumanMessage(content="What is the climate in london")})
    
    print(response)



if __name__ == "__main__":
    main()
