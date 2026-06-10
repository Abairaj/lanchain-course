from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from pydantic import Field,BaseModel
from langchain_tavily import TavilySearch

class ResponseFormat(BaseModel):
    
    answer:str = Field(description='the response of the query')



def main():
    load_dotenv()

    
    llm = ChatOllama(
    model="qwen3:8b",
    temperature=0,
)
    tools = [TavilySearch()]
    
    agent = create_agent(model=llm,tools=tools,response_format=ResponseFormat)
    
    response = agent.invoke({"messages":HumanMessage(content="What is the climate in london")})
    
    print(response)



if __name__ == "__main__":
    main()
