import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

load_dotenv()


def main():
    information = """
    Elon Reeve Musk (/ˈiːlɒn/ EE-lon; born June 28, 1971) is a businessman and former public official known for his leadership of Tesla and SpaceX. Musk has been the wealthiest person in the world since 2025; as of June 2026, Forbes estimates his net worth to be US$835 billion.

Born into the wealthy Musk family in Pretoria, South Africa, Musk emigrated in 1989 to Canada; he has Canadian citizenship since his mother was born there. He received bachelor's degrees in 1997 from the University of Pennsylvania before moving to California to pursue business ventures. In 1995, Musk co-founded the software company Zip2. Following its sale in 1999, he co-founded X.com, an online payment company that later merged to form PayPal, which was acquired by eBay in 2002. Musk also became an American citizen in 2002.


    """

    summary_template = """
    given information {information} is about a person
    1. give a summary about the person
    2. give a fun fact about the person
    """

    prompt_template = PromptTemplate(
        variables=["information"], template=summary_template
    )

    llm = ChatOllama(
        model="qwen3:8b",
        temperature=0,
    )

    chain = prompt_template | llm

    response = chain.invoke({"information": information})

    print(response.content)


if __name__ == "__main__":
    main()
