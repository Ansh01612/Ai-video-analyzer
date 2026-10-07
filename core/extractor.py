#Actionableitems , decision , questions 

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import os 
from pydantic import SecretStr


def get_llm():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set in the environment or .env file.")
    return ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", "openrouter/free"),
        api_key=SecretStr(api_key),
        base_url="https://openrouter.ai/api/v1",
        temperature=0.2,
    )



def build_chain(system_prompt : str):
    llm = get_llm()
    return (
        RunnablePassthrough() | RunnableLambda(lambda x : {"text" : x}) |ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human","{text}"),
    ]) | llm |StrOutputParser()
    )

def extract_action_items(transcript:str)->str:
    chain = build_chain(
         "You are an expert meeting analyst. From the meeting transcript, "
        "extract all action items. For each provide:\n"
        "- Task description\n"
        "- Owner (who is responsible)\n"
        "- Deadline (if mentioned, else write 'Not specified')\n\n"
        "Format as a numbered list. If none found say 'No action items found.'"
    )

    return chain.invoke(transcript)


def extract_key_decisions(transcript: str) -> str:
    chain = build_chain(
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all key decisions made. Format as a numbered list. "
        "If none found say 'No key decisions found.'"
    )
    return chain.invoke(transcript)


def extract_questions(transcript: str) -> str:
    chain = build_chain(
        "From the meeting transcript, extract all unresolved questions "
        "or topics needing follow-up. Format as a numbered list. "
        "If none found say 'No open questions found.'"
    )
    return chain.invoke(transcript)