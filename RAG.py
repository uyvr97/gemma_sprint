import re
import os
import traceback
from dotenv import load_dotenv
import deepl

from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.callbacks.manager import CallbackManager
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import chainlit as cl
from langchain.chains import RetrievalQA,RetrievalQAWithSourcesChain
from langchain_huggingface import HuggingFaceEmbeddings

PROMPT = PromptTemplate.from_template( 
    template = """
    You are an assitant to a Formula 1 team.  
    Your job is to answer team questions to the best of your ability.  
    You will be given several excerpts from regulations to help you in answering.  
    Some of the provided regulations may be irrelevant,
    so ignore these and only use those that appear to be relevant. 
    Do not mention or cite regulations that are not given to you.  
    Answer succinctly and make reference to the relevant regulation sections.
    Think carefully about the question and the provided regulations. 
    Check that your response makes sense before answering.
    Question: {question}
    Context: {context}
    Answer:
    """
)

def isKorean(text):
    pattern_hangul = re.compile('[\u3131-\u3163\uac00-\ud7a3]+')
    word = pattern_hangul.search(str(text))
    
    if word and word != 'None':
        print('Korean:', word)
        return True
    else:
        print('Not Korean:', word)
        return False

def revise_question(chat, query):
    system = ("")
    human = """
    You are an assitant to a Formula 1 team.
    Rephrase the follow up question to be a standalone question.
    Question: {question}
    """
    
    # prompt = ChatPromptTemplate.from_messages([("system", system), MessagesPlaceholder(variable_name="history"), ("human", human)])
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    print('prompt:', prompt)

    chain = prompt | chat
    try: 
        revised_question = chain.invoke({"question": query})
        print(f"result: {revised_question}")

    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)        
        raise Exception ("Not able to request to LLM")
            
    return revised_question 

def translate_text(text):

    if isKorean(text)==True :
        target_language = "EN-US"
    else :
        target_language = "KO"

    translator = load_translator()
    result = translator.translate_text(text, target_lang=target_language)
    print(f'translation: {result}')

    return result

def load_translator():
    load_dotenv()
    deepl_api_key = os.getenv("DEEPL_API_KEY")
    translator = deepl.Translator(deepl_api_key)
    return translator

def load_llm():
    llm = Ollama(
        model="gemma2:2b",
        verbose=True,
        callback_manager=CallbackManager(handlers=[StreamingStdOutCallbackHandler()]),
    )
    return llm

def retrieval_qa_chain(llm, vectorstore):
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt" : PROMPT},
        return_source_documents=True,
    )
    return qa_chain

def qa_bot():
    llm=load_llm() 
    DB_PATH = "vectorstores/db/"
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
    qa = retrieval_qa_chain(llm,vectorstore)
    return qa 

@cl.on_chat_start
async def start():
    chain=qa_bot()
    msg=cl.Message(content="Firing up F1 regulration info bot...")
    await msg.send()
    msg.content= "Hi, welcome to F1 regularation info bot. What is your query?"
    await msg.update()
    cl.user_session.set("chain",chain)

@cl.on_message
async def main(message):
    chain=cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True,
        answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached=True

    question = message.content
    if isKorean(question)==True:
        question = translate_text(question)

    chat = load_llm()
    revised_question = revise_question(chat, question)

    res=await chain.ainvoke({"query": revised_question}, callbacks=[cb])
    print(f"response: {res}")
    answer=res["result"]

    if isKorean(question)==True:
        answer = translate_text(answer)

    await cl.Message(content=answer).send() 
