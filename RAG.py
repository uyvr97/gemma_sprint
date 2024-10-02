from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
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
    Think carefully about the question and the provided regulations and definitions. 
    Check that your response makes sense before answering.
    Question: {question}
    Context: {context}
    Answer:
    """
)

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
    msg=cl.Message(content="Firing up the research info bot...")
    await msg.send()
    msg.content= "Hi, welcome to research info bot. What is your query?"
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

    res=await chain.ainvoke({"query": message.content}, callbacks=[cb])
    print(f"response: {res}")
    answer=res["result"]
    sources=res["source_documents"]

    if sources:
        answer += f"\nSources: "+str(str(sources))
    else:   
        answer += f"\nNo Sources found"

    await cl.Message(content=answer).send() 
