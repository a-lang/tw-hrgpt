from langchain_chroma import Chroma
from embedding_func import embed_hf, embed_cohere, rerank_hf
from langchain_core.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredMarkdownLoader
from langchain.schema import Document
import streamlit as st
from dotenv import load_dotenv
import os
import getpass

DB_PATH = "vector_db"
DATA_PATH = "docs"
LLM_MAX_TOKEN = 800
load_dotenv()
embed_model = embed_hf()
#embed_model = embed_cohere()
rerank_model = rerank_hf()

# Prepare the database
db = Chroma(collection_name="full_docs", persist_directory=DB_PATH, embedding_function=embed_model)

# LLM Model
## Google Gemini
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Provide your Google API Key")
llm_google = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    #model="gemini-1.5-pro-latest",
    max_output_tokens=LLM_MAX_TOKEN, 
    temperature=0.1)

## Groq LLaMA 3
from langchain_groq import ChatGroq
if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = getpass.getpass("Provide your Groq API Key")
llm_groq_llama3 = ChatGroq(
    #model="llama-3.1-8b-instant",
    #model="llama3-70b-8192",
    model="llama-3.1-70b-versatile",
    max_tokens=LLM_MAX_TOKEN, 
    temperature=0.1)

## Fireworks
from langchain_fireworks import ChatFireworks
llm_fireworks = ChatFireworks(
    #model="accounts/fireworks/models/llama-v3p1-70b-instruct",
    model="accounts/fireworks/models/mixtral-8x22b-instruct",
    temperature=0,
    max_tokens=LLM_MAX_TOKEN,
)

## Together
from langchain_together import ChatTogether
llm_together = ChatTogether(
    #model="meta-llama/Llama-3-70b-chat-hf",
    model="Qwen/Qwen2-72B-Instruct",
    temperature=0,
    max_tokens=LLM_MAX_TOKEN,
)



# Prompts
rag_template = """
### 指令
你是台灣人力資源法律的助理，熟悉就業法規、工作場所政策、員工權利和爭議解決的問題諮詢。所有回答請依據下面法規條文的內容來回答，
提供相關的法律參考和實用建議，以確保遵守法律並實踐人力資源法的最佳做法。
如果內容中查不到與問題相關的法條資訊，就回答：資料庫中查不到相關資訊。你只回答人資法律相關問題，如果非法律問題請說：我無法回答。
### 問題
你是誰
### 回答
我是台灣人資法律助理，可以回答勞動法規的相關問題。

### 法規條文
{context}

### 問題
回答問題前請先參考上述相關的法規條文：{question}

### 回答
回答："""

rag_template_google1 = """
### 指令
你是台灣人力資源法律的助理，熟悉就業法規、工作場所政策、員工權利和爭議解決的問題諮詢。提供相關的法律參考和實用建議，以確保遵守法律並實踐人力資源法的最佳做法。
其他的回答規則如下所列：
1.如果內容中查不到與問題相關的法條資訊，就回答：資料庫中查不到相關資訊。
2.你只回答人資法律相關問題，如果非法律問題請說：我無法回答。
3.所有回答請依據下面法規條文的內容來回答。
4.如果以下參考的"法規條文"內容為"空白"時，請回答：資料庫中查不到相關資訊。

### 問題
你是誰
### 回答
我是台灣人資法律助理，可以回答勞動法規的相關問題。

### 法規條文
{context}

### 問題
回答問題前請先參考上述相關的法規條文：{question}

### 回答
回答："""

rag_template_google2 = """
SYSTEM:
你是台灣人力資源法律的助理，熟悉勞動法規、工作場所政策、員工權利和爭議解決的問題諮詢。
依照下列規則回答問題：
1.所有回答請參考下面的法規內容，如果沒有與問題相關的內容，請回答：資料庫中查不到相關資訊。
2.你只回答人資法律相關問題，如果非法律問題請說：我無法回答。
3.以下法規內容如果是空白，請回答：資料庫中查不到相關資訊。
4.一律使用繁體中文回答，文字不要超過 800 字，語氣保持專業與禮貌。

QUESTION:
你是誰
ANSWER:
我是台灣人資法律助理，可以回答勞動法規的相關問題。

法規內容如下：
---------------
{context}
---------------
請根據上述的法規內容回答所有問題，如果沒有任何內容，請回答：資料庫中查不到相關資訊。

QUESTION:
問題：{question}
ANSWER:
回答："""

rag_template_google3 = """
角色設定:
你是台灣的人力資源法律顧問，專精於勞動法規、職場政策、員工權益與爭議處理。你的知識庫包含完整的台灣勞動法規。

任務:
針對使用者提出的問題，提供專業且符合知識庫包含的法規的解答。

回答原則:
* 專業用語: 使用專業且易懂的法律用語，簡潔並專業地回答使用者的問題
* 客觀中立: 提供客觀且中立的資訊，避免給予個人意見或建議。
* 字數限制: 每則回覆字數控制在800字以內。
* 資料依據: 你的回答必須基於已知的知識庫內容。如果無法從中得到答案，請說 "根據已知資訊無法回答該問題" 或 "沒有提供足夠的相關資訊"。 
  不允許在答案中添加編造成分。

問題範例:
* 問題: 員工要求加班費，公司該如何計算？
* 回答: 依據《勞動基準法》第24條規定，...

請注意:
* 法規更新: 法律條文可能隨時更新，請以最新版本為準。
* 個案差異: 實際狀況可能因個案差異而有所不同，建議使用者尋求專業法律意見。

知識庫:
{context}

開始提問吧！

問題: {question}
"""

rag_template_noanswer = """
You are an AI assistant, Do not answer any user question, just rewrite the one of following sentences in Traditional Chinese.
- 根據已知資訊無法回答該問題
- 沒有提供足夠的相關資訊
"""

qa_system_prompt = """
你是台灣人力資源法律的助理，熟悉就業法規、工作場所政策、員工權利和爭議解決的問題諮詢，其他的回答規則如下所列：
1.如果內容中查不到與問題相關的法條資訊，就回答：資料庫中查不到相關資訊。
2.你只回答人資法律相關問題，如果非法律問題請說：我無法回答。
3.所有回答請依據下面法規條文的內容來回答。
4.如果以下參考的"法規條文"內容為"空白"時，請回答：資料庫中查不到相關資訊。
5.回答的內容時應該簡單扼要，並包含引用的法規條文內容。
6.回答語氣保持專業與禮貌，文字不要超過 500 字。
### 問題
你是誰
### 回答
我是台灣人資法律助理，可以回答勞動法規的相關問題。

### 法規條文
{context}

"""

qa2_system_prompt = """
你是台灣人力資源法律的顧問，熟悉勞動法規、工作場所政策、員工權利和爭議解決的問題諮詢。"""

qa2_human_prompt = """
###
知識庫:
{context}
###

回答的規則:
- 你只回答人資法律相關問題，如果非法律相關問題，請說："我無法回答非人資法律相關的其他問題"。
- 請根據知識庫內容的條文來回答問題。
- 回答的內容時應該簡單扼要，並包含引用的法規條文內容。
- 回答語氣保持專業與禮貌，文字不要超過 800 字。
- 你的回答必須基於已知的知識庫內容。如果無法從中得到答案，請說 "根據已知資訊無法回答該問題" 或 "沒有提供足夠的相關資訊"。

問題: 你是誰
回答: 我是台灣人資法律助理，可以回答勞動法規的相關問題。

請根據知識庫的內容回答問題: {question}
"""


def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

def format_docs(docs):
    return "\n".join("法規名稱： " + os.path.basename(str(doc.metadata['source'])).split('.', 1)[0] + 
                       "\n" + str(doc.page_content) + "\n----" for doc in docs)

def load_files(dir=DATA_PATH):
    pages = []
    path = os.path.join(os.getcwd(), dir)

    for filename in os.listdir(path):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(path + '/' + filename)
            pages.extend(loader.load())
        elif filename.endswith(".txt"):
            loader = TextLoader(path + '/' + filename)
            pages.extend(loader.load())
        elif filename.endswith(".md"):
            loader = UnstructuredMarkdownLoader(path + '/' + filename)
            pages.extend(loader.load())
        else:
            print(f"Unsupported file type: {filename}")
        
    return pages

def load_files2(dir=DATA_PATH):
    docs = []
    path = os.path.join(os.getcwd(), dir)

    for filename in os.listdir(path):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(path + '/' + filename)
            pdf_pages = loader.load()
            doc = merge_pages_contenct(pdf_pages)
            docs.extend(doc)
        elif filename.endswith(".txt"):
            loader = TextLoader(path + '/' + filename)
            docs.extend(loader.load())
        elif filename.endswith(".md"):
            loader = UnstructuredMarkdownLoader(path + '/' + filename)
            docs.extend(loader.load())
        else:
            print(f"Unsupported file type: {filename}")
        
    return docs

def merge_pages_contenct(pages):
    merged_content = ""
    doc = []
    for page in pages:
        merged_content += page.page_content + "\n"

    doc = yield Document(page_content=merged_content, metadata={'source': page.metadata['source']})
    return doc

def split_text(docs: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=120, 
        add_start_index=True,
        separators=[
            "\uff0e",  # Fullwidth full stop "．"
            "\u3002",  # Ideographic full stop "。"
            "\uff0c",  # Fullwidth comma "，"
        ]
    )
    return text_splitter.split_documents(docs)

def simple_multi_chain(vectorstore, llm):
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough

    #retriever = vectorstore.as_retriever(search_type="mmr")
    retriever = simple_retriever(vectorstore)
    simple_mq_retriever = MultiQueryRetriever.from_llm(
        retriever=retriever,
        llm=llm
    )

    rag_prompt = PromptTemplate.from_template(rag_template_google1)
    
    rag_chain = (
        {"context": simple_mq_retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


def simple_retriever(vectorstore):
    retriever = vectorstore.as_retriever(search_type="mmr")
    #retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    return retriever

def multiq_retriever(vectorstore, llm):
    from typing import List
    from langchain_core.output_parsers import StrOutputParser, BaseOutputParser

    # Output parser will split the LLM result into a list of queries
    class LineListOutputParser(BaseOutputParser[List[str]]):
        """Output parser for a list of lines."""

        def parse(self, text: str) -> List[str]:
            lines = text.strip().split("\n")
            return lines
        
    output_parser = LineListOutputParser()

    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate three 
        different versions of the given user question to retrieve relevant documents from a vector 
        database. By generating multiple perspectives on the user question, your goal is to help
        the user overcome some of the limitations of the distance-based similarity search. 
        Provide these alternative questions separated by newlines. 
        Original question: {question}
        Please answer in Traditional Chinese and output as the following format.
        1.XXX
        2.XXX
        3.XXX
        """,
    )
    # Chain
    llm_chain = QUERY_PROMPT | llm | output_parser

    retriever = simple_retriever(vectorstore)
    retriever_from_llm_chain = MultiQueryRetriever(
        retriever=retriever,
        llm_chain=llm_chain,
        parser_key="lines"
    )
    return retriever_from_llm_chain

def multiq_chain(vectorstore, llm):
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough

    retriever = multiq_retriever(vectorstore, llm)

    rag_prompt = PromptTemplate.from_template(rag_template_google1)
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    return chain

def multi_memory_chain(vectorstore, llm):
    from langchain.memory import ConversationBufferMemory
    from langchain.chains import ConversationalRetrievalChain

    retriever = multiq_retriever(vectorstore, llm)

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
        )
    memory = st.session_state.memory
    
    custom_prompt = PromptTemplate.from_template(rag_template_google1)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        combine_docs_chain_kwargs={'prompt': custom_prompt},
        memory=memory
    )
    return conversation_chain

def save_to_pickle(obj, filename):
    import pickle
    with open(filename, "wb") as file:
        pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)

def load_from_pickle(filename):
    import pickle
    with open(filename, "rb") as file:
        return pickle.load(file)

def parent_retriever():
    """Loads the vector store and document store, initializing the retriever."""
    from langchain.retrievers import ParentDocumentRetriever
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain.storage import InMemoryStore

    doc_name = "parent_documents"
    doc_id_key = "doc_id"
    doc_file = "docstore.pkl"
    vector_store = Chroma(
        collection_name=doc_name, embedding_function=embed_model, persist_directory=DB_PATH
    )
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
    store_dict = load_from_pickle(os.path.join(DB_PATH, doc_file))
    store = InMemoryStore()
    store.mset(list(store_dict.items()))

    retriever = ParentDocumentRetriever(
        vectorstore=vector_store,
        docstore=store,
        child_splitter=child_splitter,
        id_key=doc_id_key,
    )
    return retriever

def parent_chain(llm):
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough

    retriever = parent_retriever()
    rag_prompt = PromptTemplate.from_template(rag_template_google2)

    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    return qa_chain


def parent_memory_chain(llm):
    from langchain.memory import ConversationBufferMemory
    from langchain.chains import ConversationalRetrievalChain

    retriever = parent_retriever()

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
        )
    memory = st.session_state.memory
    
    custom_prompt = PromptTemplate.from_template(rag_template)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        combine_docs_chain_kwargs={'prompt': custom_prompt},
        memory=memory
    )
    return conversation_chain

def parent_memory_chain2(llm, chat_memory=ChatMessageHistory()):
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.chat_history import BaseChatMessageHistory
    from langchain_core.runnables.history import RunnableWithMessageHistory

    qa_prompt = ChatPromptTemplate.from_messages(
        [ 
            ("system", qa_system_prompt), 
            MessagesPlaceholder("chat_history"),
            ("human", "回答問題前請先參考上述相關的法規條文：{input}"),
        ]
    )

    retriever = parent_retriever()

    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    retrieve_docs = (lambda x: x["input"]) | retriever
    rag_chain = RunnablePassthrough.assign(context=retrieve_docs).assign(
        answer=rag_chain_from_docs
    )

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        return chat_memory

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    return conversational_rag_chain

def rerank_retriever(vectorstore):
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import CrossEncoderReranker

    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 10})
    compressor = CrossEncoderReranker(model=rerank_model, top_n=3)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    return compression_retriever

def multiq_rerank_retriever(vectorstore, llm):
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import CrossEncoderReranker

    retriever = multiq_retriever(vectorstore, llm)
    compressor = CrossEncoderReranker(model=rerank_model, top_n=3)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    return compression_retriever

def rerank_chain(vectorstore, llm):
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.prompts import ChatPromptTemplate

    retriever = rerank_retriever(vectorstore)

    qa_prompt = ChatPromptTemplate.from_messages(
        [ 
            ("system", qa2_system_prompt), 
            ("human", qa2_human_prompt),
        ]
    )
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    return chain

def multiq_rerank_chain(vectorstore, llm):
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.prompts import ChatPromptTemplate

    retriever = multiq_rerank_retriever(vectorstore, llm)

    qa_prompt = ChatPromptTemplate.from_messages(
        [ 
            ("system", qa2_system_prompt), 
            ("human", qa2_human_prompt),
        ]
    )
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    return chain

def hybrid_retriever():
    from langchain.retrievers import EnsembleRetriever
    from langchain_community.retrievers import BM25Retriever

    text_splits = chunks
    retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 5})
    keyword_retriever = BM25Retriever.from_documents(text_splits)
    keyword_retriever.k =  5
    ensemble_retriever = EnsembleRetriever(
        retrievers=[retriever,
                    keyword_retriever],
                    weights=[0.5, 0.5])
    return ensemble_retriever

def hybrid_rerank_retriever():
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import CrossEncoderReranker

    retriever = hybrid_retriever()
    compressor = CrossEncoderReranker(model=rerank_model, top_n=3)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    return compression_retriever

def hybrid_rerank_chain(llm):
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.prompts import ChatPromptTemplate

    retriever = hybrid_rerank_retriever()

    qa_prompt = ChatPromptTemplate.from_messages(
        [ 
            ("system", qa2_system_prompt), 
            ("human", qa2_human_prompt),
        ]
    )
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    return chain




def ask_question(query):
    #llm = llm_google
    #llm = llm_groq_llama3
    #llm = llm_fireworks
    llm = llm_together

    
    #response = simple_multi_chain(db, llm).invoke(query)
    #response = multiq_chain(db, llm).invoke(query)
    #response = multi_memory_chain(db, llm)({"question": query})["answer"]
    #response = parent_memory_chain(llm)({"question": query})["answer"]
    #response = parent_memory_chain2(llm).invoke({"input": query}, config={"configurable": {"session_id": "abc123"}})["answer"]
    #response = parent_chain(llm).invoke(query)
    #response = rerank_chain(db, llm).invoke(query)
    response = multiq_rerank_chain(db, llm).invoke(query)
    #response = hybrid_rerank_chain(llm).invoke(query)

    #response = chain.invoke(
    #    {"question": query},
    #    config={"configurable": {"session_id": "foo"}}
    #    )

    return response

# For Hybrid Search only
#pages = load_files()
#chunks = split_text(pages)
docs = load_files2()
chunks = split_text(docs)


if __name__ == "__main__":

    response = ask_question("勞工可以申請最多幾天病假")
    print(response)
    # Streaming
    #streams = ask_question("流產是否可以申請生育給付？")
    #for chunk in streams:
    #    print(chunk, end="", flush=True)

