from langchain_chroma import Chroma
from embedding_func import embed_hf
from langchain_core.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import streamlit as st
from dotenv import load_dotenv
import os
import getpass

DB_PATH = "vector_db"
load_dotenv()
embed_model = embed_hf()

# Prepare the database
db = Chroma(collection_name="full_docs", persist_directory=DB_PATH, embedding_function=embed_model)

# LLM Model
## Google Gemini
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Provide your Google API Key")
llm_google = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.1)

## Groq LLaMA 3
from langchain_groq import ChatGroq
if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = getpass.getpass("Provide your Groq API Key")

llm_groq_llama3 = ChatGroq(model="llama3-8b-8192", temperature=0.1)


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

rag_template2 = """
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

rag_template3 = """
SYSTEM:
你是台灣人力資源法律的助理，熟悉勞動法規、工作場所政策、員工權利和爭議解決的問題諮詢。
依照下列規則回答問題：
1.所有回答請參考下面的法規內容，如果沒有與問題相關的內容，請回答：資料庫中查不到相關資訊。
2.你只回答人資法律相關問題，如果非法律問題請說：我無法回答。
3.以下法規內容如果是空白，請回答：資料庫中查不到相關資訊。
4.一律使用繁體中文回答，文字不要超過 500 字，語氣保持專業與禮貌。

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

qa_system_prompt = """
你是台灣人力資源法律的助理，熟悉就業法規、工作場所政策、員工權利和爭議解決的問題諮詢，其他的回答規則如下所列：
1.如果內容中查不到與問題相關的法條資訊，就回答：資料庫中查不到相關資訊。
2.你只回答人資法律相關問題，如果非法律問題請說：我無法回答。
3.所有回答請依據下面法規條文的內容來回答。
4.如果以下參考的"法規條文"內容為"空白"時，請回答：資料庫中查不到相關資訊。
5.回答的內容時應該簡單扼要，並包含引用的法規條文內容。
6.回答語氣保持專業與禮貌，文字不要超過 800 字。
### 問題
你是誰
### 回答
我是台灣人資法律助理，可以回答勞動法規的相關問題。

### 法規條文
{context}

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


def simple_multi_chain(vectorstore, llm):
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough

    #retriever = vectorstore.as_retriever(search_type="mmr")
    retriever = simple_retriever(vectorstore)
    simple_mq_retriever = MultiQueryRetriever.from_llm(
        retriever=retriever,
        llm=llm
    )

    rag_prompt = PromptTemplate.from_template(rag_template2)
    
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

def multi_retriever(vectorstore, llm):
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

def multi_chain(vectorstore, llm):
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough

    retriever = multi_retriever(vectorstore, llm)

    rag_prompt = PromptTemplate.from_template(rag_template2)
    
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

    retriever = multi_retriever(vectorstore, llm)

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
        )
    memory = st.session_state.memory
    
    custom_prompt = PromptTemplate.from_template(rag_template2)

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
    rag_prompt = PromptTemplate.from_template(rag_template3)

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
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder

    model = HuggingFaceCrossEncoder(model_name="maidalun1020/bce-reranker-base_v1")
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 10})
    compressor = CrossEncoderReranker(model=model, top_n=3)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    return compression_retriever

def rerank_chain(vectorstore, llm):
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough

    retriever = rerank_retriever(vectorstore)

    rag_prompt = PromptTemplate.from_template(rag_template3)
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    return chain


def ask_question(query):
    #response = simple_multi_chain(db, llm_google).invoke(query)
    #response = multi_chain(db, llm_google).invoke(query)
    #response = multi_memory_chain(db, llm_google)({"question": query})["answer"]
    #response = parent_memory_chain(llm_google)({"question": query})["answer"]
    #response = parent_memory_chain2(llm_google).invoke({"input": query}, config={"configurable": {"session_id": "abc123"}})["answer"]
    #response = parent_chain(llm_google).invoke(query)
    response = rerank_chain(db, llm_google).invoke(query)

    #response = chain.invoke(
    #    {"question": query},
    #    config={"configurable": {"session_id": "foo"}}
    #    )

    return response


def main():

    ap_title = "台灣人資法律AI小幫手"
    ap_icon = ":books:"
    st.set_page_config(
        page_title = ap_title,
        page_icon = ap_icon,
    )
    @st.cache_resource
    def initialize():
        chat = ask_question
        return chat

    st.session_state.chat = initialize()
    st.header(ap_title + " " + ap_icon, divider='rainbow')
    st.caption("請注意!! AI 提供的建議僅作參考，無法代表專業法律立場，你需要自行判斷其準確性。")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.greetings = False

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Greet user
    if not st.session_state.greetings:
        with st.chat_message("assistant"):
            intro = "我是台灣人資法律小幫手，可以回答勞動法規的相關問題。"
            st.markdown(intro)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": intro})
            st.session_state.greetings = True

    # Example Prompts
    example_prompts = [
        "育嬰留停有哪些規定事項要注意?",
        "勞基法有哪些關於特休假的規定?",
        "勞工可以申請最多幾天病假?",
    ]

    #example_prompts_help = [
    #    "Look for a specific card effect",
    #    "Search for card type: 'Vampires', card color: 'black', and ability: 'flying'",
    #    "Color cards and card type",
    #    "Specifc card effect to another mana color",
    #    "Search for card names",
    #    "Search for card types with specific abilities",
    #]

    button_cols = st.columns(3)
    button_cols_2 = st.columns(3)
    button_pressed = ""

    #if button_cols[0].button(example_prompts[0], help=example_prompts_help[0]):
    if button_cols[0].button(example_prompts[0]):
        button_pressed = example_prompts[0]
    elif button_cols[1].button(example_prompts[1]):
        button_pressed = example_prompts[1]
    elif button_cols[2].button(example_prompts[2]):
        button_pressed = example_prompts[2]

    if prompt := (st.chat_input("在這裡輸入你的問題?") or button_pressed):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Validate the prompt
        prompt = prompt.replace('"', "").replace("'", "")

        with st.chat_message("assistant"):
            with st.spinner(text="助理正在輸入..."):
                response = st.session_state.chat(prompt)

            st.markdown(response)
            #st.markdown(response['answer'])


        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        #st.session_state.messages.append({"role": "assistant", "content": response['answer']})

        # Re-Show Example Prompts
        #st.rerun()

if __name__ == "__main__":
    #response = ask_question("流產是否可以申請生育給付？")
    #print(response)
    # Streaming
    #streams = ask_question("流產是否可以申請生育給付？")
    #for chunk in streams:
    #    print(chunk, end="", flush=True)
    main()

