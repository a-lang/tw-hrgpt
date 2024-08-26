from langchain_chroma import Chroma
from embedding_func import embed_hf
from langchain_core.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
import os
import getpass
from rag_chains_func import ask_question

def display_conversation(history):
    i = 1
    for msg in history:
        if msg["role"] == "assistant":
            message(msg["content"], key=str(i))
        elif msg["role"] == "user":
            message(msg["content"], is_user=True, key=str(i) + "_user")
        i += 1

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
    if "message" not in st.session_state:
        st.session_state.message = []
        st.session_state.greetings = False

    # Greet user
    if not st.session_state.greetings:
        intro = "我是台灣人資法律小幫手，可以回答勞動法規的相關問題。"
        message(intro)
        # Add assistant response to chat history
        st.session_state.message.append({"role": "assistant", "content": intro})
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

        # Validate the prompt
        prompt = prompt.replace('"', "").replace("'", "")

        response = st.session_state.chat(prompt)
        # Add user message to chat history
        st.session_state.message.append({"role": "user", "content": prompt})
        # Add assistant response to chat history
        st.session_state.message.append({"role": "assistant", "content": response})

        if st.session_state.message:
            display_conversation(st.session_state.message)

        # Re-Show Example Prompts
        #st.rerun()

if __name__ == "__main__":
    main()

