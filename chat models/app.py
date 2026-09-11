import streamlit as st

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


# ------------------ Page Configuration ------------------

st.set_page_config(
    page_title="AI Personality Chatbot",
    page_icon="🤖",
    layout="centered",
)


# ------------------ Load Local LLM ------------------

@st.cache_resource
def load_model():

    llm = HuggingFacePipeline.from_model_id(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        task="text-generation",
        pipeline_kwargs={
            "max_new_tokens": 64,
            "do_sample": False,
            "repetition_penalty": 1.03,
        },
    )

    return ChatHuggingFace(llm=llm)


chat_model = load_model()


# ------------------ AI Modes ------------------

modes = {
    "😡 Angry": (
        "You are an angry AI agent. "
        "You respond aggressively and impatiently, "
        "but remain helpful and do not insult the user."
    ),

    "😂 Funny": (
        "You are a very funny AI agent. "
        "You respond with humor, jokes, and a playful personality "
        "while still being helpful."
    ),

    "😢 Sad": (
        "You are a very sad AI agent. "
        "You respond in a melancholic and emotional tone "
        "while still being helpful."
    ),
}


# ------------------ Initialize Session State ------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


if "mode" not in st.session_state:
    st.session_state.mode = None


# ------------------ Sidebar ------------------

st.sidebar.title("🤖 AI Personality")

st.sidebar.write("Choose your AI mode:")

selected_mode = st.sidebar.radio(
    "AI Mode",
    list(modes.keys()),
)


# ------------------ Handle Mode Change ------------------

if st.session_state.mode != selected_mode:

    st.session_state.mode = selected_mode

    st.session_state.messages = [
        SystemMessage(content=modes[selected_mode])
    ]


# ------------------ Main UI ------------------

st.title("🤖 AI Personality Chatbot")

st.caption(f"Current mode: **{selected_mode}**")


# ------------------ Display Chat History ------------------

for message in st.session_state.messages:

    if isinstance(message, SystemMessage):
        continue

    if isinstance(message, HumanMessage):

        with st.chat_message("user"):
            st.write(message.content)

    elif isinstance(message, AIMessage):

        with st.chat_message("assistant"):
            st.write(message.content)


# ------------------ Chat Input ------------------

prompt = st.chat_input("Type your message...")


if prompt:

    # Add user message
    human_message = HumanMessage(content=prompt)

    st.session_state.messages.append(human_message)

    # Display user message immediately
    with st.chat_message("user"):
        st.write(prompt)


    # Generate AI response
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            response = chat_model.invoke(
                st.session_state.messages
            )

            st.write(response.content)


    # Add AI response to conversation history
    st.session_state.messages.append(
        AIMessage(content=response.content)
    )

