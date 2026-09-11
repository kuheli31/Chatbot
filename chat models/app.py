
import streamlit as st

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


st.set_page_config(
    page_title="AI Personality Chatbot",
    page_icon="🤖",
)


# --------------------------------------------------
# Load TinyLlama only once
# --------------------------------------------------

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


with st.spinner("Loading TinyLlama..."):
    chat_model = load_model()


# --------------------------------------------------
# Personality modes
# --------------------------------------------------

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


# --------------------------------------------------
# Session state
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "mode" not in st.session_state:
    st.session_state.mode = None


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("🤖 AI Personality")
st.sidebar.write("Choose your AI mode:")

selected_mode = st.sidebar.radio(
    "AI Mode",
    list(modes.keys())
)


# Clear conversation when mode changes
if st.session_state.mode != selected_mode:
    st.session_state.mode = selected_mode
    st.session_state.messages = []


# --------------------------------------------------
# Main UI
# --------------------------------------------------

st.title("🤖 AI Personality Chatbot")
st.caption(f"Current mode: **{selected_mode}**")


# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])


# --------------------------------------------------
# Chat input
# --------------------------------------------------

prompt = st.chat_input("Type your message...")


if prompt:

    # Add user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):
        st.write(prompt)


    # Build LangChain messages
    messages = [
        SystemMessage(content=modes[selected_mode])
    ]

    for message in st.session_state.messages:

        if message["role"] == "user":
            messages.append(
                HumanMessage(content=message["content"])
            )

        elif message["role"] == "assistant":
            messages.append(
                AIMessage(content=message["content"])
            )


    # Generate response
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            response = chat_model.invoke(messages)

            answer = response.content

            st.write(answer)


    # Save assistant response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )
