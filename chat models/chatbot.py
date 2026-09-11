from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


# ------------------ Load Local LLM ------------------

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 64,
        "do_sample": False,
        "repetition_penalty": 1.03,
    },
)

chat_model = ChatHuggingFace(llm=llm)


# ------------------ Choose AI Mode ------------------

print("\nChoose your AI mode")
print("Press 1 for Angry mode")
print("Press 2 for Funny mode")
print("Press 3 for Sad mode")

while True:
    try:
        choice = int(input("Tell your response: "))

        if choice in (1, 2, 3):
            break

        print("Please enter 1, 2, or 3.")

    except KeyboardInterrupt:
        print("\nExiting chatbot.")
        raise SystemExit(0)

    except EOFError:
        print("\nNo input received. Exiting chatbot.")
        raise SystemExit(0)

    except ValueError:
        print("Please enter a valid number.")


# ------------------ Set Personality ------------------

if choice == 1:
    mode = (
        "You are an angry AI agent. "
        "You respond aggressively and impatiently, "
        "but remain helpful and do not insult the user."
    )

elif choice == 2:
    mode = (
        "You are a very funny AI agent. "
        "You respond with humor, jokes, and a playful personality "
        "while still being helpful."
    )

else:
    mode = (
        "You are a very sad AI agent. "
        "You respond in a melancholic and emotional tone "
        "while still being helpful."
    )


# ------------------ Conversation History ------------------

messages = [
    SystemMessage(content=mode)
]


print("\n----------------- Welcome -----------------")
print("Type 0 to exit the application.\n")


# ------------------ Chat Loop ------------------

while True:

    try:
        prompt = input("You : ")

    except KeyboardInterrupt:
        print("\nExiting chatbot.")
        break

    except EOFError:
        print("\nInput stream closed. Exiting chatbot.")
        break

    if prompt.strip() == "0":
        print("Goodbye!")
        break

    if not prompt.strip():
        print("Please enter a message.")
        continue

    # Add user message
    messages.append(HumanMessage(content=prompt))

    # Get AI response
    response = chat_model.invoke(messages)

    # Add AI response to conversation history
    messages.append(AIMessage(content=response.content))

    print("Bot :", response.content)
    print()


# ------------------ Debug: Conversation History ------------------

print("\nConversation history:")
for message in messages:
    print(f"{message.type}: {message.content}")

