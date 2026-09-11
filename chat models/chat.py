from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0.7,
    timeout=None,
    max_retries=2,
)

response = model.invoke("What's mastoidectomy?")

print(response.content)