from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import base64

# Load the image and convert it to base64
with open('./media/people.jpeg', 'rb') as image_file:
    image_bytes = image_file.read()
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')


llm = ChatOpenAI(
    model="mistralai/mistral-small-3.1-24b-instruct:free",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key="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",  # if you prefer to pass api key in directly instaed of using env vars
    base_url="https://openrouter.ai/api/v1",
)

messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                }
            ]
        }
    ]

ai_msg = llm.invoke(messages)
print(ai_msg)