import os
os.environ["OPENAI_API_KEY"] = "sk-uNG4fMQjLQhyq8pRBaD8D7115bA14b21B94e0d2a150dDbB4"
os.environ["OPENAI_BASE_URL"] = "https://gtapi.xiaoerchaoren.com:8932/v1"


from openai import OpenAI

client = OpenAI()

completion = client.chat.completions.create(
  model="gpt-4",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ]
)

print(completion.choices[0].message.content)