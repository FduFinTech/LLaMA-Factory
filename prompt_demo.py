from openai import OpenAI

key = 'YOUR_API_KEY'
client = OpenAI(api_key=key)

prompt = "As a Chinese financial expert, please combine China's national conditions and design some issues infinancial scenarios for the concept of securities investment funds."

input = "Question: What does securities investment fund mean? What is the difference and connection between it and other financial products?"

response = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {
      "role": "system",
      "content": prompt
    },
    {
      "role": "user",
      "content": input
    }
  ],
  temperature=0.7,
  max_tokens=150,
  top_p=1
)

# Accessing the message content from the response
output = response.choices[0].message.content.strip()
print(output)