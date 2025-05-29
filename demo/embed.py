import openai

client = openai.OpenAI(
    base_url="http://localhost:8111/v1",
    api_key="<KEY>",
)

response = client.embeddings.create(
    model="multilingual-e5-large-instruct",
    input="Academia.edu uses",
    encoding_format="float",
)

print(response.data[0].embedding)
print(len(response.data[0].embedding))
