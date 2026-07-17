from openai import OpenAI

# 1. Point to your local FastAPI server
# This mimics exactly what LibreChat is doing
client = OpenAI(
    api_key="sk-fake-key-not-needed", 
    base_url="http://localhost:8000/v1" 
)

print("Sending request to your local proxy...")

try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "Say 'The proxy is working!'"}
        ]
    )
    print("\n✅ SUCCESS! The proxy is working.")
    print("Response:", response.choices[0].message.content)
except Exception as e:
    print("\n❌ FAILURE! The proxy is not responding correctly.")
    print("Error:", str(e))