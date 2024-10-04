from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

def test_openai(prompt: str):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert content writer and generate the required content."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=400,
        temperature=0.7,
    )
    
    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    prompt = "Write a creative story about a magical kingdom facing a drought."
    result = test_openai(prompt)
    print("Test Output:", result)