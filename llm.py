from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access the OpenAI API key from the loaded environment variables
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_response(messages):
    """Generate assistant's response using OpenAI GPT-3."""
    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages, stream=True)

    # Stream and concatenate the response text
    full_response = ''
    for chunk in response:
        # Ensure content exists and is not None
        if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content is not None:
            full_response += chunk.choices[0].delta.content

    return full_response
