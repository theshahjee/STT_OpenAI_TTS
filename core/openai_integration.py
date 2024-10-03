# openai_integration.py

from openai import OpenAI
from dotenv import load_dotenv
from prompts import Appointment

load_dotenv()
client = OpenAI()

# Function to interact with OpenAI's API
def test_openai(user_query: str):
    # Get the doctor's basic prompt
    doctor_prompt = Appointment.basic_prompt()
    
    # Combine the doctor's prompt and the user's query
    combined_prompt = f"{doctor_prompt}\n\nUser Query: {user_query}"
    
    # Send the combined prompt to OpenAI's GPT-3.5 API
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a medical doctor providing consultations."},
            {"role": "user", "content": combined_prompt}
        ],
        max_tokens=400,
        temperature=0.7,
    )
    
    # Return the AI's response
    return response.choices[0].message.content.strip()
