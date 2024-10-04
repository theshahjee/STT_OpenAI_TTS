# openai_integration.py

from openai import OpenAI
from dotenv import load_dotenv
from prompts import Appointment

load_dotenv()
client = OpenAI()

# Function to interact with OpenAI's API using chat history
def response_openai(user_query: str, chat_history: list):
    # Get the doctor's basic prompt
    doctor_prompt = Appointment.basic_prompt()
    
    # Create the initial system message
    system_message = {"role": "system", "content": "You are a medical doctor providing consultations."}
    
    # Add the initial doctor's instructions prompt
    prompt_message = {"role": "system", "content": doctor_prompt}
    
    # Add user query to the conversation history
    user_message = {"role": "user", "content": f"User Query: {user_query}"}
    
    # Build the chat history by combining all messages
    messages = [system_message, prompt_message] + chat_history + [user_message]

    # Send the chat history along with the new query to OpenAI's GPT-3.5 API
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=400,
        temperature=0.7,
    )
    
    # Get the AI's response
    ai_response = response.choices[0].message.content.strip()

    # Update chat history with the latest response
    chat_history.append({"role": "assistant", "content": ai_response})

    # Return the AI's response
    return ai_response, chat_history
