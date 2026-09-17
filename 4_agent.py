from langchain_groq import ChatGroq
from langchain_core.tools import tool
import requests
from langchain_core.prompts import PromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_agent

from dotenv import load_dotenv
import os 
os.environ['LANGCHAIN_PROJECT'] = 'ReAct Agent Demo'

load_dotenv()

search_tool = DuckDuckGoSearchRun()

@tool
def get_weather_data(city: str) -> str:
    """
    This function fetches the current weather data for a given city
    """
    api_key=os.getenv('WEATHERSTACK_API_KEY')

    url = (
        f"https://api.weatherstack.com/current"
        f"?access_key={api_key}&query={city}"
    )
    response = requests.get(url)

    if response.status_code !=200:
        return f"Weatherstack api request failed :{response.status_code}"

    data = response.json()

    if "error" in data:
        return f"Weather API error: {data['error']}"

    current = data.get("current", {})

    return (
        f"City: {city}\n"
        f"Temperature: {current.get('temperature')}°C\n"
        f"Weather: {current.get('weather_descriptions')}\n"
        f"Humidity: {current.get('humidity')}%\n"
        f"Wind Speed: {current.get('wind_speed')} km/h"
    )


llm = ChatGroq(model='openai/gpt-oss-20b',temperature=0.7)


agent = create_agent(
    model=llm,
    tools=[search_tool, get_weather_data],
    system_prompt=(
        "You are a helpful assistant. "
        "Use the available tools whenever they are needed. "
        "Use the weather tool for current weather information "
        "and the search tool for web searches."
    )
)


# What is the release date of Dhadak 2?
# What is the current temp of gurgaon
# Identify the birthplace city of Kalpana Chawla (search) and give its current temperature.

# Step 5: Invoke Agent
response = agent.invoke( {
        "messages": [
            {
                "role": "user",
                "content": "What is the current temperature of Mumbai?"
            }
        ]})

print(response)
print("\nFinal Answer:")
for message in response["messages"]:
    if hasattr(message, "content") and message.content:
        print(message.content)