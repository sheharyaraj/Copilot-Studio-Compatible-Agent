import asyncio
import json
import os
import re
import requests
from typing import Annotated
from pydantic import Field
from dotenv import load_dotenv
from agent_framework import ChatAgent, HostedMCPTool
from agent_framework.openai import OpenAIChatClient

# Load environment variables
load_dotenv()

# Weather function using OpenWeatherMap API
def get_weather(
    location: Annotated[str, Field(description="The location to get the weather for (e.g., 'London', 'New York').")]
) -> str:
    """Get real weather data for a given location using OpenWeatherMap API."""
    try:
        api_key = os.getenv("OPENWEATHER_API_KEY")
        if not api_key:
            return "Weather API key not configured. Please set OPENWEATHER_API_KEY in .env file."
        
        # First get coordinates for the location using Geocoding API
        geocoding_url = "https://api.openweathermap.org/geo/1.0/direct"
        geocoding_params = {
            'q': location,
            'limit': 1,
            'appid': api_key
        }

        # Avoid printing secrets (API keys) in logs.
        print(f"Geocoding request: {geocoding_url}?q={location}&limit=1&appid=***")
        geo_response = requests.get(geocoding_url, params=geocoding_params, timeout=10)
        geo_response.raise_for_status()
        geo_data = geo_response.json()
        
        if not geo_data:
            return f"Could not find coordinates for '{location}'. Please check the location name."
        
        lat = geo_data[0]['lat']
        lon = geo_data[0]['lon']
        
        # Get weather data using Current Weather Data API (more widely available than One Call 3.0)
        # Docs: https://openweathermap.org/current
        weather_url = "https://api.openweathermap.org/data/2.5/weather"
        weather_params = {
            'lat': lat,
            'lon': lon,
            'appid': api_key,
            'units': 'metric'  # Use Celsius
        }

        # Construct full URL for debugging without leaking secrets
        full_weather_url = f"{weather_url}?lat={lat}&lon={lon}&appid=***&units=metric"
        print(f"Weather request: {full_weather_url}")
        
        weather_response = requests.get(weather_url, params=weather_params, timeout=10)
        weather_response.raise_for_status()
        weather_data = weather_response.json()

        # Build a response that includes the *full* raw API payload plus a short summary.
        # Returning the raw JSON here ensures the agent can pass it through verbatim.
        raw_payload = {
            "geocoding": geo_data[0],
            "weather": weather_data,
        }
        
        # Extract weather information
        temp = weather_data['main']['temp']
        humidity = weather_data['main']['humidity']
        weather_desc = weather_data['weather'][0]['description'].title()

        summary = (
            f"The weather in {location.title()} is {weather_desc} with a temperature of {temp}°C "
            f"and humidity of {humidity}%."
        )

        return (
            "FULL_OPENWEATHERMAP_API_RESPONSE (JSON):\n"
            f"```json\n{json.dumps(raw_payload, ensure_ascii=False, indent=2)}\n```\n\n"
            "SUMMARY:\n"
            f"{summary}"
        )

    except requests.exceptions.HTTPError as e:
        status = getattr(e.response, "status_code", None)
        # Try to surface OpenWeather's error message if present.
        details = None
        try:
            if e.response is not None:
                payload = e.response.json()
                details = payload.get("message") if isinstance(payload, dict) else None
        except Exception:
            details = None

        msg = f"OpenWeatherMap API error{f' ({status})' if status else ''}."
        if details:
            msg += f" Details: {details}"
        return msg

    except requests.exceptions.RequestException as e:
        return f"Network/connection error fetching weather data: {str(e)}"
    except KeyError as e:
        return f"Error parsing weather data: Missing field {str(e)}. This might be due to API limitations."
    except Exception as e:
        return f"Unexpected error getting weather: {str(e)}"

class OpenAIAgent:
    def __init__(self):
        """Initialize the OpenAI Agent with configuration from environment variables."""
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4")
        self.agent_name = os.getenv("AGENT_NAME", "OpenAI-Agent")
        self.agent_description = os.getenv("AGENT_DESCRIPTION", "An AI agent powered by OpenAI that can answer user queries and use various tools")
        self.mcp_server_url = os.getenv("MCP_SERVER_URL")
        self.mcp_server_timeout = int(os.getenv("MCP_SERVER_TIMEOUT", "30"))
        
        # Initialize OpenAI client
        self.chat_client = OpenAIChatClient(
            model_id=self.openai_model,
            api_key=self.openai_api_key
        )
        
        # Initialize agent with tools
        self.agent = ChatAgent(
            name=self.agent_name,
            chat_client=self.chat_client,
            instructions=f"""You are {self.agent_name}. {self.agent_description}
    
    You have access to the following tools:
    1. Weather - Get real weather information using OpenWeatherMap API
    2. Microsoft Learn MCP - Access Microsoft documentation and learning resources
    
    IMPORTANT INSTRUCTIONS:
    - When providing weather information, ALWAYS call the Weather tool.
    - Then output the Weather tool result *verbatim* (do not rewrite/reformat the JSON).
    - After the verbatim tool output, add a short human-readable summary.
    
    Use these tools when appropriate to provide comprehensive and accurate answers.
    Always be helpful, accurate, and provide detailed responses with specific numbers and facts.""",
            tools=[
                get_weather,
                HostedMCPTool(
                    name="Microsoft Learn MCP",
                    url="https://learn.microsoft.com/api/mcp"
                )
            ]
        )
    
    async def run_query(self, query: str) -> str:
        """
        Run a query through the agent and return the response.
        
        Args:
            query (str): The user's query
            
        Returns:
            str: The agent's response
        """
        try:
            # Hard guarantee for weather: return the full tool output verbatim.
            # Some agent/LLM stacks may paraphrase or omit parts of a tool response;
            # this bypass ensures callers always get the raw API payload.
            if self._looks_like_weather_query(query):
                location = self._extract_location_from_weather_query(query)
                return get_weather(location)

            result = await self.agent.run(query)
            return result
        except Exception as e:
            return f"Error processing query: {str(e)}"

    @staticmethod
    def _looks_like_weather_query(query: str) -> bool:
        return bool(re.search(r"\bweather\b", query, re.IGNORECASE))

    @staticmethod
    def _extract_location_from_weather_query(query: str) -> str:
        """Best-effort location extractor.

        The caller often sends queries like:
        - "weather in Faisalabad"
        - "weather in Faisalabad, including temperature, humidity, wind speed..."

        We only want the actual place name for geocoding, not the rest of the request.
        """

        # Common patterns: "weather in Karachi", "weather for London".
        m = re.search(r"\bweather\b.*?\b(?:in|for)\s+([^?\n\r]+)", query, re.IGNORECASE)
        location = m.group(1).strip() if m else query.strip()

        # Truncate on separators / common continuation phrases.
        # e.g. "Faisalabad, including temperature..." -> "Faisalabad"
        location = re.split(
            r"\s*(?:,|;|\.|\(|\)|\bincluding\b|\bwith\b|\bshow\b|\bgive\b|\band\b)\s*",
            location,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]

        return location.strip().strip("?").strip()
    
    async def start_interactive_mode(self):
        """Start an interactive chat session with the agent."""
        print(f"🤖 {self.agent_name} is ready!")
        print(f"Description: {self.agent_description}")
        print("Available tools: Weather (OpenWeatherMap), Microsoft Learn MCP")
        print("Type 'quit' or 'exit' to end the conversation.\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye! 👋")
                    break
                
                if not user_input:
                    continue
                
                print("🤔 Thinking...")
                response = await self.run_query(user_input)
                print(f"\n{self.agent_name}: {response}\n")
                
            except KeyboardInterrupt:
                print("\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"Error: {str(e)}")

async def main():
    """Main function to run the agent."""
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY is not set in the .env file")
        print("Please set your OpenAI API key in the .env file and try again.")
        return
    
    # Create and run the agent
    agent = OpenAIAgent()
    
    # You can either run a single query or start interactive mode
    import sys
    
    if len(sys.argv) > 1:
        # Run single query from command line
        query = " ".join(sys.argv[1:])
        print(f"Query: {query}")
        response = await agent.run_query(query)
        print(f"Response: {response}")
    else:
        # Start interactive mode by default
        await agent.start_interactive_mode()

if __name__ == "__main__":
    asyncio.run(main())
