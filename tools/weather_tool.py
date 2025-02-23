import os
import requests
from datetime import datetime
from typing import Dict, Any

def kelvin_to_celsius(kelvin: float) -> float:
    """Convert Kelvin to Celsius"""
    return kelvin - 273.15

def kelvin_to_fahrenheit(kelvin: float) -> float:
    """Convert Kelvin to Fahrenheit"""
    return (kelvin - 273.15) * 9/5 + 32

def format_temperature(temp: float, units: str) -> str:
    """Format temperature with proper unit"""
    if units == "metric":
        return f"{kelvin_to_celsius(temp):.1f}°C"
    else:
        return f"{kelvin_to_fahrenheit(temp):.1f}°F"

def format_location_query(location: str) -> str:
    """Format location query for better geocoding results
    
    Handles formats like:
    - city state country
    - city, state, country
    - city state
    - city, state
    """
    # Replace multiple spaces with single space
    location = " ".join(location.split())
    
    # Split by commas or spaces
    parts = [part.strip() for part in location.replace(",", " ").split()]
    
    # If we have 2-3 parts (city state [country]), join with commas
    if len(parts) >= 2:
        return ",".join(parts)
    
    return location

def execute(arguments: Dict[str, Any]) -> str:
    """Execute weather tool to get current weather information for a location
    
    Args:
        arguments (dict): Dictionary containing:
            location (str): Location to get weather for (e.g. "York PA USA" or "London UK")
            units (str): Units system ('metric' or 'imperial')
            
    Returns:
        str: Formatted weather information or error message
    """
    location = arguments['location']
    units = arguments.get('units', 'metric')
    
    # Get API key from environment variable
    api_key = os.getenv('OPENWEATHERMAP_API_KEY')
    if not api_key:
        return "Error: OpenWeatherMap API key not found. Please set OPENWEATHERMAP_API_KEY environment variable."
    
    try:
        # Format location query
        formatted_location = format_location_query(location)
        
        # First get coordinates for the location
        geocoding_url = f"http://api.openweathermap.org/geo/1.0/direct"
        params = {
            'q': formatted_location,
            'limit': 5,  # Get more results to find best match
            'appid': api_key
        }
        
        response = requests.get(geocoding_url, params=params)
        response.raise_for_status()
        
        location_data = response.json()
        if not location_data:
            return f"Error: Location '{location}' not found"
        
        # Try to find best match based on state/country if provided
        location_parts = formatted_location.lower().split(',')
        best_match = location_data[0]  # Default to first result
        
        if len(location_parts) > 1:
            # If state/country provided, try to match them
            for loc in location_data:
                state = loc.get('state', '').lower()
                country = loc.get('country', '').lower()
                
                # Check if state or country matches what was provided
                if (state and state in location_parts) or (country and country in location_parts):
                    best_match = loc
                    break
        
        lat = best_match['lat']
        lon = best_match['lon']
        
        # Get current weather data
        weather_url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            'lat': lat,
            'lon': lon,
            'appid': api_key,
            'units': units  # Use requested units directly
        }
        
        response = requests.get(weather_url, params=params)
        response.raise_for_status()
        
        weather_data = response.json()
        
        # Format the response
        temp = weather_data['main']['temp']
        feels_like = weather_data['main']['feels_like']
        humidity = weather_data['main']['humidity']
        wind_speed = weather_data['wind']['speed']
        description = weather_data['weather'][0]['description'].capitalize()
        
        # Get location name from weather data (more accurate than search)
        location_name = weather_data['name']
        if 'state' in best_match:
            location_name += f", {best_match['state']}"
        location_name += f", {weather_data['sys']['country']}"
        
        # Convert temperature if needed (API returns in requested units)
        temp_str = f"{temp:.1f}°{'C' if units == 'metric' else 'F'}"
        feels_like_str = f"{feels_like:.1f}°{'C' if units == 'metric' else 'F'}"
        
        return f"""Weather for {location_name}
Temperature: {temp_str}
Feels like: {feels_like_str}
Conditions: {description}
Humidity: {humidity}%
Wind speed: {wind_speed} {'m/s' if units == 'metric' else 'mph'}"""

    except requests.exceptions.RequestException as e:
        return f"Error fetching weather data: {str(e)}"
    except (KeyError, IndexError) as e:
        return f"Error parsing weather data: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}" 