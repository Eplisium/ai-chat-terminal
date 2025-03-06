import os
import requests
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    
    # Handle US state codes specifically
    us_state_codes = {
        'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 
        'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware',
        'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho', 
        'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas',
        'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
        'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi',
        'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada',
        'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York',
        'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma',
        'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
        'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah',
        'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia',
        'WI': 'Wisconsin', 'WY': 'Wyoming', 'DC': 'District of Columbia'
    }
    
    # Split by commas or spaces
    parts = [part.strip() for part in location.replace(",", " ").split()]
    
    # Check for US city with state code format (e.g., "York PA")
    if len(parts) >= 2:
        potential_state_code = parts[1].upper()
        if potential_state_code in us_state_codes:
            # Add "USA" to ensure we get US locations
            if len(parts) == 2 or (len(parts) > 2 and parts[2].upper() != "USA"):
                parts.append("USA")
    
    # If we have 2-3 parts (city state [country]), join with commas
    if len(parts) >= 2:
        return ",".join(parts)
    
    return location

def format_timestamp(timestamp: int, timezone_offset: int = 0) -> str:
    """Format Unix timestamp to readable time, adjusting for timezone offset
    
    Args:
        timestamp: Unix timestamp in seconds
        timezone_offset: Timezone offset in seconds
        
    Returns:
        Formatted time string
    """
    dt = datetime.utcfromtimestamp(timestamp + timezone_offset)
    return dt.strftime("%H:%M:%S")

def format_wind_direction(degrees: float) -> str:
    """Convert wind direction in degrees to cardinal direction
    
    Args:
        degrees: Wind direction in degrees (meteorological)
        
    Returns:
        Cardinal direction (N, NE, E, SE, S, SW, W, NW)
    """
    directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", 
                  "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    index = round(degrees / 22.5) % 16
    return directions[index]

def find_best_location_match(location_query: str, location_data: List[Dict]) -> Optional[Dict]:
    """Find the best location match from geocoding results
    
    Args:
        location_query: Original location query string
        location_data: List of location data from geocoding API
        
    Returns:
        Best matching location or None if no match found
    """
    if not location_data:
        return None
    
    # Default to first result
    best_match = location_data[0]
    
    # Parse the query into parts for better matching
    query_parts = location_query.lower().replace(',', ' ').split()
    
    # Extract potential city, state, country from query
    query_city = query_parts[0] if query_parts else ""
    query_state = query_parts[1] if len(query_parts) > 1 else ""
    query_country = query_parts[2] if len(query_parts) > 2 else ""
    
    # US state codes to full names mapping
    us_state_codes = {
        'AL': 'alabama', 'AK': 'alaska', 'AZ': 'arizona', 'AR': 'arkansas', 
        'CA': 'california', 'CO': 'colorado', 'CT': 'connecticut', 'DE': 'delaware',
        'FL': 'florida', 'GA': 'georgia', 'HI': 'hawaii', 'ID': 'idaho', 
        'IL': 'illinois', 'IN': 'indiana', 'IA': 'iowa', 'KS': 'kansas',
        'KY': 'kentucky', 'LA': 'louisiana', 'ME': 'maine', 'MD': 'maryland',
        'MA': 'massachusetts', 'MI': 'michigan', 'MN': 'minnesota', 'MS': 'mississippi',
        'MO': 'missouri', 'MT': 'montana', 'NE': 'nebraska', 'NV': 'nevada',
        'NH': 'new hampshire', 'NJ': 'new jersey', 'NM': 'new mexico', 'NY': 'new york',
        'NC': 'north carolina', 'ND': 'north dakota', 'OH': 'ohio', 'OK': 'oklahoma',
        'OR': 'oregon', 'PA': 'pennsylvania', 'RI': 'rhode island', 'SC': 'south carolina',
        'SD': 'south dakota', 'TN': 'tennessee', 'TX': 'texas', 'UT': 'utah',
        'VT': 'vermont', 'VA': 'virginia', 'WA': 'washington', 'WV': 'west virginia',
        'WI': 'wisconsin', 'WY': 'wyoming', 'DC': 'district of columbia'
    }
    
    # If query has a state code, convert to full state name for matching
    if query_state.upper() in us_state_codes:
        query_state_full = us_state_codes[query_state.upper()]
    else:
        query_state_full = query_state
    
    # Score each location match
    best_score = 0
    for loc in location_data:
        score = 0
        loc_name = loc.get('name', '').lower()
        loc_state = loc.get('state', '').lower()
        loc_country = loc.get('country', '').lower()
        
        # Check city name match
        if query_city and query_city in loc_name:
            score += 3
            if query_city == loc_name:
                score += 2  # Exact city match
        
        # Check state match - both code and full name
        if query_state and (query_state == loc_state or query_state_full == loc_state):
            score += 5  # State match is important
        
        # Check country match
        if query_country and query_country == loc_country:
            score += 4
        elif query_country == "usa" and loc_country == "us":
            score += 4  # Special case for USA/US
        
        # For US locations, prioritize if state is provided
        if loc_country == "us" and loc_state and query_state:
            score += 3
        
        # If this location has a better score, use it
        if score > best_score:
            best_score = score
            best_match = loc
    
    # Log the matching process
    logger.info(f"Best location match for '{location_query}': {best_match.get('name')}, {best_match.get('state', '')}, {best_match.get('country')} (score: {best_score})")
    
    return best_match

def execute(arguments: Dict[str, Any]) -> str:
    """Execute weather tool to get current weather information for a location
    
    Args:
        arguments (dict): Dictionary containing:
            location (str, optional): Location to get weather for (e.g. "York PA USA" or "London UK").
                                     If not provided, will use DEFAULT_LOCATION from .env file.
            units (str): Units system ('metric' or 'imperial', default: 'imperial')
            detailed_view (bool): Whether to return a detailed report (default: True)
            forecast (bool): Whether to include forecast data (default: False)
            
    Returns:
        str: Formatted weather information or error message
    """
    # Check if location is provided, otherwise use default from .env
    if 'location' in arguments and arguments['location']:
        location = arguments['location']
    else:
        # Try to get default location from environment variable
        default_location = os.getenv('DEFAULT_LOCATION')
        if not default_location:
            return "Error: No location provided and no DEFAULT_LOCATION set in .env file. Please specify a location or set DEFAULT_LOCATION."
        location = default_location
        logger.info(f"Using default location from .env: {location}")
    
    units = arguments.get('units', 'imperial')  # Default to imperial (Fahrenheit)
    detailed_view = arguments.get('detailed_view', True)
    include_forecast = arguments.get('forecast', False)
    
    # Get API key from environment variable
    api_key = os.getenv('OPENWEATHERMAP_API_KEY')
    if not api_key:
        return "Error: OpenWeatherMap API key not found. Please set OPENWEATHERMAP_API_KEY environment variable."
    
    try:
        # Format location query
        formatted_location = format_location_query(location)
        logger.info(f"Formatted location query: '{location}' -> '{formatted_location}'")
        
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
            # Try a more generic search if specific search fails
            if "," in formatted_location:
                # Try with just the city name
                city_only = formatted_location.split(',')[0]
                logger.info(f"Retrying with city only: '{city_only}'")
                params['q'] = city_only
                response = requests.get(geocoding_url, params=params)
                response.raise_for_status()
                location_data = response.json()
        
        if not location_data:
            return f"Error: Location '{location}' not found. Please try a different location or format (e.g., 'City, State, Country')."
        
        # Log the geocoding results for debugging
        logger.info(f"Geocoding results for '{formatted_location}': {json.dumps(location_data[:2])}")
        
        # Find best match using our improved matching function
        best_match = find_best_location_match(formatted_location, location_data)
        if not best_match:
            return f"Error: Could not find a good match for location '{location}'. Please try a more specific query."
        
        lat = best_match['lat']
        lon = best_match['lon']
        
        # Get current weather data with more parameters
        weather_url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            'lat': lat,
            'lon': lon,
            'appid': api_key,
            'units': units
        }
        
        response = requests.get(weather_url, params=params)
        response.raise_for_status()
        
        weather_data = response.json()
        logger.info(f"Weather data received for {best_match.get('name')}, {best_match.get('state', '')}, {best_match.get('country')}")
        
        # Extract all relevant weather information
        temp = weather_data['main']['temp']
        feels_like = weather_data['main']['feels_like']
        temp_min = weather_data['main']['temp_min']
        temp_max = weather_data['main']['temp_max']
        humidity = weather_data['main']['humidity']
        pressure = weather_data['main']['pressure']
        
        # Wind information
        wind_speed = weather_data['wind']['speed']
        wind_deg = weather_data.get('wind', {}).get('deg')
        wind_direction = format_wind_direction(wind_deg) if wind_deg is not None else "N/A"
        wind_gust = weather_data.get('wind', {}).get('gust', "N/A")
        
        # Weather description
        weather_id = weather_data['weather'][0]['id']
        weather_main = weather_data['weather'][0]['main']
        description = weather_data['weather'][0]['description'].capitalize()
        weather_icon = weather_data['weather'][0]['icon']
        
        # Clouds, visibility, rain, snow
        clouds = weather_data.get('clouds', {}).get('all', 0)
        visibility = weather_data.get('visibility', 0)
        rain_1h = weather_data.get('rain', {}).get('1h', 0)
        snow_1h = weather_data.get('snow', {}).get('1h', 0)
        
        # Sunrise, sunset
        sunrise = weather_data['sys']['sunrise']
        sunset = weather_data['sys']['sunset']
        timezone_offset = weather_data.get('timezone', 0)  # Timezone shift in seconds from UTC
        
        # Get location name from weather data (more accurate than search)
        location_name = weather_data['name']
        if 'state' in best_match and best_match['state']:
            location_name += f", {best_match['state']}"
        location_name += f", {weather_data['sys']['country']}"
        
        # Format temperature strings
        temp_str = f"{temp:.1f}°{'C' if units == 'metric' else 'F'}"
        feels_like_str = f"{feels_like:.1f}°{'C' if units == 'metric' else 'F'}"
        temp_min_str = f"{temp_min:.1f}°{'C' if units == 'metric' else 'F'}"
        temp_max_str = f"{temp_max:.1f}°{'C' if units == 'metric' else 'F'}"
        
        # Format wind speed
        wind_speed_unit = 'm/s' if units == 'metric' else 'mph'
        
        # Format visibility (convert from meters to km or miles)
        if visibility > 0:
            if units == 'metric':
                visibility_str = f"{visibility / 1000:.1f} km"
            else:
                visibility_str = f"{visibility / 1609.34:.1f} miles"
        else:
            visibility_str = "N/A"
            
        # Format precipitation
        precipitation = []
        if rain_1h > 0:
            precipitation.append(f"Rain: {rain_1h} mm/h")
        if snow_1h > 0:
            precipitation.append(f"Snow: {snow_1h} mm/h")
        precipitation_str = ", ".join(precipitation) if precipitation else "None"
        
        # Format sunrise/sunset times
        sunrise_str = format_timestamp(sunrise, timezone_offset)
        sunset_str = format_timestamp(sunset, timezone_offset)
        
        # Get current time at location
        current_time = format_timestamp(weather_data['dt'], timezone_offset)
        
        # Try to get UV index and forecast data from One Call API if available
        uvi = "N/A"
        forecast_data = []
        
        try:
            onecall_url = "https://api.openweathermap.org/data/2.5/onecall"
            onecall_params = {
                'lat': lat,
                'lon': lon,
                'exclude': 'minutely,alerts' if include_forecast else 'minutely,hourly,daily,alerts',
                'appid': api_key,
                'units': units
            }
            onecall_response = requests.get(onecall_url, params=onecall_params)
            if onecall_response.status_code == 200:
                onecall_data = onecall_response.json()
                
                # Get UV index
                uvi = onecall_data.get('current', {}).get('uvi', "N/A")
                if isinstance(uvi, (int, float)):
                    uvi = f"{uvi:.1f}"
                
                # Get forecast data if requested
                if include_forecast:
                    # Get hourly forecast for next 24 hours (8 data points at 3-hour intervals)
                    hourly_data = onecall_data.get('hourly', [])
                    if hourly_data:
                        for i, hour_data in enumerate(hourly_data[:8]):
                            if i % 3 == 0:  # Get data every 3 hours
                                hour_time = datetime.utcfromtimestamp(hour_data['dt'] + timezone_offset)
                                hour_temp = hour_data['temp']
                                hour_desc = hour_data['weather'][0]['description'].capitalize()
                                hour_icon = hour_data['weather'][0]['icon']
                                hour_pop = hour_data.get('pop', 0) * 100  # Probability of precipitation
                                
                                forecast_data.append({
                                    'time': hour_time.strftime("%H:%M"),
                                    'day': hour_time.strftime("%a"),
                                    'temp': f"{hour_temp:.1f}°{'C' if units == 'metric' else 'F'}",
                                    'description': hour_desc,
                                    'icon': hour_icon,
                                    'pop': f"{hour_pop:.0f}%"
                                })
                    
                    # Get daily forecast for next 5 days
                    daily_data = onecall_data.get('daily', [])
                    if daily_data and len(forecast_data) < 5:
                        for i, day_data in enumerate(daily_data[1:6]):  # Skip today, get next 5 days
                            day_time = datetime.utcfromtimestamp(day_data['dt'] + timezone_offset)
                            day_temp_max = day_data['temp']['max']
                            day_temp_min = day_data['temp']['min']
                            day_desc = day_data['weather'][0]['description'].capitalize()
                            day_icon = day_data['weather'][0]['icon']
                            day_pop = day_data.get('pop', 0) * 100  # Probability of precipitation
                            
                            forecast_data.append({
                                'time': day_time.strftime("%d %b"),
                                'day': day_time.strftime("%a"),
                                'temp': f"{day_temp_max:.1f}°{'C' if units == 'metric' else 'F'} / {day_temp_min:.1f}°{'C' if units == 'metric' else 'F'}",
                                'description': day_desc,
                                'icon': day_icon,
                                'pop': f"{day_pop:.0f}%"
                            })
        except Exception as e:
            # If One Call API fails, continue without UV index and forecast
            logger.warning(f"Error fetching One Call API data: {str(e)}")
            pass
        
        # Build weather report based on detail level
        if detailed_view:
            # Detailed weather report with all available information
            weather_report = f"""Weather for {location_name} (as of {current_time})

🌡️ Temperature Information:
   Current: {temp_str}
   Feels like: {feels_like_str}
   Min/Max: {temp_min_str} / {temp_max_str}

🌤️ Weather Conditions:
   Description: {description} ({weather_main})
   Cloudiness: {clouds}%
   Visibility: {visibility_str}
   Precipitation: {precipitation_str}

💨 Wind Information:
   Speed: {wind_speed} {wind_speed_unit}
   Direction: {wind_direction} ({wind_deg}°)"""

            # Add wind gust if available
            if wind_gust != "N/A":
                weather_report += f"\n   Gusts: {wind_gust} {wind_speed_unit}"

            # Add additional atmospheric information
            weather_report += f"""

🌡️ Atmospheric Conditions:
   Humidity: {humidity}%
   Pressure: {pressure} hPa
   UV Index: {uvi}

🌅 Sun Information:
   Sunrise: {sunrise_str}
   Sunset: {sunset_str}

📍 Location Coordinates:
   Latitude: {lat}
   Longitude: {lon}

🔗 Weather Icon: https://openweathermap.org/img/wn/{weather_icon}@2x.png

🌐 Additional Information for AI:
   Weather ID: {weather_id} (OpenWeatherMap code)
   Country: {weather_data['sys']['country']}
   Timezone: UTC{'+' if timezone_offset > 0 else ''}{timezone_offset//3600} hours
   Data timestamp: {weather_data['dt']}
"""
        else:
            # Concise weather report with essential information
            weather_report = f"""Weather for {location_name} (as of {current_time})
🌡️ {temp_str}, feels like {feels_like_str}
🌤️ {description}
💨 Wind: {wind_speed} {wind_speed_unit} {wind_direction}
💧 Humidity: {humidity}%
☀️ UV Index: {uvi}
🌅 Sunrise: {sunrise_str}, Sunset: {sunset_str}
"""

        # Add forecast data if requested and available
        if include_forecast and forecast_data:
            weather_report += "\n📅 Forecast:\n"
            
            for i, forecast in enumerate(forecast_data):
                if i < 3:  # Show first 3 forecast periods
                    weather_report += f"   {forecast['day']} {forecast['time']}: {forecast['temp']} - {forecast['description']} (Rain: {forecast['pop']})\n"
            
            # Add link to full forecast
            weather_report += f"\n🔗 Full Forecast: https://openweathermap.org/city/{weather_data.get('id', '')}?units={units}\n"

        return weather_report

    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {str(e)}")
        return f"Error fetching weather data: {str(e)}"
    except (KeyError, IndexError) as e:
        logger.error(f"Data parsing error: {str(e)}")
        return f"Error parsing weather data: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return f"Unexpected error: {str(e)}" 