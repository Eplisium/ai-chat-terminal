from datetime import datetime
import pytz
import platform
import re
from typing import List, Tuple

def get_system_timezone() -> str:
    """Get the system timezone"""
    try:
        # For Unix-like systems
        if platform.system() != 'Windows':
            with open('/etc/timezone', 'r') as f:
                return f.read().strip()
        # For Windows systems
        return datetime.now().astimezone().tzname()
    except:
        return 'UTC'

def get_available_timezones() -> List[str]:
    """Return a list of all available timezones"""
    return pytz.all_timezones

def is_valid_timezone(timezone: str) -> bool:
    """Check if the provided timezone is valid"""
    return timezone in pytz.all_timezones

def find_similar_timezones(timezone: str) -> List[str]:
    """Find similar timezone names for suggestions"""
    similar = []
    timezone_lower = timezone.lower()
    
    # Convert timezone string to a regex pattern
    pattern = timezone_lower.replace(' ', '.*')
    
    for tz in pytz.all_timezones:
        tz_lower = tz.lower()
        if (timezone_lower in tz_lower or  # Direct substring match
            re.search(pattern, tz_lower)):  # Pattern match
            similar.append(tz)
    
    return similar[:5]  # Return top 5 matches

def format_timezone_name(timezone: str) -> str:
    """Format timezone name to be more readable"""
    # Extract city name from timezone
    if '/' in timezone:
        city = timezone.split('/')[-1].replace('_', ' ')
        return city
    return timezone

def execute(arguments: dict) -> str:
    """Execute time tool
    
    Args:
        arguments (dict): Dictionary containing the parameters
            timezone (str, optional): Timezone name (e.g., 'America/New_York', 'Europe/London')
            format (str, optional): Output format ('natural' or 'technical')
            
    Returns:
        str: Formatted current time string or error message
    """
    # Get settings from arguments
    timezone = arguments.get('timezone')
    format_style = arguments.get('format', 'natural')
    
    # If no timezone provided, try to get from settings or use system timezone
    if not timezone:
        settings_manager = arguments.get('settings_manager')
        if settings_manager:
            timezone_settings = settings_manager.get_setting('timezone', {})
            timezone = timezone_settings.get('preferred_timezone')
            format_style = timezone_settings.get('format', format_style)
        
        if not timezone:
            timezone = get_system_timezone()
    
    if not is_valid_timezone(timezone):
        similar = find_similar_timezones(timezone)
        suggestion_msg = ""
        if similar:
            suggestion_msg = f"\nDid you mean one of these?\n" + "\n".join(f"- {tz}" for tz in similar)
        
        return (f"Error: Invalid timezone '{timezone}'.{suggestion_msg}\n"
                f"Please use a valid timezone name from the pytz database.\n"
                f"Example timezones: {', '.join(get_available_timezones()[:5])}...")
    
    try:
        tz = pytz.timezone(timezone)
        current_time = datetime.now(tz)
        
        if format_style == 'natural':
            city = format_timezone_name(timezone)
            return f"It's {current_time.strftime('%I:%M %p')} on {current_time.strftime('%A, %B %d, %Y')} in {city}"
        else:
            return f"Current time: {current_time.strftime('%I:%M %p')} on {current_time.strftime('%A, %B %d, %Y')} ({timezone})"
            
    except Exception as e:
        return f"Error: Unable to get current time - {str(e)}"