from datetime import datetime
import pytz
import platform
import re
import os
from typing import List, Tuple

# Windows timezone name to pytz timezone mapping
WINDOWS_TO_PYTZ = {
    'Eastern Standard Time': 'America/New_York',
    'Eastern Daylight Time': 'America/New_York',
    'Central Standard Time': 'America/Chicago',
    'Central Daylight Time': 'America/Chicago',
    'Mountain Standard Time': 'America/Denver',
    'Mountain Daylight Time': 'America/Denver',
    'Pacific Standard Time': 'America/Los_Angeles',
    'Pacific Daylight Time': 'America/Los_Angeles',
    'Alaska Standard Time': 'America/Anchorage',
    'Alaska Daylight Time': 'America/Anchorage',
    'Hawaii-Aleutian Standard Time': 'Pacific/Honolulu',
    'Atlantic Standard Time': 'America/Halifax',
    'Atlantic Daylight Time': 'America/Halifax',
    'Newfoundland Standard Time': 'America/St_Johns',
    'Newfoundland Daylight Time': 'America/St_Johns',
    'GMT Standard Time': 'Europe/London',
    'British Summer Time': 'Europe/London',
    'W. Europe Standard Time': 'Europe/Berlin',
    'Central European Standard Time': 'Europe/Budapest',
    'Romance Standard Time': 'Europe/Paris',
    'Central Europe Standard Time': 'Europe/Prague',
    'E. Europe Standard Time': 'Europe/Bucharest',
    'FLE Standard Time': 'Europe/Kiev',
    'GTB Standard Time': 'Europe/Athens',
    'Russian Standard Time': 'Europe/Moscow',
    'India Standard Time': 'Asia/Kolkata',
    'China Standard Time': 'Asia/Shanghai',
    'Tokyo Standard Time': 'Asia/Tokyo',
    'Korea Standard Time': 'Asia/Seoul',
    'AUS Eastern Standard Time': 'Australia/Sydney',
    'AUS Central Standard Time': 'Australia/Adelaide',
    'AUS Western Standard Time': 'Australia/Perth',
    'New Zealand Standard Time': 'Pacific/Auckland',
    'UTC': 'UTC',
    'Coordinated Universal Time': 'UTC',
}

def get_system_timezone() -> str:
    """Get the system timezone as a pytz-compatible timezone name"""
    try:
        # First check for DEFAULT_TIMEZONE environment variable
        env_tz = os.getenv('DEFAULT_TIMEZONE')
        if env_tz and is_valid_timezone(env_tz):
            return env_tz
        
        # For Unix-like systems
        if platform.system() != 'Windows':
            try:
                with open('/etc/timezone', 'r') as f:
                    tz = f.read().strip()
                    if is_valid_timezone(tz):
                        return tz
            except:
                pass
        
        # For Windows systems - get the timezone name and convert it
        win_tz = datetime.now().astimezone().tzname()
        
        # Check if it's already a valid pytz timezone
        if is_valid_timezone(win_tz):
            return win_tz
        
        # Try to map Windows timezone name to pytz
        if win_tz in WINDOWS_TO_PYTZ:
            return WINDOWS_TO_PYTZ[win_tz]
        
        # Try to find a match by searching
        for win_name, pytz_name in WINDOWS_TO_PYTZ.items():
            if win_tz.lower() in win_name.lower() or win_name.lower() in win_tz.lower():
                return pytz_name
        
        # Fallback: try to detect from UTC offset
        local_offset = datetime.now().astimezone().utcoffset()
        if local_offset:
            offset_hours = local_offset.total_seconds() / 3600
            # Map common offsets to timezones
            offset_map = {
                -5: 'America/New_York',
                -6: 'America/Chicago',
                -7: 'America/Denver',
                -8: 'America/Los_Angeles',
                -4: 'America/New_York',  # EDT
                0: 'Europe/London',
                1: 'Europe/Paris',
                5.5: 'Asia/Kolkata',
                8: 'Asia/Shanghai',
                9: 'Asia/Tokyo',
            }
            if offset_hours in offset_map:
                return offset_map[offset_hours]
        
        return 'UTC'
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