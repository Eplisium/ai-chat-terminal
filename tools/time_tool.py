from datetime import datetime
import pytz

def get_available_timezones():
    """Return a list of all available timezones"""
    return pytz.all_timezones

def is_valid_timezone(timezone):
    """Check if the provided timezone is valid"""
    return timezone in pytz.all_timezones

def execute(arguments):
    """Execute time tool
    
    Args:
        arguments (dict): Dictionary containing the timezone parameter
            timezone (str): Timezone name (e.g., 'America/New_York', 'Europe/London')
            
    Returns:
        str: Formatted current time string or error message
    """
    timezone = arguments.get('timezone', 'UTC')  # Default to UTC if no timezone specified
    
    if not is_valid_timezone(timezone):
        available = get_available_timezones()
        return (f"Error: Invalid timezone '{timezone}'. "
                f"Please use a valid timezone name from the pytz database.\n"
                f"Example timezones: {', '.join(available[:5])}...\n"
                f"Use get_available_timezones() to see all options.")
    
    try:
        tz = pytz.timezone(timezone)
        current_time = datetime.now(tz)
        return f"Current time: {current_time.strftime('%I:%M %p')} on {current_time.strftime('%A, %B %d, %Y')} ({timezone})"
    except Exception as e:
        return f"Error: {str(e)}" 