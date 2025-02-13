from datetime import datetime
import pytz

def execute(arguments):
    """Execute time tool"""
    timezone = arguments.get('timezone', 'America/New_York')
    try:
        tz = pytz.timezone(timezone)
        current_time = datetime.now(tz)
        return f"Current time: {current_time.strftime('%I:%M %p')} on {current_time.strftime('%A, %B %d, %Y')} ({timezone})"
    except Exception as e:
        return f"Error: {str(e)}" 