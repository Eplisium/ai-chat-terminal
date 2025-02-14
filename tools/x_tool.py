import tweepy
from typing import Dict, Any
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class XPoster:
    def __init__(self):
        # Get credentials from environment variables
        self.api_key = os.getenv('X_API_KEY')
        self.api_secret = os.getenv('X_API_SECRET')
        self.access_token = os.getenv('X_ACCESS_TOKEN')
        self.access_token_secret = os.getenv('X_ACCESS_TOKEN_SECRET')
        
        # Validate credentials
        missing_creds = []
        if not self.api_key:
            missing_creds.append('X_API_KEY')
        if not self.api_secret:
            missing_creds.append('X_API_SECRET')
        if not self.access_token:
            missing_creds.append('X_ACCESS_TOKEN')
        if not self.access_token_secret:
            missing_creds.append('X_ACCESS_TOKEN_SECRET')
            
        if missing_creds:
            raise ValueError(f"Missing X (Twitter) API credentials in environment variables: {', '.join(missing_creds)}")
        
        try:
            # Initialize API v2 client (recommended for newer features)
            self.client = tweepy.Client(
                consumer_key=self.api_key,
                consumer_secret=self.api_secret,
                access_token=self.access_token,
                access_token_secret=self.access_token_secret,
                wait_on_rate_limit=True  # Automatically handle rate limits
            )
            
            # Test authentication
            self.client.get_me()
        except tweepy.errors.Unauthorized:
            raise ValueError("Invalid X (Twitter) API credentials. Please check your credentials and try again.")
        except Exception as e:
            raise ValueError(f"Failed to initialize X API client: {str(e)}")
    
    def post(self, text: str) -> Dict[str, Any]:
        """Post a tweet using X API v2
        
        Args:
            text (str): The text content of the tweet (max 280 characters)
            
        Returns:
            Dict[str, Any]: Response containing success status and tweet details
        """
        if not text:
            return {
                "success": False,
                "error": "Tweet text cannot be empty"
            }
        
        if len(text) > 280:
            return {
                "success": False,
                "error": f"Tweet text exceeds 280 characters (current: {len(text)})"
            }
        
        try:
            # Create the tweet using v2 endpoint
            response = self.client.create_tweet(text=text)
            
            if not response or not response.data:
                return {
                    "success": False,
                    "error": "Failed to create tweet - no response data"
                }
            
            # Get the tweet ID and other metadata
            tweet_data = response.data
            tweet_id = tweet_data['id']
            
            return {
                "success": True,
                "tweet_id": tweet_id,
                "text": text,
                "message": "Tweet posted successfully"
            }
            
        except tweepy.errors.TooManyRequests:
            return {
                "success": False,
                "error": "Rate limit exceeded. Please try again later."
            }
        except tweepy.errors.Forbidden as e:
            return {
                "success": False,
                "error": f"Forbidden: {str(e)}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error posting tweet: {str(e)}"
            }

def execute(arguments: Dict) -> str:
    """Post a tweet to X (Twitter).
    
    Args:
        arguments (Dict): Dictionary containing the tweet parameters
            - text (str): The text content of the tweet (max 280 characters)
            
    Returns:
        str: Success message with tweet ID or error message
    """
    # Get parameters
    text = arguments.get('text', '').strip()
    
    try:
        # Initialize poster
        poster = XPoster()
        
        # Post tweet
        result = poster.post(text)
        
        if result['success']:
            return f"Tweet posted successfully! Tweet ID: {result['tweet_id']}"
        else:
            return f"Error: {result['error']}"
            
    except ValueError as e:
        return f"Configuration Error: {str(e)}"
    except Exception as e:
        return f"Unexpected Error: {str(e)}" 