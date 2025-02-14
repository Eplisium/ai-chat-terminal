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
        
        if not all([self.api_key, self.api_secret, self.access_token, self.access_token_secret]):
            raise ValueError("Missing X (Twitter) API credentials in environment variables")
        
        # Initialize API client
        auth = tweepy.OAuthHandler(self.api_key, self.api_secret)
        auth.set_access_token(self.access_token, self.access_token_secret)
        self.api = tweepy.API(auth)
        self.client = tweepy.Client(
            consumer_key=self.api_key,
            consumer_secret=self.api_secret,
            access_token=self.access_token,
            access_token_secret=self.access_token_secret
        )
    
    def post(self, text: str) -> Dict[str, Any]:
        """Post a tweet using X API"""
        try:
            # Create the tweet
            response = self.client.create_tweet(text=text)
            
            # Get the tweet ID from response
            tweet_id = response.data['id']
            
            return {
                "success": True,
                "tweet_id": tweet_id,
                "message": "Tweet posted successfully"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

def execute(arguments: Dict) -> str:
    """Post a tweet to X (Twitter).
    
    :param text: The text content of the tweet (max 280 characters)
    :return: Success message with tweet ID or error message
    """
    # Get parameters
    text = arguments.get('text', '')
    
    # Validate tweet text
    if not text:
        return "Error: Tweet text cannot be empty"
    
    if len(text) > 280:
        return f"Error: Tweet text exceeds 280 characters (current: {len(text)})"
    
    try:
        # Initialize poster
        poster = XPoster()
        
        # Post tweet
        result = poster.post(text)
        
        if result['success']:
            return f"Tweet posted successfully! Tweet ID: {result['tweet_id']}"
        else:
            return f"Error posting tweet: {result['error']}"
            
    except Exception as e:
        return f"Error: {str(e)}" 