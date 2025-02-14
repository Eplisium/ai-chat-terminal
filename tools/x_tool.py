import tweepy
from typing import Dict, Any, List, Optional
import os
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

class XClient:
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
            self.me = self.client.get_me()
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
    
    def get_user_info(self, username: str) -> Dict[str, Any]:
        """Get information about a user by their username"""
        try:
            # Get user info with additional fields
            response = self.client.get_user(
                username=username,
                user_fields=['created_at', 'description', 'public_metrics', 'verified']
            )
            
            if not response or not response.data:
                return {"success": False, "error": f"User @{username} not found"}
            
            user_data = response.data
            
            return {
                "success": True,
                "user": {
                    "id": user_data.id,
                    "username": user_data.username,
                    "name": user_data.name,
                    "description": user_data.description,
                    "created_at": user_data.created_at,
                    "verified": user_data.verified,
                    "metrics": user_data.public_metrics
                }
            }
        except Exception as e:
            return {"success": False, "error": f"Error getting user info: {str(e)}"}
    
    def search_tweets(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """Search for tweets matching a query"""
        try:
            # Search tweets with additional fields
            response = self.client.search_recent_tweets(
                query=query,
                max_results=min(max_results, 100),
                tweet_fields=['created_at', 'public_metrics', 'author_id'],
                expansions=['author_id']
            )
            
            if not response or not response.data:
                return {"success": False, "error": f"No tweets found for query: {query}"}
            
            # Process tweets and include user information
            tweets = []
            users = {user.id: user for user in response.includes['users']} if response.includes else {}
            
            for tweet in response.data:
                author = users.get(tweet.author_id, {})
                tweets.append({
                    "id": tweet.id,
                    "text": tweet.text,
                    "created_at": tweet.created_at,
                    "metrics": tweet.public_metrics,
                    "author": {
                        "id": author.id,
                        "username": author.username,
                        "name": author.name
                    } if author else None
                })
            
            return {
                "success": True,
                "tweets": tweets,
                "result_count": len(tweets)
            }
        except Exception as e:
            return {"success": False, "error": f"Error searching tweets: {str(e)}"}
    
    def get_tweet(self, tweet_id: str) -> Dict[str, Any]:
        """Get a specific tweet by ID"""
        try:
            response = self.client.get_tweet(
                tweet_id,
                tweet_fields=['created_at', 'public_metrics', 'author_id'],
                expansions=['author_id']
            )
            
            if not response or not response.data:
                return {"success": False, "error": f"Tweet with ID {tweet_id} not found"}
            
            tweet = response.data
            author = response.includes['users'][0] if response.includes else None
            
            return {
                "success": True,
                "tweet": {
                    "id": tweet.id,
                    "text": tweet.text,
                    "created_at": tweet.created_at,
                    "metrics": tweet.public_metrics,
                    "author": {
                        "id": author.id,
                        "username": author.username,
                        "name": author.name
                    } if author else None
                }
            }
        except Exception as e:
            return {"success": False, "error": f"Error getting tweet: {str(e)}"}

def execute(arguments: Dict) -> str:
    """Execute X API operations.
    
    Args:
        arguments (Dict): Dictionary containing operation parameters
            - operation (str): The operation to perform (post, get_user, search, get_tweet)
            - text (str): The text content for posting (max 280 characters)
            - username (str): Username for user lookup
            - query (str): Search query for tweets
            - tweet_id (str): Tweet ID to retrieve
            - max_results (int): Maximum number of search results (default: 10)
    
    Returns:
        str: Operation result or error message
    """
    operation = arguments.get('operation', 'post')
    
    try:
        client = XClient()
        
        if operation == 'post':
            text = arguments.get('text', '').strip()
            result = client.post(text)
        elif operation == 'get_user':
            username = arguments.get('username', '').strip()
            if not username:
                return "Error: Username is required for user lookup"
            result = client.get_user_info(username)
        elif operation == 'search':
            query = arguments.get('query', '').strip()
            max_results = int(arguments.get('max_results', 10))
            if not query:
                return "Error: Search query is required"
            result = client.search_tweets(query, max_results)
        elif operation == 'get_tweet':
            tweet_id = arguments.get('tweet_id', '').strip()
            if not tweet_id:
                return "Error: Tweet ID is required"
            result = client.get_tweet(tweet_id)
        else:
            return f"Error: Unknown operation '{operation}'"
        
        if result['success']:
            if operation == 'post':
                return f"Tweet posted successfully! Tweet ID: {result['tweet_id']}"
            elif operation == 'get_user':
                user = result['user']
                return f"User Info for @{user['username']}:\n" + \
                       f"Name: {user['name']}\n" + \
                       f"Description: {user['description']}\n" + \
                       f"Created: {user['created_at']}\n" + \
                       f"Verified: {user['verified']}\n" + \
                       f"Metrics: {user['metrics']}"
            elif operation == 'search':
                tweets = result['tweets']
                return f"Found {len(tweets)} tweets:\n\n" + \
                       "\n\n".join([f"@{t['author']['username']}: {t['text']}" for t in tweets])
            elif operation == 'get_tweet':
                tweet = result['tweet']
                return f"Tweet by @{tweet['author']['username']}:\n" + \
                       f"{tweet['text']}\n" + \
                       f"Created: {tweet['created_at']}\n" + \
                       f"Metrics: {tweet['metrics']}"
        else:
            return f"Error: {result['error']}"
            
    except ValueError as e:
        return f"Configuration Error: {str(e)}"
    except Exception as e:
        return f"Unexpected Error: {str(e)}" 