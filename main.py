import os
import sys
import json
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
import dotenv
import rich
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.logging import RichHandler
import inquirer
from inquirer import themes
import requests
from collections import defaultdict
from rich.syntax import Syntax

# Conditionally import based on provider
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import openai
    OPENROUTER_AVAILABLE = True
except ImportError:
    OPENROUTER_AVAILABLE = False

# Add HTTP headers helper
def get_openrouter_headers(api_key):
    """Get required headers for OpenRouter API"""
    return {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/cursor-ai",  # Replace with your actual site
        "X-Title": "AI Chat Script",  # Replace with your app name
        "Content-Type": "application/json"
    }

def setup_logging():
    """
    Set up logging configuration with Rich logging handler
    
    Returns:
        tuple: Logger and Rich Console object
    """
    console = Console()
    logs_dir = 'logs'
    os.makedirs(logs_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                rich_tracebacks=True, 
                console=console, 
                show_path=False,
                omit_repeated_times=False
            ),
            RotatingFileHandler(
                os.path.join(logs_dir, f'ai_chat_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                maxBytes=10*1024*1024,
                backupCount=5,
                encoding='utf-8'
            )
        ]
    )

    return logging.getLogger("AIChat"), console

class OpenRouterAPI:
    """Class to handle OpenRouter API interactions"""
    def __init__(self, api_key):
        self.api_key = api_key
        self.api_base = "https://openrouter.ai/api/v1"
        self.headers = get_openrouter_headers(api_key)
    
    def fetch_models(self):
        """Fetch available models from OpenRouter API"""
        try:
            self.headers["Content-Type"] = "application/json"  # Ensure content type is set
            response = requests.get(
                f"{self.api_base}/models",
                headers=self.headers
            )
            
            # Debug response
            if response.status_code != 200:
                raise Exception(f"API returned status code {response.status_code}: {response.text}")
            
            data = response.json()
            if not data or 'data' not in data:
                raise Exception(f"Unexpected API response format: {data}")
            
            # The models are in the 'data' field of the response
            models = data['data']
            if not models:
                raise Exception("No models returned from API")
            
            # Transform all models without filtering
            transformed_models = []
            for model in models:
                transformed_models.append({
                    'id': model['id'],
                    'name': model.get('name', model['id']),
                    'description': model.get('description', 'No description available'),
                    'context_length': model.get('context_length', 'Unknown'),
                    'pricing': model.get('pricing', {}),
                    'top_provider': model.get('top_provider', False),
                    'family': model.get('family', 'Unknown'),
                    'created_at': model.get('created_at', None)
                })
            
            return transformed_models
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Network error while fetching models: {str(e)}")
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON response from API: {str(e)}")
        except Exception as e:
            raise Exception(f"Failed to fetch OpenRouter models: {str(e)}")
    
    def group_models_by_company(self, models):
        """Group models by their company"""
        company_models = defaultdict(list)
        
        for model in models:
            # Extract company name from model ID (e.g., 'anthropic/claude-3' -> 'Anthropic')
            model_id = model['id']
            if '/' in model_id:
                company = model_id.split('/')[0].title()
            else:
                company = 'Other'  # Fallback category
            company_models[company].append(model)
        
        # Sort models within each company by name and then by creation date
        for company in company_models:
            company_models[company].sort(
                key=lambda x: (
                    not x.get('top_provider', False),  # Top providers first
                    x['name'].lower(),  # Then alphabetically
                    x.get('created_at', '0')  # Then by creation date
                )
            )
        
        return dict(company_models)

class AIChat:
    def __init__(self, model_config, logger, console):
        """
        Initialize AI Chat with specific model configuration
        
        Args:
            model_config (dict): Configuration for the selected model
            logger (logging.Logger): Logging instance
            console (rich.console.Console): Rich console for output
        """
        self.logger = logger
        self.console = console
        dotenv.load_dotenv()

        self.model_id = model_config['id']
        self.model_name = model_config['name']
        self.provider = model_config.get('provider', 'openai')
        self.start_time = datetime.now()  # Add start time for chat

        # Initialize client based on provider
        self._setup_client(model_config)

        # System prompt
        self.messages = [
            {
                "role": "system", 
                "content": f"You are {self.model_name}, an advanced AI assistant. "
                           "Provide clear, concise, and helpful responses."
            }
        ]
        
        self.logger.info(f"Initialized chat with model: {self.model_name}")
    
    def _setup_client(self, model_config):
        """
        Set up the appropriate client based on the model provider
        
        Args:
            model_config (dict): Configuration for the selected model
        """
        if self.provider == 'openrouter':
            if not OPENROUTER_AVAILABLE:
                raise ImportError("OpenRouter library not installed. Please install 'openai'.")
            
            self.api_key = os.getenv('OPENROUTER_API_KEY')
            if not self.api_key:
                self.logger.error("OpenRouter API key not found")
                raise ValueError("OpenRouter API key not found in .env file")
            
            self.api_base = "https://openrouter.ai/api/v1"
        else:
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI library not installed. Please install 'openai'.")
            
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                self.logger.error("OpenAI API key not found")
                raise ValueError("OpenAI API key not found in .env file")
            
            self.client = OpenAI(api_key=api_key)

    def send_message(self, user_input):
        """
        Send user message and get AI response
        
        Args:
            user_input (str): User's input message
        
        Returns:
            str or None: AI's response or None if error occurs
        """
        try:
            if not user_input.strip():
                self.logger.warning("Empty user input")
                return None
            
            # Add user message to conversation history
            self.messages.append({"role": "user", "content": user_input})
            
            # Show thinking message
            thinking_message = self.console.status("[bold yellow]AI is thinking...[/bold yellow]")
            thinking_message.start()
            
            try:
                # API call based on provider
                if self.provider == 'openrouter':
                    # OpenRouter API call
                    response = requests.post(
                        f"{self.api_base}/chat/completions",
                        headers=get_openrouter_headers(self.api_key),
                        json={
                            "model": self.model_id,
                            "messages": self.messages,
                            "max_tokens": 4000,
                            "temperature": 0.7,
                            "stream": False
                        }
                    )
                    response.raise_for_status()  # Raise exception for bad status codes
                    data = response.json()
                    ai_response = data['choices'][0]['message']['content'].strip()
                else:
                    # OpenAI API call
                    response = self.client.chat.completions.create(
                        model=self.model_id,
                        messages=self.messages,
                        max_tokens=4000,
                        temperature=0.7,
                        stream=False
                    )
                    ai_response = response.choices[0].message.content.strip()
            finally:
                thinking_message.stop()
            
            # Format and display the response
            self._display_response(ai_response)
            
            # Add AI response to conversation history
            self.messages.append({"role": "assistant", "content": ai_response})
            
            return ai_response
        
        except Exception as e:
            self.logger.error(f"API Error: {e}", exc_info=True)
            self.console.print(f"[bold red]Error: {e}[/bold red]")
            return None

    def _display_response(self, response_text):
        """
        Format and display the AI response with enhanced markdown and code formatting
        
        Args:
            response_text (str): The response text to display
        """
        if not response_text:
            return

        # Create a panel to display the response
        self.console.print(
            Panel(
                Markdown(response_text),
                title=f"[bold #A6E22E]{self.model_name}[/]",
                border_style="bright_blue",
                padding=(1, 2),
                expand=True
            )
        )
    
    def save_chat(self):
        """Save the chat conversation to both JSON and text files"""
        try:
            # Create base chats directory
            chats_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chats')
            os.makedirs(chats_dir, exist_ok=True)

            # Format the timestamp in 12-hour format
            timestamp = self.start_time.strftime("%Y%m%d_%I%M%p")

            if self.provider == 'openrouter':
                # For OpenRouter, create company/model subdirectories
                company, model_name = self.model_id.split('/')
                chat_dir = os.path.join(chats_dir, 'openrouter', company, model_name)
            else:
                # For other providers (e.g., OpenAI), just use provider name
                chat_dir = os.path.join(chats_dir, self.provider)

            os.makedirs(chat_dir, exist_ok=True)

            # Base filename without extension
            base_filename = f"chat_{timestamp}"
            
            # Save JSON format
            json_filepath = os.path.join(chat_dir, f"{base_filename}.json")
            chat_data = {
                "model_name": self.model_name,
                "model_id": self.model_id,
                "provider": self.provider,
                "timestamp": self.start_time.isoformat(),
                "messages": self.messages
            }
            with open(json_filepath, 'w', encoding='utf-8') as f:
                json.dump(chat_data, f, indent=2, ensure_ascii=False)

            # Save text format with nice formatting
            text_filepath = os.path.join(chat_dir, f"{base_filename}.txt")
            with open(text_filepath, 'w', encoding='utf-8') as f:
                # Write header
                f.write("="*80 + "\n")
                f.write(f"Chat with {self.model_name}\n")
                f.write(f"Model ID: {self.model_id}\n")
                f.write(f"Provider: {self.provider}\n")
                f.write(f"Date: {self.start_time.strftime('%Y-%m-%d %I:%M:%S %p')}\n")
                f.write("="*80 + "\n\n")

                # Write messages
                for msg in self.messages[1:]:  # Skip system message
                    role = msg['role'].upper()
                    content = msg['content']
                    
                    # Format role header
                    if role == "USER":
                        f.write("YOU:\n")
                    else:
                        f.write(f"{self.model_name}:\n")
                    
                    # Write message content with proper formatting
                    lines = content.split('\n')
                    in_code_block = False
                    for line in lines:
                        if line.startswith('```'):
                            if in_code_block:
                                # End code block
                                f.write("-"*80 + "\n")
                                in_code_block = False
                            else:
                                # Start code block
                                f.write("-"*80 + "\n")
                                # Get language if specified
                                lang = line[3:].strip()
                                if lang:
                                    f.write(f"Code ({lang}):\n")
                                in_code_block = True
                        else:
                            if in_code_block:
                                # Indent code
                                f.write("    " + line + "\n")
                            else:
                                f.write(line + "\n")
                    
                    # Add spacing between messages
                    f.write("\n" + "-"*40 + "\n\n")

            self.console.print(f"[green]Chat saved to:[/green]")
            self.console.print(f"[blue]JSON: {json_filepath}[/blue]")
            self.console.print(f"[blue]Text: {text_filepath}[/blue]")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save chat: {e}", exc_info=True)
            self.console.print(f"[bold red]Error saving chat: {e}[/bold red]")
            return False

    def chat_loop(self):
        """
        Main chat interaction loop
        """
        self.logger.info(f"Starting chat loop for {self.model_name}")
        
        # Get the script directory for relative paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.console.print(
            Panel(
                f"[bold cyan]Chat session with {self.model_name}[/bold cyan]\n\n"
                "📝 Type your message and press Enter to send\n"
                "🔗 Reference files and directories:\n"
                "   [[ file:example.py]]          - View single file contents\n"
                '   [[ file:"path/to/file.txt"]]  - Paths with spaces need quotes\n'
                "   [[ dir:folder]]               - List directory contents\n"
                "   [[ codebase:folder]]          - View all code files in directory\n"
                '   [[ codebase:"src/*.py"]]      - View Python files in src folder\n'
                "💾 Commands:\n"
                "   - /save - Save the chat history\n"
                "   - /clear - Clear the screen and chat history\n"
                "❌ Type 'exit', 'quit', or press Ctrl+C to end the session",
                title="[bold white]Chat Session[/bold white]",
                border_style="cyan",
                padding=(1, 2)
            )
        )
        
        try:
            while True:
                try:
                    user_input = self.console.input("[bold yellow]You: [/bold yellow]")
                    
                    if user_input.lower() in ['exit', 'quit', 'bye']:
                        self.logger.info("Chat session ended by user")
                        self.console.print("[bold cyan]Goodbye![/bold cyan]")
                        break

                    # Handle commands
                    if user_input.startswith('/'):
                        command = user_input.lower().strip()
                        if command == '/save':
                            if len(self.messages) > 1:
                                if self.save_chat():
                                    self.console.print("[bold green]Chat history saved successfully![/bold green]")
                                else:
                                    self.console.print("[bold red]Failed to save chat history.[/bold red]")
                            else:
                                self.console.print("[yellow]No messages to save yet.[/yellow]")
                            continue
                        elif command == '/clear':
                            # Clear screen
                            os.system('cls' if os.name == 'nt' else 'clear')
                            # Reset chat history to only system message
                            self.messages = [self.messages[0]]
                            # Redisplay the welcome panel
                            self.console.print(
                                Panel(
                                    f"[bold cyan]Chat session with {self.model_name}[/bold cyan]\n\n"
                                    "📝 Type your message and press Enter to send\n"
                                    "🔗 Reference files and directories:\n"
                                    "   [[ file:example.py]]          - View single file contents\n"
                                    '   [[ file:"path/to/file.txt"]]  - Paths with spaces need quotes\n'
                                    "   [[ dir:folder]]               - List directory contents\n"
                                    "   [[ codebase:folder]]          - View all code files in directory\n"
                                    '   [[ codebase:"src/*.py"]]      - View Python files in src folder\n'
                                    "💾 Commands:\n"
                                    "   - /save - Save the chat history\n"
                                    "   - /clear - Clear the screen and chat history\n"
                                    "❌ Type 'exit', 'quit', or press Ctrl+C to end the session",
                                    title="[bold white]Chat Session[/bold white]",
                                    border_style="cyan",
                                    padding=(1, 2)
                                )
                            )
                            self.console.print("[bold green]Screen and chat history cleared![/bold green]")
                            continue
                        else:
                            self.console.print(f"[yellow]Unknown command: {command}[/yellow]")
                            continue
                    
                    # Handle directory references
                    if '[[dir:' in user_input and ']]' in user_input:
                        modified_input = user_input
                        
                        # Find all directory references
                        while '[[dir:' in modified_input and ']]' in modified_input:
                            start = modified_input.find('[[dir:')
                            end = modified_input.find(']]', start)
                            
                            if start != -1 and end != -1:
                                # Extract and normalize the directory path
                                raw_path = modified_input[start+6:end].strip()
                                # Handle quoted paths
                                if raw_path.startswith('"') and raw_path.endswith('"'):
                                    raw_path = raw_path[1:-1]  # Remove quotes
                                elif raw_path.startswith("'") and raw_path.endswith("'"):
                                    raw_path = raw_path[1:-1]  # Remove quotes
                                
                                try:
                                    # Try different path resolutions
                                    possible_paths = []
                                    
                                    # Convert path to proper format
                                    normalized_path = os.path.normpath(raw_path.replace('\\', '/'))
                                    
                                    # If it's an absolute path
                                    if os.path.isabs(normalized_path):
                                        possible_paths.append(normalized_path)
                                    else:
                                        # Try relative paths
                                        possible_paths.extend([
                                            os.path.normpath(os.path.join(script_dir, normalized_path)),  # Relative to script
                                            os.path.normpath(os.path.join(os.getcwd(), normalized_path)),  # Relative to current directory
                                            os.path.abspath(normalized_path)  # Absolute path
                                        ])
                                    
                                    # Remove duplicates while preserving order
                                    possible_paths = list(dict.fromkeys(possible_paths))
                                    
                                    # Debug output
                                    self.logger.debug(f"Attempting to read directory: {raw_path}")
                                    self.logger.debug("Trying paths:")
                                    for p in possible_paths:
                                        self.logger.debug(f"- {p}")
                                    
                                    dir_found = False
                                    for path in possible_paths:
                                        if os.path.isdir(path):
                                            try:
                                                # Get directory contents
                                                contents = os.listdir(path)
                                                
                                                # Separate files and directories
                                                dirs = []
                                                files = []
                                                for item in contents:
                                                    item_path = os.path.join(path, item)
                                                    if os.path.isdir(item_path):
                                                        dirs.append(f"📁 {item}/")
                                                    else:
                                                        files.append(f"📄 {item}")
                                                
                                                # Sort alphabetically
                                                dirs.sort()
                                                files.sort()
                                                
                                                # Format the content with directory info
                                                formatted_content = (
                                                    f"\nContents of directory: {os.path.basename(path) or path}\n"
                                                    f"{'=' * 40}\n"
                                                )
                                                
                                                if dirs:
                                                    formatted_content += "Directories:\n" + "\n".join(dirs) + "\n"
                                                if files:
                                                    if dirs:
                                                        formatted_content += "\n"
                                                    formatted_content += "Files:\n" + "\n".join(files) + "\n"
                                                if not dirs and not files:
                                                    formatted_content += "(Empty directory)\n"
                                                
                                                formatted_content += f"{'=' * 40}\n"
                                                
                                                # Replace the directory reference with the content
                                                modified_input = (
                                                    modified_input[:start] + 
                                                    formatted_content + 
                                                    modified_input[end+2:]
                                                )
                                                
                                                dir_found = True
                                                self.logger.debug(f"Successfully read directory: {path}")
                                                break
                                            except Exception as e:
                                                self.logger.error(f"Failed to read directory {path}: {str(e)}")
                                                continue
                                    
                                    if not dir_found:
                                        error_msg = (
                                            f"Could not find or read directory: {raw_path}\n"
                                            f"Tried the following paths:\n" + 
                                            "\n".join(f"- {p}" for p in possible_paths)
                                        )
                                        raise FileNotFoundError(error_msg)
                                    
                                except Exception as e:
                                    error_msg = f"Error reading directory: {str(e)}"
                                    self.logger.error(error_msg)
                                    self.console.print(f"[bold red]{error_msg}[/bold red]")
                                    # Remove the failed directory reference
                                    modified_input = modified_input[:start] + modified_input[end+2:]
                        
                        # Use the modified input with directory contents
                        user_input = modified_input
                    
                    # Handle file references
                    if '[[file:' in user_input and ']]' in user_input:
                        modified_input = user_input
                        
                        # Find all file references
                        while '[[file:' in modified_input and ']]' in modified_input:
                            start = modified_input.find('[[file:')
                            end = modified_input.find(']]', start)
                            
                            if start != -1 and end != -1:
                                # Extract and normalize the file path
                                raw_path = modified_input[start+7:end].strip()
                                # Handle quoted paths
                                if raw_path.startswith('"') and raw_path.endswith('"'):
                                    raw_path = raw_path[1:-1]  # Remove quotes
                                elif raw_path.startswith("'") and raw_path.endswith("'"):
                                    raw_path = raw_path[1:-1]  # Remove quotes
                                
                                try:
                                    # Try different path resolutions
                                    possible_paths = []
                                    
                                    # Convert path to proper format
                                    normalized_path = os.path.normpath(raw_path.replace('\\', '/'))
                                    
                                    # If it's an absolute path
                                    if os.path.isabs(normalized_path):
                                        possible_paths.append(normalized_path)
                                    else:
                                        # Try relative paths
                                        possible_paths.extend([
                                            os.path.normpath(os.path.join(script_dir, normalized_path)),  # Relative to script
                                            os.path.normpath(os.path.join(os.getcwd(), normalized_path)),  # Relative to current directory
                                            os.path.abspath(normalized_path)  # Absolute path
                                        ])
                                    
                                    # Remove duplicates while preserving order
                                    possible_paths = list(dict.fromkeys(possible_paths))
                                    
                                    # Debug output
                                    self.logger.debug(f"Attempting to read file: {raw_path}")
                                    self.logger.debug("Trying paths:")
                                    for p in possible_paths:
                                        self.logger.debug(f"- {p}")
                                    
                                    file_found = False
                                    for path in possible_paths:
                                        if os.path.isfile(path):
                                            try:
                                                # Read file contents
                                                with open(path, 'r', encoding='utf-8') as f:
                                                    file_content = f.read()
                                                
                                                # Get file extension for syntax highlighting
                                                ext = os.path.splitext(path)[1][1:] or 'text'
                                                
                                                # Format the content with file info
                                                formatted_content = (
                                                    f"\nContents of {os.path.basename(path)}:\n"
                                                    f"```{ext}\n"
                                                    f"{file_content}\n"
                                                    f"```\n"
                                                )
                                                
                                                # Replace the file reference with the content
                                                modified_input = (
                                                    modified_input[:start] + 
                                                    formatted_content + 
                                                    modified_input[end+2:]
                                                )
                                                
                                                file_found = True
                                                self.logger.debug(f"Successfully read file: {path}")
                                                break
                                            except Exception as e:
                                                self.logger.error(f"Failed to read {path}: {str(e)}")
                                                continue
                                    
                                    if not file_found:
                                        error_msg = (
                                            f"Could not find or read file: {raw_path}\n"
                                            f"Tried the following paths:\n" + 
                                            "\n".join(f"- {p}" for p in possible_paths)
                                        )
                                        raise FileNotFoundError(error_msg)
                                    
                                except Exception as e:
                                    error_msg = f"Error reading file: {str(e)}"
                                    self.logger.error(error_msg)
                                    self.console.print(f"[bold red]{error_msg}[/bold red]")
                                    # Remove the failed file reference
                                    modified_input = modified_input[:start] + modified_input[end+2:]
                        
                        # Use the modified input with file contents
                        user_input = modified_input
                    
                    # Handle codebase references
                    if '[[codebase:' in user_input and ']]' in user_input:
                        modified_input = user_input
                        
                        # Find all codebase references
                        while '[[codebase:' in modified_input and ']]' in modified_input:
                            start = modified_input.find('[[codebase:')
                            end = modified_input.find(']]', start)
                            
                            if start != -1 and end != -1:
                                # Extract and normalize the path
                                raw_path = modified_input[start+11:end].strip()
                                # Handle quoted paths
                                if raw_path.startswith('"') and raw_path.endswith('"'):
                                    raw_path = raw_path[1:-1]  # Remove quotes
                                elif raw_path.startswith("'") and raw_path.endswith("'"):
                                    raw_path = raw_path[1:-1]  # Remove quotes
                                
                                try:
                                    # Try different path resolutions
                                    possible_paths = []
                                    
                                    # Convert path to proper format
                                    normalized_path = os.path.normpath(raw_path.replace('\\', '/'))
                                    
                                    # If it's an absolute path
                                    if os.path.isabs(normalized_path):
                                        possible_paths.append(normalized_path)
                                    else:
                                        # Try relative paths
                                        possible_paths.extend([
                                            os.path.normpath(os.path.join(script_dir, normalized_path)),  # Relative to script
                                            os.path.normpath(os.path.join(os.getcwd(), normalized_path)),  # Relative to current directory
                                            os.path.abspath(normalized_path)  # Absolute path
                                        ])
                                    
                                    # Remove duplicates while preserving order
                                    possible_paths = list(dict.fromkeys(possible_paths))
                                    
                                    # Debug output
                                    self.logger.debug(f"Attempting to read codebase: {raw_path}")
                                    self.logger.debug("Trying paths:")
                                    for p in possible_paths:
                                        self.logger.debug(f"- {p}")
                                    
                                    def is_code_file(filename):
                                        """Check if a file is likely to contain code or documentation"""
                                        code_extensions = {
                                            # Code files
                                            '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c',
                                            '.h', '.cs', '.php', '.rb', '.go', '.rs', '.swift', '.kt',
                                            '.scala', '.m', '.mm', '.sh', '.bash', '.ps1', '.r', '.pl',
                                            '.html', '.css', '.scss', '.sass', '.less', '.vue', '.sql',
                                            # Documentation files
                                            '.md', '.markdown', '.rst', '.txt', '.json', '.yaml', '.yml',
                                            '.toml', '.ini', '.cfg', '.conf'
                                        }
                                        return os.path.splitext(filename)[1].lower() in code_extensions
                                    
                                    def collect_code_files(path):
                                        """Recursively collect code and documentation files from directory"""
                                        code_files = []
                                        try:
                                            if os.path.isfile(path):
                                                if is_code_file(path):
                                                    code_files.append(path)
                                            elif os.path.isdir(path):
                                                for root, _, files in os.walk(path):
                                                    for file in sorted(files):  # Sort files within each directory
                                                        file_path = os.path.join(root, file)
                                                        if is_code_file(file):
                                                            code_files.append(file_path)
                                        except Exception as e:
                                            self.logger.error(f"Error collecting files from {path}: {e}")
                                        return sorted(code_files)  # Sort all paths for consistent ordering
                                    
                                    def format_file_content(file_path, base_path):
                                        """Format a single file's content with proper headers and syntax highlighting"""
                                        try:
                                            with open(file_path, 'r', encoding='utf-8') as f:
                                                content = f.read()
                                            
                                            # Get relative path for display
                                            if os.path.isdir(base_path):
                                                rel_path = os.path.relpath(file_path, base_path)
                                            else:
                                                rel_path = os.path.basename(file_path)
                                            
                                            # Get file extension for syntax highlighting
                                            ext = os.path.splitext(file_path)[1][1:] or 'text'
                                            if ext in ['md', 'markdown']:
                                                ext = 'markdown'  # Ensure proper markdown highlighting
                                            
                                            return (
                                                f"\n📄 {rel_path}\n"
                                                f"{'-' * 80}\n"
                                                f"```{ext}\n"
                                                f"{content}\n"
                                                f"```\n"
                                            )
                                        except Exception as e:
                                            self.logger.error(f"Failed to read {file_path}: {e}")
                                            return f"\n⚠️ Error reading {rel_path}: {str(e)}\n"
                                    
                                    codebase_found = False
                                    for path in possible_paths:
                                        if os.path.exists(path):
                                            try:
                                                # Collect all code files
                                                code_files = collect_code_files(path)
                                                
                                                if not code_files:
                                                    continue
                                                
                                                # Format the content with codebase info
                                                formatted_content = (
                                                    f"\nCodebase contents from: {os.path.basename(path) or path}\n"
                                                    f"{'=' * 80}\n"
                                                    f"Found {len(code_files)} file(s)\n"
                                                    f"{'-' * 80}\n"
                                                )
                                                
                                                # Add each file's contents
                                                for file_path in code_files:
                                                    formatted_content += format_file_content(file_path, path)
                                                
                                                formatted_content += f"{'=' * 80}\n"
                                                
                                                # Replace the codebase reference with the content
                                                modified_input = (
                                                    modified_input[:start] + 
                                                    formatted_content + 
                                                    modified_input[end+2:]
                                                )
                                                
                                                codebase_found = True
                                                self.logger.debug(f"Successfully read codebase: {path}")
                                                break
                                            except Exception as e:
                                                self.logger.error(f"Failed to process codebase {path}: {str(e)}")
                                                continue
                                    
                                    if not codebase_found:
                                        error_msg = (
                                            f"Could not find or read codebase: {raw_path}\n"
                                            f"Tried the following paths:\n" + 
                                            "\n".join(f"- {p}" for p in possible_paths)
                                        )
                                        raise FileNotFoundError(error_msg)
                                    
                                except Exception as e:
                                    error_msg = f"Error reading codebase: {str(e)}"
                                    self.logger.error(error_msg)
                                    self.console.print(f"[bold red]{error_msg}[/bold red]")
                                    # Remove the failed codebase reference
                                    modified_input = modified_input[:start] + modified_input[end+2:]
                        
                        # Use the modified input with codebase contents
                        user_input = modified_input
                    
                    self.send_message(user_input)
                
                except KeyboardInterrupt:
                    self.logger.warning("Chat interrupted by user")
                    self.console.print("\n[bold cyan]Chat interrupted. Exiting...[/bold cyan]")
                    break
                except Exception as e:
                    self.logger.error(f"Error in chat loop: {e}", exc_info=True)
                    self.console.print(f"[bold red]Error: {str(e)}[/bold red]")
                    continue
        
        except Exception as e:
            self.logger.error(f"Fatal error in chat loop: {e}", exc_info=True)
            self.console.print(f"[bold red]Fatal error: {e}[/bold red]")

class AIChatApp:
    def __init__(self, logger, console):
        """
        Initialize AI Chat Application
        
        Args:
            logger (logging.Logger): Logging instance
            console (rich.console.Console): Rich console for output
        """
        self.logger = logger
        self.console = console
        
        # Load models from JSON
        try:
            models_path = os.path.join(os.path.dirname(__file__), 'models.json')
            with open(models_path, 'r') as f:
                self.models_config = json.load(f)['models']
            
            # Load favorites
            self.favorites_path = os.path.join(os.path.dirname(__file__), 'favorites.json')
            if os.path.exists(self.favorites_path):
                with open(self.favorites_path, 'r') as f:
                    self.favorites = json.load(f)['favorites']
            else:
                self.favorites = []
                self.save_favorites()
            
            # Initialize OpenRouter API if key exists
            dotenv.load_dotenv()
            self.openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
            if self.openrouter_api_key:
                self.openrouter = OpenRouterAPI(self.openrouter_api_key)
            else:
                self.openrouter = None
                self.logger.warning("OpenRouter API key not found, OpenRouter models will not be available")
            
            self.logger.info(f"Loaded {len(self.models_config)} models from configuration")
        except FileNotFoundError:
            self.logger.error("models.json not found")
            raise FileNotFoundError("models.json configuration file is missing")
        except json.JSONDecodeError:
            self.logger.error("Invalid models.json format")
            raise ValueError("Invalid models.json configuration")

    def save_favorites(self):
        """Save favorites to JSON file"""
        with open(self.favorites_path, 'w') as f:
            json.dump({'favorites': self.favorites}, f, indent=4)

    def add_to_favorites(self, model_config):
        """Add a model to favorites"""
        # Create a unique identifier for the model
        model_id = f"{model_config.get('provider', 'unknown')}:{model_config['id']}"
        
        # Check if model is already in favorites
        if not any(f['id'] == model_config['id'] for f in self.favorites):
            favorite = {
                'id': model_config['id'],
                'name': model_config['name'],
                'provider': model_config.get('provider', 'unknown'),
                'description': model_config.get('description', 'No description')
            }
            self.favorites.append(favorite)
            self.save_favorites()
            self.console.print(f"[green]Added {model_config['name']} to favorites[/green]")
        else:
            self.console.print(f"[yellow]{model_config['name']} is already in favorites[/yellow]")

    def remove_from_favorites(self, model_id):
        """Remove a model from favorites"""
        self.favorites = [f for f in self.favorites if f['id'] != model_id]
        self.save_favorites()

    def manage_favorites(self):
        """Display favorites management menu"""
        if not self.favorites:
            self.console.print("[yellow]No favorite models yet[/yellow]")
            return

        # Create choices for favorite models
        choices = [(f"{f['name']} ({f['provider']})", f) for f in self.favorites]
        choices.append(("Back", None))

        questions = [
            inquirer.List('favorite',
                message="Select favorite to manage",
                choices=choices,
                carousel=True
            ),
        ]

        answer = inquirer.prompt(questions)
        if not answer or answer['favorite'] is None:
            return

        selected = answer['favorite']
        action_choices = [
            ("Start Chat", "chat"),
            ("Remove from Favorites", "remove"),
            ("Back", "back")
        ]

        action_question = [
            inquirer.List('action',
                message=f"Manage {selected['name']}",
                choices=action_choices,
                carousel=True
            ),
        ]

        action_answer = inquirer.prompt(action_question)
        if not action_answer:
            return

        if action_answer['action'] == "chat":
            self.start_chat(selected)
        elif action_answer['action'] == "remove":
            self.remove_from_favorites(selected['id'])
            self.console.print(f"[green]Removed {selected['name']} from favorites[/green]")

    def select_openrouter_model(self):
        """Display nested menu for OpenRouter model selection"""
        try:
            if not self.openrouter:
                self.console.print("[bold red]OpenRouter API key not found. Please add it to your .env file.[/bold red]")
                return None
            
            # Fetch models with debug output
            self.console.print("[cyan]Fetching available models...[/cyan]")
            try:
                models = self.openrouter.fetch_models()
                self.console.print(f"[green]Found {len(models)} available models[/green]")
            except Exception as e:
                self.console.print(f"[bold red]Error fetching models: {e}[/bold red]")
                return None
            
            # Group models by company
            grouped_models = self.openrouter.group_models_by_company(models)
            if not grouped_models:
                self.console.print("[bold yellow]No models available[/bold yellow]")
                return None
            
            # Company selection
            companies = list(grouped_models.keys())
            companies.sort()
            
            # Show company statistics
            self.console.print(f"\n[cyan]Available Companies ({len(companies)}):[/cyan]")
            for company in companies:
                models = grouped_models[company]
                top_models = sum(1 for m in models if m.get('top_provider', False))
                self.console.print(
                    f"[cyan]{company}: {len(models)} models "
                    f"({top_models} featured)[/cyan]"
                )
            
            companies.append("Back")
            
            company_question = [
                inquirer.List('company',
                    message="Select AI Company",
                    choices=companies,
                    carousel=True
                ),
            ]
            
            company_answer = inquirer.prompt(company_question)
            if not company_answer or company_answer['company'] == "Back":
                return None
            
            selected_company = company_answer['company']
            
            # Model selection for chosen company
            company_models = grouped_models[selected_company]
            model_choices = []
            
            # Show company info
            self.console.print(f"\n[cyan]Models from {selected_company}:[/cyan]")
            
            for model in company_models:
                # Create a concise model description
                name = model['name'].split(':')[-1].strip()  # Remove company prefix
                
                # Handle context length
                try:
                    context_length = int(model.get('context_length', 0))
                    context = f"{context_length // 1000}K" if context_length else "Unknown"
                except (ValueError, TypeError):
                    context = "Unknown"
                
                # Handle price formatting
                try:
                    price = model.get('pricing', {}).get('prompt')
                    if price is None:
                        price_str = "Price N/A"
                    else:
                        # Convert string to float if necessary
                        price_float = float(price) if isinstance(price, str) else price
                        price_str = f"${price_float:.6f}"
                except (ValueError, TypeError):
                    price_str = "Price N/A"
                
                # Add featured badge for top providers
                featured = "⭐ " if model.get('top_provider', False) else ""
                
                # Format the model information in a single line
                model_info = f"{featured}{name} (Context: {context}, {price_str})"
                model_choices.append((model_info, model))
            
            model_choices.append(("Back", None))
            
            model_question = [
                inquirer.List('model',
                    message=f"Select {selected_company} Model",
                    choices=model_choices,
                    carousel=True
                ),
            ]
            
            model_answer = inquirer.prompt(model_question)
            if not model_answer or model_answer['model'] is None:
                return self.select_openrouter_model()  # Go back to company selection
            
            return model_answer['model']
        
        except Exception as e:
            self.logger.error(f"Error selecting OpenRouter model: {e}", exc_info=True)
            self.console.print(f"[bold red]Error in OpenRouter model selection: {e}[/bold red]")
            return None
    
    def display_main_menu(self):
        """
        Display the main menu for model selection
        """
        self.logger.info("Displaying main menu")
        
        # Display welcome message
        self.console.print(Panel(
            "[bold cyan]Welcome to AI Chat[/bold cyan]\n\n"
            "🤖 Choose your AI companion from the available models below.\n"
            "📝 Each model has different capabilities and specialties.\n"
            "❌ Press Ctrl+C to exit at any time.",
            title="[bold white]AI Chat Menu[/bold white]",
            border_style="cyan",
            padding=(1, 2)
        ))
        
        while True:
            try:
                # Main menu choices
                main_choices = [
                    ("=== Select Provider ===", None),
                    ("⭐ Favorite Models", "favorites"),
                    ("OpenAI Models", "openai"),
                    ("OpenRouter Models", "openrouter"),
                    ("Exit Application", "exit")
                ]
                
                # Create the main menu question
                questions = [
                    inquirer.List('provider',
                        message="Select AI Provider",
                        choices=main_choices,
                        carousel=True
                    ),
                ]
                
                # Prompt user for provider selection
                answer = inquirer.prompt(questions)
                
                if not answer or answer['provider'] == "exit":
                    self.logger.info("User selected to exit")
                    self.console.print(Panel(
                        "[bold yellow]Thank you for using AI Chat! Goodbye![/bold yellow]",
                        border_style="yellow"
                    ))
                    break
                
                selected_provider = answer['provider']

                if selected_provider == "favorites":
                    self.manage_favorites()
                    continue
                
                # Handle provider selection
                if selected_provider == "openai":
                    # Filter and display OpenAI models
                    openai_models = [
                        (f"{model['name']} - {model.get('description', 'No description')}", model)
                        for model in self.models_config
                        if model.get('provider', 'openai').lower() == 'openai'
                    ]
                    openai_models.append(("Back", None))
                    
                    model_question = [
                        inquirer.List('model',
                            message="Select OpenAI Model",
                            choices=openai_models,
                            carousel=True
                        ),
                    ]
                    
                    model_answer = inquirer.prompt(model_question)
                    if model_answer and model_answer['model'] is not None:
                        selected_model = model_answer['model']
                        # Add option to favorite after selection
                        action_choices = [
                            ("Start Chat", "chat"),
                            ("Add to Favorites", "favorite"),
                            ("Back", "back")
                        ]
                        
                        action_question = [
                            inquirer.List('action',
                                message=f"Action for {selected_model['name']}",
                                choices=action_choices,
                                carousel=True
                            ),
                        ]
                        
                        action_answer = inquirer.prompt(action_question)
                        if action_answer:
                            if action_answer['action'] == "chat":
                                self.start_chat(selected_model)
                            elif action_answer['action'] == "favorite":
                                self.add_to_favorites(selected_model)
                
                elif selected_provider == "openrouter":
                    # Show OpenRouter nested menu
                    selected_model = self.select_openrouter_model()
                    if selected_model:
                        # Convert OpenRouter API model to our format
                        model_config = {
                            'id': selected_model['id'],
                            'name': selected_model['name'],
                            'description': selected_model.get('description', 'No description'),
                            'provider': 'openrouter'
                        }
                        # Add option to favorite after selection
                        action_choices = [
                            ("Start Chat", "chat"),
                            ("Add to Favorites", "favorite"),
                            ("Back", "back")
                        ]
                        
                        action_question = [
                            inquirer.List('action',
                                message=f"Action for {model_config['name']}",
                                choices=action_choices,
                                carousel=True
                            ),
                        ]
                        
                        action_answer = inquirer.prompt(action_question)
                        if action_answer:
                            if action_answer['action'] == "chat":
                                self.start_chat(model_config)
                            elif action_answer['action'] == "favorite":
                                self.add_to_favorites(model_config)
            
            except KeyboardInterrupt:
                self.logger.warning("Application interrupted")
                self.console.print(Panel(
                    "\n[bold yellow]Thanks for using AI Chat! Goodbye![/bold yellow]",
                    border_style="yellow"
                ))
                break
            except Exception as e:
                self.logger.error(f"Menu error: {e}", exc_info=True)
                self.console.print(f"[bold red]Error in menu: {e}[/bold red]")
                continue
    
    def start_chat(self, model_config):
        """
        Start chat with selected model
        
        Args:
            model_config (dict): Configuration for the selected model
        """
        try:
            # Initialize and start chat with selected model
            chat = AIChat(model_config, self.logger, self.console)
            chat.chat_loop()
        except Exception as e:
            self.logger.error(f"Error starting chat: {e}", exc_info=True)
            self.console.print(f"[bold red]Error starting chat: {e}[/bold red]")

def main():
    """
    Main application entry point
    """
    # Setup logging
    logger, console = setup_logging()
    
    try:
        # Ensure .env file exists
        env_path = os.path.join(os.path.dirname(__file__), '.env')
        if not os.path.exists(env_path):
            logger.error("Missing .env file")
            console.print("[bold red]Create a .env file with API keys[/bold red]")
            sys.exit(1)
        
        # Create and run the application
        app = AIChatApp(logger, console)
        app.display_main_menu()
    
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}", exc_info=True)
        console.print(f"[bold red]Critical error: {e}[/bold red]")
        sys.exit(1)

if __name__ == "__main__":
    main()