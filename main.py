import os
import sys
import json
import logging
import base64
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
from urllib.parse import urlparse
from io import BytesIO
from PIL import Image

# Document handling imports
try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    from PyPDF2 import PdfReader
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    import odf
    from odf import text, teletype
    ODF_AVAILABLE = True
except ImportError:
    ODF_AVAILABLE = False

try:
    import rtf
    RTF_AVAILABLE = True
except ImportError:
    RTF_AVAILABLE = False

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

try:
    import anthropic
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

class SettingsManager:
    """Class to manage application settings"""
    def __init__(self, logger, console):
        self.logger = logger
        self.console = console
        self.settings_file = os.path.join(os.path.dirname(__file__), 'settings.json')
        self.settings = self._load_settings()

    def _load_settings(self):
        """Load settings from file"""
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {
                'codebase_search': {
                    'file_types': ['.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h', '.cs', '.php', '.rb', '.go', '.rs', '.swift', '.kt', '.scala', '.html', '.css', '.scss', '.sass', '.less', '.vue', '.sql', '.md', '.txt', '.json', '.yaml', '.yml', '.toml'],
                    'enabled_types': ['.py'],  # Default to only Python files
                    'search_subdirectories': True,
                    'max_file_size_mb': 10,
                    'exclude_patterns': ['node_modules', 'venv', '.git', '__pycache__', 'build', 'dist']
                }
            }
        except Exception as e:
            self.logger.error(f"Error loading settings: {e}")
            return self._load_settings()  # Return default settings

    def _save_settings(self):
        """Save settings to file"""
        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=4)
        except Exception as e:
            self.logger.error(f"Error saving settings: {e}")

    def get_setting(self, category, key):
        """Get a specific setting value"""
        return self.settings.get(category, {}).get(key)

    def update_setting(self, category, key, value):
        """Update a specific setting value"""
        if category not in self.settings:
            self.settings[category] = {}
        self.settings[category][key] = value
        self._save_settings()

    def manage_codebase_settings(self):
        """Manage codebase search settings"""
        while True:
            codebase_settings = self.settings['codebase_search']
            enabled_types = set(codebase_settings['enabled_types'])
            all_types = codebase_settings['file_types']

            choices = [
                ("=== Codebase Search Settings ===", None),
                (f"Search Subdirectories: {'✓' if codebase_settings['search_subdirectories'] else '✗'}", "toggle_subdirs"),
                ("Manage File Types", "file_types"),
                ("Set Max File Size (MB)", "max_size"),
                ("Manage Exclude Patterns", "exclude_patterns"),
                ("Back to Settings Menu", "back")
            ]

            questions = [
                inquirer.List('action',
                    message="Select setting to modify",
                    choices=choices,
                    carousel=True
                ),
            ]

            answer = inquirer.prompt(questions)
            if not answer or answer['action'] == "back":
                break

            if answer['action'] == "toggle_subdirs":
                current = codebase_settings['search_subdirectories']
                self.update_setting('codebase_search', 'search_subdirectories', not current)
                self.console.print(f"[green]Search subdirectories {'enabled' if not current else 'disabled'}[/green]")

            elif answer['action'] == "file_types":
                # Create checkboxes for file types
                file_type_choices = [(f"{ft} {'[✓]' if ft in enabled_types else '[ ]'}", ft) for ft in all_types]
                file_type_choices.append(("Done", None))

                while True:
                    type_question = [
                        inquirer.List('file_type',
                            message="Toggle file types (select to toggle, 'Done' to finish)",
                            choices=file_type_choices,
                            carousel=True
                        ),
                    ]

                    type_answer = inquirer.prompt(type_question)
                    if not type_answer or type_answer['file_type'] is None:
                        break

                    selected_type = type_answer['file_type']
                    if selected_type in enabled_types:
                        enabled_types.remove(selected_type)
                    else:
                        enabled_types.add(selected_type)

                    # Update choices to reflect changes
                    file_type_choices = [(f"{ft} {'[✓]' if ft in enabled_types else '[ ]'}", ft) for ft in all_types]
                    file_type_choices.append(("Done", None))

                self.update_setting('codebase_search', 'enabled_types', list(enabled_types))
                self.console.print("[green]File types updated[/green]")

            elif answer['action'] == "max_size":
                size_question = [
                    inquirer.Text('size',
                        message="Enter maximum file size in MB",
                        validate=lambda _, x: x.isdigit() and int(x) > 0,
                        default=str(codebase_settings['max_file_size_mb'])
                    )
                ]

                size_answer = inquirer.prompt(size_question)
                if size_answer:
                    self.update_setting('codebase_search', 'max_file_size_mb', int(size_answer['size']))
                    self.console.print("[green]Maximum file size updated[/green]")

            elif answer['action'] == "exclude_patterns":
                patterns = codebase_settings['exclude_patterns']
                while True:
                    pattern_choices = [
                        ("Add New Pattern", "add"),
                        ("Remove Pattern", "remove"),
                        ("View Current Patterns", "view"),
                        ("Back", "back")
                    ]

                    pattern_question = [
                        inquirer.List('action',
                            message="Manage exclude patterns",
                            choices=pattern_choices,
                            carousel=True
                        ),
                    ]

                    pattern_answer = inquirer.prompt(pattern_question)
                    if not pattern_answer or pattern_answer['action'] == "back":
                        break

                    if pattern_answer['action'] == "add":
                        add_question = [
                            inquirer.Text('pattern',
                                message="Enter pattern to exclude (e.g., node_modules)",
                                validate=lambda _, x: len(x.strip()) > 0
                            )
                        ]

                        add_answer = inquirer.prompt(add_question)
                        if add_answer:
                            patterns.append(add_answer['pattern'].strip())
                            self.update_setting('codebase_search', 'exclude_patterns', patterns)
                            self.console.print("[green]Pattern added[/green]")

                    elif pattern_answer['action'] == "remove":
                        if not patterns:
                            self.console.print("[yellow]No patterns to remove[/yellow]")
                            continue

                        remove_choices = [(p, p) for p in patterns]
                        remove_choices.append(("Back", None))

                        remove_question = [
                            inquirer.List('pattern',
                                message="Select pattern to remove",
                                choices=remove_choices,
                                carousel=True
                            ),
                        ]

                        remove_answer = inquirer.prompt(remove_question)
                        if remove_answer and remove_answer['pattern']:
                            patterns.remove(remove_answer['pattern'])
                            self.update_setting('codebase_search', 'exclude_patterns', patterns)
                            self.console.print("[green]Pattern removed[/green]")

                    elif pattern_answer['action'] == "view":
                        if patterns:
                            self.console.print("\n[bold]Current exclude patterns:[/bold]")
                            for pattern in patterns:
                                self.console.print(f"• {pattern}")
                        else:
                            self.console.print("[yellow]No exclude patterns defined[/yellow]")
                        self.console.input("\nPress Enter to continue...")

class SystemInstructionsManager:
    """Class to manage system instructions for AI models"""
    def __init__(self, logger, console):
        self.logger = logger
        self.console = console
        self.instructions_file = os.path.join(os.path.dirname(__file__), 'system_instructions.json')
        self.instructions = self._load_instructions()

    def _load_instructions(self):
        """Load system instructions from file"""
        try:
            if os.path.exists(self.instructions_file):
                with open(self.instructions_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {
                'instructions': [
                    {
                        'name': 'Default',
                        'content': 'You are a helpful AI assistant. Provide clear, concise, and helpful responses.'
                    }
                ],
                'selected': 'Default'
            }
        except Exception as e:
            self.logger.error(f"Error loading instructions: {e}")
            return {'instructions': [], 'selected': None}

    def _save_instructions(self):
        """Save system instructions to file"""
        try:
            with open(self.instructions_file, 'w', encoding='utf-8') as f:
                json.dump(self.instructions, f, indent=4)
        except Exception as e:
            self.logger.error(f"Error saving instructions: {e}")

    def add_instruction(self, name, content):
        """Add a new system instruction"""
        if any(i['name'] == name for i in self.instructions['instructions']):
            return False, "Instruction with this name already exists"
        
        self.instructions['instructions'].append({
            'name': name,
            'content': content
        })
        self._save_instructions()
        return True, "Instruction added successfully"

    def remove_instruction(self, name):
        """Remove a system instruction"""
        if name == 'Default':
            return False, "Cannot remove default instruction"
        
        self.instructions['instructions'] = [
            i for i in self.instructions['instructions'] if i['name'] != name
        ]
        if self.instructions['selected'] == name:
            self.instructions['selected'] = 'Default'
        self._save_instructions()
        return True, "Instruction removed successfully"

    def select_instruction(self, name):
        """Select a system instruction as active"""
        if any(i['name'] == name for i in self.instructions['instructions']):
            self.instructions['selected'] = name
            self._save_instructions()
            return True, "Instruction selected successfully"
        return False, "Instruction not found"

    def get_selected_instruction(self):
        """Get the currently selected instruction content"""
        selected = self.instructions['selected']
        for instruction in self.instructions['instructions']:
            if instruction['name'] == selected:
                return instruction['content']
        # Return default if nothing is selected
        return "You are a helpful AI assistant. Provide clear, concise, and helpful responses."

    def list_instructions(self):
        """List all available instructions"""
        return self.instructions['instructions']

    def get_current_name(self):
        """Get the name of the currently selected instruction"""
        return self.instructions['selected']

# Add HTTP headers helper
def get_openrouter_headers(api_key):
    """Get required headers for OpenRouter API"""
    # Mask API key for logging
    masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "***"
    logging.getLogger("ACT").debug(f"Setting up OpenRouter headers with API key: {masked_key}")
    
    return {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/eplisium",
        "X-Title": "ACT",
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

    # Define log format with more details
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    detailed_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s"

    # Create formatters
    file_formatter = logging.Formatter(detailed_format, datefmt="%Y-%m-%d %H:%M:%S")

    # Create rotating file handler for general logs
    file_handler = RotatingFileHandler(
        os.path.join(logs_dir, f'act_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)  # File shows DEBUG and above

    # Create error file handler for error-only logs
    error_handler = RotatingFileHandler(
        os.path.join(logs_dir, f'act_error_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    error_handler.setFormatter(file_formatter)
    error_handler.setLevel(logging.ERROR)  # Only ERROR and CRITICAL

    # Create null handler for console to suppress output
    null_handler = logging.NullHandler()

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all levels
    root_logger.addHandler(null_handler)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)

    # Create ACT logger as a child of root logger
    logger = logging.getLogger("ACT")
    
    # Log system information at startup (only to files)
    logger.info("="*80)
    logger.info("ACT - AI Chat Terminal Starting")
    logger.info("-"*80)
    logger.info(f"Python Version: {sys.version}")
    logger.info(f"Operating System: {os.name} - {sys.platform}")
    logger.info(f"Log Directory: {os.path.abspath(logs_dir)}")
    logger.info("="*80)

    return logger, console

def sanitize_path(path):
    """
    Sanitize a path string to be valid on all operating systems
    
    Args:
        path (str): The path to sanitize
    
    Returns:
        str: The sanitized path
    """
    # Characters not allowed in file/directory names on most systems
    invalid_chars = '<>:"/\\|?*'
    
    # Replace invalid characters with underscores
    for char in invalid_chars:
        path = path.replace(char, '_')
    
    # Remove any leading/trailing periods or spaces
    path = path.strip('. ')
    
    # Ensure the path isn't empty after sanitization
    if not path:
        path = 'unnamed'
    
    return path

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

def read_document_content(file_path):
    """
    Read content from various document formats
    
    Args:
        file_path (str): Path to the document file
    
    Returns:
        tuple: (success, content or error message)
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        # Handle DOCX files
        if file_ext == '.docx':
            if not DOCX_AVAILABLE:
                return False, "python-docx library not installed. Install with: pip install python-docx"
            doc = docx.Document(file_path)
            content = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
            return True, content

        # Handle PDF files
        elif file_ext == '.pdf':
            if not PDF_AVAILABLE:
                return False, "PyPDF2 library not installed. Install with: pip install PyPDF2"
            reader = PdfReader(file_path)
            content = '\n'.join([page.extract_text() for page in reader.pages])
            return True, content

        # Handle ODT files
        elif file_ext == '.odt':
            if not ODF_AVAILABLE:
                return False, "odfpy library not installed. Install with: pip install odfpy"
            doc = odf.load(file_path)
            content = teletype.extractText(doc)
            return True, content

        # Handle RTF files
        elif file_ext == '.rtf':
            if not RTF_AVAILABLE:
                return False, "pyth library not installed. Install with: pip install pyth"
            with open(file_path, 'rb') as f:
                doc = rtf.Rtf(f)
                content = doc.getText()
                return True, content

        # Handle text files with different encodings
        else:
            encodings = ['utf-8', 'latin1', 'cp1252', 'ascii']
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                        return True, content
                except UnicodeDecodeError:
                    continue
            
            return False, f"Could not decode file with any of the following encodings: {', '.join(encodings)}"

    except Exception as e:
        return False, str(e)

def encode_image_to_base64(image_path_or_url):
    """
    Encode an image to base64 string, handling both local files and URLs
    
    Args:
        image_path_or_url (str): Local path or URL to the image
    
    Returns:
        tuple: (success, base64_string or error_message)
    """
    try:
        # Check if it's a URL
        parsed = urlparse(image_path_or_url)
        is_url = bool(parsed.scheme and parsed.netloc)
        
        if is_url:
            # Download image from URL
            response = requests.get(image_path_or_url)
            response.raise_for_status()  # Raise exception for bad status codes
            image_data = response.content
        else:
            # Read local file
            with open(image_path_or_url, 'rb') as image_file:
                image_data = image_file.read()
        
        # Validate image data
        try:
            img = Image.open(BytesIO(image_data))
            img.verify()  # Verify it's a valid image
        except Exception as e:
            return False, f"Invalid image data: {str(e)}"
        
        # Convert to base64
        base64_string = base64.b64encode(image_data).decode('utf-8')
        return True, base64_string
    
    except requests.exceptions.RequestException as e:
        return False, f"Failed to download image: {str(e)}"
    except Exception as e:
        return False, f"Failed to process image: {str(e)}"

def get_image_mime_type(image_path_or_url):
    """
    Get the MIME type of an image
    
    Args:
        image_path_or_url (str): Local path or URL to the image
    
    Returns:
        str: MIME type of the image
    """
    try:
        # Check if it's a URL
        parsed = urlparse(image_path_or_url)
        is_url = bool(parsed.scheme and parsed.netloc)
        
        if is_url:
            # Download image from URL
            response = requests.get(image_path_or_url)
            response.raise_for_status()
            image_data = response.content
        else:
            # Read local file
            with open(image_path_or_url, 'rb') as image_file:
                image_data = image_file.read()
        
        # Use PIL to determine image type
        img = Image.open(BytesIO(image_data))
        mime_type = f"image/{img.format.lower()}"
        return mime_type
    
    except Exception:
        # Default to generic image type if detection fails
        return "image/jpeg"

class AIChat:
    def __init__(self, model_config, logger, console, system_instruction=None):
        """
        Initialize AI Chat with specific model configuration
        
        Args:
            model_config (dict): Configuration for the selected model
            logger (logging.Logger): Logging instance
            console (rich.console.Console): Rich console for output
            system_instruction (str, optional): System instruction to use. Defaults to None.
        """
        self.logger = logger
        self.console = console
        dotenv.load_dotenv()

        self.model_id = model_config['id']
        self.model_name = model_config['name']
        self.provider = model_config.get('provider', 'openai')
        self.start_time = datetime.now()  # Add start time for chat
        
        # Initialize settings manager and system instructions manager
        self.settings_manager = SettingsManager(logger, console)
        self.instructions_manager = SystemInstructionsManager(logger, console)

        # Initialize client based on provider
        self._setup_client(model_config)

        # System prompt
        self.messages = [
            {
                "role": "system", 
                "content": system_instruction or "You are a helpful AI assistant. Provide clear, concise, and helpful responses."
            }
        ]
        
        self.logger.info(f"Initialized chat with model: {self.model_name}")
    
    def _setup_client(self, model_config):
        """
        Set up the appropriate client based on the model provider
        
        Args:
            model_config (dict): Configuration for the selected model
        """
        self.provider = model_config.get('provider', 'openai').lower()
        
        def mask_key(key):
            """Helper to mask API key for logging"""
            return f"{key[:4]}...{key[-4:]}" if len(key) > 8 else "***"
        
        if self.provider == 'openrouter':
            if not OPENROUTER_AVAILABLE:
                raise ImportError("OpenRouter library not installed. Please install 'openai'.")
            
            self.api_key = os.getenv('OPENROUTER_API_KEY')
            if not self.api_key:
                self.logger.error("OpenRouter API key not found")
                raise ValueError("OpenRouter API key not found in .env file")
            
            self.logger.debug(f"Initializing OpenRouter client with API key: {mask_key(self.api_key)}")
            self.api_base = "https://openrouter.ai/api/v1"
        
        elif self.provider == 'anthropic':
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("Anthropic library not installed. Please install 'anthropic'.")
            
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                self.logger.error("Anthropic API key not found")
                raise ValueError("Anthropic API key not found in .env file")
            
            self.logger.debug(f"Initializing Anthropic client with API key: {mask_key(api_key)}")
            self.client = Anthropic(api_key=api_key)
        
        else:  # openai
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI library not installed. Please install 'openai'.")
            
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                self.logger.error("OpenAI API key not found")
                raise ValueError("OpenAI API key not found in .env file")
            
            self.logger.debug(f"Initializing OpenAI client with API key: {mask_key(api_key)}")
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
            
            # Process image references before sending
            messages = self.messages.copy()
            current_message = {"role": "user", "content": user_input}
            
            # Check for image references
            if '[[img:' in user_input and ']]' in user_input:
                # Convert to list format for content
                content = []
                remaining_text = user_input
                
                while '[[img:' in remaining_text and ']]' in remaining_text:
                    # Find image reference
                    start = remaining_text.find('[[img:')
                    end = remaining_text.find(']]', start)
                    
                    if start > 0:
                        # Add text before image reference
                        content.append({
                            "type": "text",
                            "text": remaining_text[:start].strip()
                        })
                    
                    # Extract image path/url
                    img_ref = remaining_text[start+6:end].strip()
                    # Handle quoted paths
                    if img_ref.startswith('"') and img_ref.endswith('"'):
                        img_ref = img_ref[1:-1]
                    elif img_ref.startswith("'") and img_ref.endswith("'"):
                        img_ref = img_ref[1:-1]
                    
                    # Process image
                    success, result = encode_image_to_base64(img_ref)
                    if success:
                        if self.provider == 'anthropic':
                            content.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": get_image_mime_type(img_ref),
                                    "data": result
                                }
                            })
                        else:
                            content.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{get_image_mime_type(img_ref)};base64,{result}"
                                }
                            })
                    else:
                        self.logger.error(f"Failed to process image {img_ref}: {result}")
                        self.console.print(f"[bold red]Error processing image: {result}[/bold red]")
                    
                    # Move to remaining text
                    remaining_text = remaining_text[end+2:]
                
                # Add any remaining text
                if remaining_text.strip():
                    content.append({
                        "type": "text",
                        "text": remaining_text.strip()
                    })
                
                current_message["content"] = content
            
            # Add processed message to conversation history
            messages.append(current_message)
            # Also add to main message history
            self.messages.append(current_message)
            
            # Show thinking message
            thinking_message = self.console.status(f"[bold yellow]{self.model_name} is thinking...[/bold yellow]")
            thinking_message.start()
            
            try:
                if self.provider == 'anthropic':
                    # Convert messages to Anthropic format
                    anthropic_messages = []
                    for msg in messages[1:]:  # Skip system message
                        if isinstance(msg["content"], list):
                            # Handle multimodal content
                            anthropic_messages.append({
                                "role": "user" if msg["role"] == "user" else "assistant",
                                "content": msg["content"]
                            })
                        else:
                            anthropic_messages.append({
                                "role": "user" if msg["role"] == "user" else "assistant",
                                "content": msg["content"]
                            })

                    # Anthropic API call with max tokens
                    response = self.client.messages.create(
                        model=self.model_id,
                        messages=anthropic_messages,
                        system=self.messages[0]['content'],
                        max_tokens=4096,  # Maximum tokens for most Claude models
                        temperature=0.7
                    )
                    ai_response = response.content[0].text

                elif self.provider == 'openrouter':
                    # OpenRouter API call with max tokens
                    response = requests.post(
                        f"{self.api_base}/chat/completions",
                        headers=get_openrouter_headers(self.api_key),
                        json={
                            "model": self.model_id,
                            "messages": messages,
                            "temperature": 0.7,
                            "max_tokens": 4096,  # Maximum tokens
                            "stream": False
                        }
                    )
                    response.raise_for_status()
                    data = response.json()
                    ai_response = data['choices'][0]['message']['content'].strip()

                else:
                    # OpenAI API call with max tokens
                    response = self.client.chat.completions.create(
                        model=self.model_id,
                        messages=messages,
                        temperature=0.7,
                        max_tokens=4096,  # Maximum tokens for most GPT models
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

        # Process the response text to identify natural breaks (like paragraphs and lists)
        sections = []
        current_section = []
        lines = response_text.split('\n')
        
        for line in lines:
            # Check for natural breaks like empty lines or list markers
            if (not line.strip() and current_section) or \
               (line.strip().startswith(('•', '-', '*', '1.', '#')) and current_section):
                if current_section:
                    sections.append('\n'.join(current_section))
                    current_section = []
            if line.strip():  # Only add non-empty lines
                current_section.append(line)
        
        if current_section:
            sections.append('\n'.join(current_section))
        
        # If no clear sections, treat entire response as one section
        if not sections:
            sections = [response_text]
        
        # Create a single panel with enhanced formatting
        formatted_text = []
        for i, section in enumerate(sections):
            if i > 0:  # Add spacing between sections
                formatted_text.append("")  # Empty line for spacing
            formatted_text.append(section)
        
        # Join all sections with proper spacing
        final_text = '\n'.join(formatted_text)

        # Get the current system instruction name
        current_instruction_name = self.instructions_manager.get_current_name()
        
        # Create a single panel with the entire formatted content
        self.console.print(
            Panel(
                Markdown(final_text),
                title=f"[bold #A6E22E]{self.model_name}[/] [dim]({current_instruction_name})[/]",
                border_style="bright_blue",
                padding=(1, 2),
                expand=True,
                highlight=True
            )
        )
    
    def save_chat(self, custom_name=None):
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
                # Sanitize paths
                company = sanitize_path(company)
                model_name = sanitize_path(model_name)
                chat_dir = os.path.join(chats_dir, 'openrouter', company, model_name)
            else:
                # For other providers (e.g., OpenAI), just use provider name
                chat_dir = os.path.join(chats_dir, sanitize_path(self.provider))

            os.makedirs(chat_dir, exist_ok=True)

            # Base filename without extension
            if custom_name:
                # Sanitize custom name and combine with timestamp
                custom_name = sanitize_path(custom_name)
                base_filename = f"{custom_name}_{timestamp}"
            else:
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
                if custom_name:
                    f.write(f"Chat Name: {custom_name}\n")
                f.write(f"Model ID: {self.model_id}\n")
                f.write(f"Provider: {self.provider}\n")
                f.write(f"Date: {self.start_time.strftime('%Y-%m-%d %I:%M:%S %p')}\n")
                f.write("="*80 + "\n\n")

                # Write messages
                for msg in self.messages[1:]:  # Skip system message
                    role = msg['role'].upper()
                    content = msg['content']
                    
                    # Format role header with proper spacing
                    if role == "USER":
                        f.write("\nYOU:\n" + "-"*40 + "\n")
                    else:
                        f.write(f"\n{self.model_name}:\n" + "-"*40 + "\n")
                    
                    # Write message content with proper formatting
                    if isinstance(content, list):  # Handle multimodal content
                        for item in content:
                            if item['type'] == 'text':
                                f.write(item['text'] + "\n")
                            elif item['type'] in ['image', 'image_url']:
                                f.write("[Image embedded]\n")
                    else:
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
                    f.write("\n")

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
        
        # Store welcome panel text to avoid processing its example paths
        welcome_text = (
            f"[bold cyan]Chat session with {self.model_name}[/bold cyan]\n\n"
            "📝 Type your message and press Enter to send\n"
            "🔗 Reference files and directories:\n"
            "   [[ file:example.py]]          - View single file contents\n"
            '   [[ file:"path/to/file.txt"]]  - Paths with spaces need quotes\n'
            "   [[ dir:folder]]               - List directory contents\n"
            "   [[ codebase:folder]]          - View all code files in directory\n"
            '   [[ codebase:"src/*.py"]]      - View Python files in src folder\n'
            "💾️ Reference images:\n"
            "   [[ img:image.jpg]]            - Include local image\n"
            '   [[ img:"path/to/image.png"]]  - Paths with spaces need quotes\n'
            '   [[ img:https://...]]          - Include image from URL\n'
            "💾 Commands:\n"
            "   - /save [name] - Save the chat history (optional custom name)\n"
            "   - /clear - Clear the screen and chat history\n"
            "   - /insert - Insert multiline text (end with END on new line)\n"
            "   - /end - End the chat session\n"
            "❌ Type 'exit', 'quit', 'bye', or press Ctrl+C to end the session"
        )
        
        self.console.print(
            Panel(
                welcome_text,
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
                        command_parts = user_input.strip().split(maxsplit=1)
                        command = command_parts[0].lower()
                        command_args = command_parts[1] if len(command_parts) > 1 else None

                        if command == '/save':
                            if len(self.messages) > 1:
                                if command_args:
                                    if self.save_chat(command_args):
                                        self.console.print("[bold green]Chat history saved successfully with custom name![/bold green]")
                                    else:
                                        self.console.print("[bold red]Failed to save chat history.[/bold red]")
                                else:
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
                                    welcome_text,
                                    title="[bold white]Chat Session[/bold white]",
                                    border_style="cyan",
                                    padding=(1, 2)
                                )
                            )
                            self.console.print("[bold green]Screen and chat history cleared![/bold green]")
                            continue
                        elif command == '/insert':
                            
                            
                            # Show instructions for multiline input
                            self.console.print(
                                Panel(
                                    "[bold cyan]Enter your text below:[/bold cyan]\n"
                                    "• You can paste multiple lines of text\n"
                                    "• Press [bold]Enter[/bold] twice to start a new line\n"
                                    "• Type [bold]END[/bold] on a new line and press Enter to finish\n"
                                    "• Type [bold]CANCEL[/bold] on a new line to cancel",
                                    title="[bold white]Multiline Input[/bold white]",
                                    border_style="cyan"
                                )
                            )
                            
                            # Collect multiline input
                            content_lines = []
                            try:
                                while True:
                                    line = input()
                                    if line.strip().upper() == 'END':
                                        break
                                    if line.strip().upper() == 'CANCEL':
                                        content_lines = []
                                        break
                                    content_lines.append(line)
                            except KeyboardInterrupt:
                                self.console.print("\n[yellow]Input cancelled[/yellow]")
                                continue
                            
                            # Process the input
                            if content_lines:
                                user_input = '\n'.join(content_lines)
                            else:
                                self.console.print("[yellow]No content provided, input cancelled[/yellow]")
                                continue
                        elif command == '/end':
                            self.logger.info("Chat session ended by user (/end command)")
                            self.console.print("[bold cyan]Chat session ended. Goodbye![/bold cyan]")
                            break
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
                                                # Get file extension for syntax highlighting
                                                ext = os.path.splitext(path)[1][1:] or 'text'
                                                
                                                # Read file contents based on type
                                                success, content = read_document_content(path)
                                                if not success:
                                                    raise Exception(content)  # content contains error message
                                                
                                                # Format the content with file info
                                                formatted_content = (
                                                    f"\nContents of {os.path.basename(path)}:\n"
                                                    f"```{ext}\n"
                                                    f"{content}\n"
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
                                        # Get enabled file types from settings
                                        enabled_types = self.settings_manager.get_setting('codebase_search', 'enabled_types')
                                        return os.path.splitext(filename)[1].lower() in enabled_types
                                    
                                    def collect_code_files(path):
                                        """Recursively collect code and documentation files from directory"""
                                        code_files = []
                                        try:
                                            # Get settings
                                            search_subdirs = self.settings_manager.get_setting('codebase_search', 'search_subdirectories')
                                            max_size = self.settings_manager.get_setting('codebase_search', 'max_file_size_mb') * 1024 * 1024  # Convert to bytes
                                            exclude_patterns = self.settings_manager.get_setting('codebase_search', 'exclude_patterns')
                                            
                                            if os.path.isfile(path):
                                                if is_code_file(path) and os.path.getsize(path) <= max_size:
                                                    code_files.append(path)
                                            elif os.path.isdir(path):
                                                # Function to check if path should be excluded
                                                def should_exclude(p):
                                                    for pattern in exclude_patterns:
                                                        if pattern in p:
                                                            return True
                                                    return False
                                                
                                                if search_subdirs:
                                                    # Walk through directory recursively
                                                    for root, dirs, files in os.walk(path):
                                                        # Skip excluded directories
                                                        dirs[:] = [d for d in dirs if not should_exclude(os.path.join(root, d))]
                                                        
                                                        for file in sorted(files):
                                                            file_path = os.path.join(root, file)
                                                            if (is_code_file(file) and 
                                                                not should_exclude(file_path) and 
                                                                os.path.getsize(file_path) <= max_size):
                                                                code_files.append(file_path)
                                                else:
                                                    # Only look at files in the current directory
                                                    for file in sorted(os.listdir(path)):
                                                        file_path = os.path.join(path, file)
                                                        if (os.path.isfile(file_path) and 
                                                            is_code_file(file) and 
                                                            not should_exclude(file_path) and 
                                                            os.path.getsize(file_path) <= max_size):
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
        
        # Initialize managers
        self.instructions_manager = SystemInstructionsManager(logger, console)
        self.settings_manager = SettingsManager(logger, console)
        
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
    
    def manage_instructions(self):
        """Display system instructions management menu"""
        while True:
            # Get current instructions
            current_name = self.instructions_manager.get_current_name()
            instructions = self.instructions_manager.list_instructions()
            
            # Create choices for instructions menu
            choices = [
                ("=== System Instructions ===", None),
                (f"Currently Selected: {current_name}", None),
                ("Add New Instruction", "add"),
                ("Select Instruction", "select"),
                ("Remove Instruction", "remove"),
                ("View Instructions", "view"),
                ("Back to Main Menu", "back")
            ]
            
            questions = [
                inquirer.List('action',
                    message="Manage System Instructions",
                    choices=choices,
                    carousel=True
                ),
            ]
            
            answer = inquirer.prompt(questions)
            if not answer or answer['action'] == "back":
                break
            
            if answer['action'] == "add":
                # Get instruction name first
                name_question = [
                    inquirer.Text('name',
                        message="Enter instruction name",
                        validate=lambda _, x: len(x) > 0
                    )
                ]
                
                name_answer = inquirer.prompt(name_question)
                if not name_answer:
                    continue
                
                # Clear screen for better visibility
                os.system('cls' if os.name == 'nt' else 'clear')
                
                # Show instructions for multiline input
                self.console.print(
                    Panel(
                        "[bold cyan]Enter your system instruction below:[/bold cyan]\n"
                        "• You can paste multiple lines of text\n"
                        "• Press [bold]Enter[/bold] twice to start a new line\n"
                        "• Type [bold]END[/bold] on a new line and press Enter to finish\n"
                        "• Type [bold]CANCEL[/bold] on a new line to cancel",
                        title="[bold white]Instruction Input[/bold white]",
                        border_style="cyan"
                    )
                )
                
                # Collect multiline input
                content_lines = []
                try:
                    while True:
                        line = input()
                        if line.strip().upper() == 'END':
                            break
                        if line.strip().upper() == 'CANCEL':
                            content_lines = []
                            break
                        content_lines.append(line)
                except KeyboardInterrupt:
                    self.console.print("\n[yellow]Input cancelled[/yellow]")
                    continue
                
                # Process the input
                if content_lines:
                    content = '\n'.join(content_lines)
                    
                    # Show preview
                    self.console.print("\n[bold cyan]Preview of your instruction:[/bold cyan]")
                    self.console.print(
                        Panel(
                            content,
                            title=f"[bold white]{name_answer['name']}[/bold white]",
                            border_style="cyan"
                        )
                    )
                    
                    # Confirm save
                    confirm_question = [
                        inquirer.Confirm('save',
                            message="Save this instruction?",
                            default=True
                        )
                    ]
                    
                    if inquirer.prompt(confirm_question)['save']:
                        success, msg = self.instructions_manager.add_instruction(
                            name_answer['name'],
                            content
                        )
                        self.console.print(f"[{'green' if success else 'red'}]{msg}[/]")
                    else:
                        self.console.print("[yellow]Instruction not saved[/yellow]")
                else:
                    self.console.print("[yellow]No content provided, instruction not saved[/yellow]")
            
            elif answer['action'] == "select":
                if not instructions:
                    self.console.print("[yellow]No instructions available[/yellow]")
                    continue
                
                # Create choices for instruction selection
                instruction_choices = [
                    (f"{i['name']}", i['name']) for i in instructions
                ]
                instruction_choices.append(("Back", None))
                
                select_question = [
                    inquirer.List('instruction',
                        message="Select Instruction",
                        choices=instruction_choices,
                        carousel=True
                    ),
                ]
                
                select_answer = inquirer.prompt(select_question)
                if select_answer and select_answer['instruction']:
                    success, msg = self.instructions_manager.select_instruction(
                        select_answer['instruction']
                    )
                    self.console.print(f"[{'green' if success else 'red'}]{msg}[/]")
            
            elif answer['action'] == "remove":
                if not instructions:
                    self.console.print("[yellow]No instructions available[/yellow]")
                    continue
                
                # Create choices for instruction removal
                instruction_choices = [
                    (f"{i['name']}", i['name']) 
                    for i in instructions 
                    if i['name'] != 'Default'  # Prevent removal of default instruction
                ]
                instruction_choices.append(("Back", None))
                
                remove_question = [
                    inquirer.List('instruction',
                        message="Select Instruction to Remove",
                        choices=instruction_choices,
                        carousel=True
                    ),
                ]
                
                remove_answer = inquirer.prompt(remove_question)
                if remove_answer and remove_answer['instruction']:
                    success, msg = self.instructions_manager.remove_instruction(
                        remove_answer['instruction']
                    )
                    self.console.print(f"[{'green' if success else 'red'}]{msg}[/]")
            
            elif answer['action'] == "view":
                if not instructions:
                    self.console.print("[yellow]No instructions available[/yellow]")
                    continue
                
                # Display all instructions
                for instruction in instructions:
                    is_selected = instruction['name'] == current_name
                    self.console.print(
                        Panel(
                            f"[bold]Content:[/bold]\n{instruction['content']}",
                            title=f"[{'green' if is_selected else 'white'}]{instruction['name']}{'  [Selected]' if is_selected else ''}[/]",
                            border_style="green" if is_selected else "white"
                        )
                    )
                
                # Wait for user acknowledgment
                self.console.input("\nPress Enter to continue...")

    def manage_settings(self):
        """Display settings management menu"""
        while True:
            choices = [
                ("=== Application Settings ===", None),
                ("🔍 Codebase Search Settings", "codebase"),
                ("Back to Main Menu", "back")
            ]

            questions = [
                inquirer.List('setting',
                    message="Select settings category",
                    choices=choices,
                    carousel=True
                ),
            ]

            answer = inquirer.prompt(questions)
            if not answer or answer['setting'] == "back":
                break

            if answer['setting'] == "codebase":
                self.settings_manager.manage_codebase_settings()

    def manage_ai_settings(self):
        """Display AI settings management menu"""
        while True:
            choices = [
                ("=== AI Settings ===", None),
                ("🤖 System Instructions", "instructions"),
                # Future AI settings can be added here
                ("Back to Main Menu", "back")
            ]

            questions = [
                inquirer.List('setting',
                    message="Select AI setting to configure",
                    choices=choices,
                    carousel=True
                ),
            ]

            answer = inquirer.prompt(questions)
            if not answer or answer['setting'] == "back":
                break

            if answer['setting'] == "instructions":
                self.manage_instructions()

    def display_main_menu(self):
        """
        Display the main menu for model selection
        """
        self.logger.info("Displaying main menu")
        
        # Display welcome message with ASCII art logo
        logo = """[bold cyan]
    ╔═══╗╔═══╗╔════╗
    ║╔═╗║║╔═╗║║╔╗╔╗║
    ║║ ║║║║ ╚╝╚╝║║╚╝
    ║╔═╗║║║ ╔╗  ║║  
    ║║ ║║║╚═╝║  ║║  
    ╚╝ ╚╝╚═══╝  ╚╝  
[/bold cyan]"""
        
        welcome_text = (
            f"{logo}\n"
            "[bold white]Welcome to[/bold white] [bold cyan]ACT[/bold cyan] [bold white](AI Chat Terminal)[/bold white]\n\n"
            "[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]\n"
            "🤖 [bold]Your Gateway to Advanced AI Conversations[/bold]\n"
            "📝 Choose from multiple AI models and providers\n"
            "💡 Each model offers unique capabilities\n"
            "⚡ Powered by OpenAI, Anthropic, and OpenRouter\n"
            "[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]\n\n"
            "[dim]Press Ctrl+C at any time to exit[/dim]"
        )
        
        self.console.print(Panel(
            welcome_text,
            title="[bold white]✨ ACT - Advanced AI Chat Terminal ✨[/bold white]",
            border_style="cyan",
            padding=(1, 2),
            highlight=True
        ))
        
        while True:
            try:
                # Check API keys availability
                openai_available = bool(os.getenv('OPENAI_API_KEY'))
                openrouter_available = bool(os.getenv('OPENROUTER_API_KEY'))
                anthropic_available = bool(os.getenv('ANTHROPIC_API_KEY'))
                
                # Main menu choices with enhanced formatting
                main_choices = [
                    ("═══ Select Your AI Provider ═══", None),
                    ("★ Favorite Models   〈Your preferred AI companions〉", "favorites"),
                ]
                
                # Add providers based on API key availability with status indicators
                if openai_available:
                    main_choices.append(("🟢 OpenAI Models    〈GPT-4, GPT-3.5 & More〉", "openai"))
                else:
                    main_choices.append(("○ OpenAI Models    〈API Key Required〉", None))
                
                if openrouter_available:
                    main_choices.append(("🟢 OpenRouter Models 〈Multiple Providers〉", "openrouter"))
                else:
                    main_choices.append(("○ OpenRouter Models 〈API Key Required〉", None))
                
                if anthropic_available:
                    main_choices.append(("🟢 Anthropic Models 〈Claude & More〉", "anthropic"))
                else:
                    main_choices.append(("○ Anthropic Models 〈API Key Required〉", None))
                
                # Add remaining menu items with enhanced formatting
                main_choices.extend([
                    ("═══ System Settings ═══", None),
                    ("⚙️ AI Settings       〈Configure AI Behavior〉", "ai_settings"),
                    ("⚙️ Application Settings 〈Configure App Behavior〉", "settings"),
                    ("═══ Application ═══", None),
                    ("✖ Exit Application    〈Close ACT〉", "exit")
                ])

                # Create custom theme for inquirer
                theme = themes.load_theme_from_dict({
                    "Question": {
                        "mark_color": "cyan",
                        "brackets_color": "cyan",
                        "default_color": "white"
                    },
                    "List": {
                        "selection_color": "cyan",
                        "selection_cursor": "❯",
                        "unselected_color": "white"
                    }
                })
                
                # Create the main menu question with custom theme
                questions = [
                    inquirer.List('provider',
                        message="Select an option:",
                        choices=main_choices,
                        carousel=True
                    ),
                ]
                
                # Prompt user for provider selection with custom theme
                answer = inquirer.prompt(questions, theme=theme)
                
                if not answer or answer['provider'] == "exit":
                    self.logger.info("User selected to exit")
                    self.console.print(Panel(
                        "[bold yellow]Thank you for using ACT!\n\n"
                        "We hope to see you again soon![/bold yellow]",
                        title="[bold white]👋 Goodbye![/bold white]",
                        border_style="yellow",
                        padding=(1, 2)
                    ))
                    break
                
                selected_provider = answer['provider']

                if selected_provider == "instructions":
                    self.manage_instructions()
                    continue
                elif selected_provider == "settings":
                    self.manage_settings()
                    continue
                elif selected_provider == "favorites":
                    self.manage_favorites()
                    continue
                elif selected_provider == "ai_settings":
                    self.manage_ai_settings()
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
                        self._handle_model_selection(selected_model)
                
                elif selected_provider == "anthropic":
                    # Filter and display Anthropic models
                    anthropic_models = [
                        (f"{model['name']} - {model.get('description', 'No description')}", model)
                        for model in self.models_config
                        if model.get('provider', '').lower() == 'anthropic'
                    ]
                    anthropic_models.append(("Back", None))
                    
                    model_question = [
                        inquirer.List('model',
                            message="Select Anthropic Model",
                            choices=anthropic_models,
                            carousel=True
                        ),
                    ]
                    
                    model_answer = inquirer.prompt(model_question)
                    if model_answer and model_answer['model'] is not None:
                        selected_model = model_answer['model']
                        self._handle_model_selection(selected_model)
                
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
                        self._handle_model_selection(model_config)
            
            except KeyboardInterrupt:
                self.logger.warning("Application interrupted")
                self.console.print(Panel(
                    "\n[bold yellow]Thanks for using ACT! Goodbye![/bold yellow]",
                    border_style="yellow"
                ))
                break
            except Exception as e:
                self.logger.error(f"Menu error: {e}", exc_info=True)
                self.console.print(f"[bold red]Error in menu: {e}[/bold red]")
                continue
    
    def _handle_model_selection(self, selected_model):
        """
        Handle the model selection and subsequent actions
        
        Args:
            selected_model (dict): The selected model configuration
        """
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

    def start_chat(self, model_config):
        """
        Start chat with selected model
        
        Args:
            model_config (dict): Configuration for the selected model
        """
        try:
            # Get the current system instruction
            system_instruction = self.instructions_manager.get_selected_instruction()
            
            # Initialize and start chat with selected model
            chat = AIChat(model_config, self.logger, self.console, system_instruction)
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