from imports import *
from utils import log_api_response, encode_image_to_base64, get_image_mime_type, read_document_content, sanitize_path

def get_openrouter_headers(api_key):
    """Get required headers for OpenRouter API"""
    return {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/eplisium",
        "X-Title": "ACT",
        "Content-Type": "application/json"
    }

class OpenRouterAPI:
    """Class to handle OpenRouter API interactions"""
    def __init__(self, api_key):
        self.api_key = api_key
        self.api_base = "https://openrouter.ai/api/v1"
        self.headers = get_openrouter_headers(api_key)
    
    def fetch_models(self):
        """Fetch available models from OpenRouter API"""
        try:
            response = requests.get(
                f"{self.api_base}/models",
                headers=self.headers
            )
            
            if response.status_code != 200:
                raise Exception(f"API returned status code {response.status_code}: {response.text}")
            
            data = response.json()
            if not data or 'data' not in data:
                raise Exception(f"Unexpected API response format: {data}")
            
            models = data['data']
            if not models:
                raise Exception("No models returned from API")
            
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
            model_id = model['id']
            if '/' in model_id:
                company = model_id.split('/')[0].title()
            else:
                company = 'Other'
            company_models[company].append(model)
        
        for company in company_models:
            company_models[company].sort(
                key=lambda x: (
                    not x.get('top_provider', False),
                    x['name'].lower(),
                    x.get('created_at', '0')
                )
            )
        
        return dict(company_models)

class AIChat:
    def __init__(self, model_config, logger, console, system_instruction=None, settings_manager=None):
        """Initialize AI Chat with specific model configuration"""
        self.logger = logger
        self.console = console
        self.settings_manager = settings_manager
        dotenv.load_dotenv()

        self.model_id = model_config['id']
        self.model_name = model_config['name']
        self.provider = model_config.get('provider', 'openai')
        self.start_time = datetime.now()
        
        # Handle system instruction name and content
        if isinstance(system_instruction, dict):
            self.instruction_name = system_instruction.get('name', 'Default')
            self.instruction_content = system_instruction.get('content', '')
        else:
            self.instruction_name = 'Default'
            self.instruction_content = system_instruction if system_instruction else "You are a helpful AI assistant. Provide clear, concise, and helpful responses."
        
        self._setup_client(model_config)

        self.messages = [
            {
                "role": "system", 
                "content": self.instruction_content
            }
        ]
        
        self.logger.info(f"Initialized chat with model: {self.model_name} using instruction: {self.instruction_name}")
    
    def _setup_client(self, model_config):
        """Set up the appropriate client based on the model provider"""
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

    def _process_file_reference(self, file_ref):
        """Process a file reference and return its contents"""
        try:
            # Remove [[ and ]] and file: prefix
            file_path = file_ref.strip('[]').replace('file:', '').strip()
            
            # Remove quotes if present
            if (file_path.startswith('"') and file_path.endswith('"')) or \
               (file_path.startswith("'") and file_path.endswith("'")):
                file_path = file_path[1:-1]
            
            # Make path absolute if relative
            if not os.path.isabs(file_path):
                file_path = os.path.join(os.getcwd(), file_path)
            
            # Check if file exists
            if not os.path.exists(file_path):
                return False, f"File not found: {file_path}"
            
            # Read file content based on extension
            success, content = read_document_content(file_path)
            if success:
                return True, f"Contents of {os.path.basename(file_path)}:\n\n{content}"
            else:
                return False, content
        except Exception as e:
            return False, f"Error reading file: {str(e)}"

    def _process_directory_reference(self, dir_ref):
        """Process a directory reference and return its contents"""
        try:
            # Remove [[ and ]] and dir: prefix
            dir_path = dir_ref.strip('[]').replace('dir:', '').strip()
            
            # Remove quotes if present
            if (dir_path.startswith('"') and dir_path.endswith('"')) or \
               (dir_path.startswith("'") and dir_path.endswith("'")):
                dir_path = dir_path[1:-1]
            
            # Make path absolute if relative
            if not os.path.isabs(dir_path):
                dir_path = os.path.join(os.getcwd(), dir_path)
            
            # Check if directory exists
            if not os.path.exists(dir_path):
                return False, f"Directory not found: {dir_path}"
            
            if not os.path.isdir(dir_path):
                return False, f"Not a directory: {dir_path}"
            
            # Get directory contents
            try:
                contents = os.listdir(dir_path)
                # Sort contents with directories first, then files
                dirs = []
                files = []
                for item in contents:
                    full_path = os.path.join(dir_path, item)
                    if os.path.isdir(full_path):
                        dirs.append(f"📁 {item}/")
                    else:
                        files.append(f"📄 {item}")
                
                dirs.sort()
                files.sort()
                formatted_contents = dirs + files
                
                if formatted_contents:
                    content_str = "\n".join(formatted_contents)
                    return True, f"Contents of directory {os.path.basename(dir_path)}:\n\n{content_str}"
                else:
                    return True, f"Directory {os.path.basename(dir_path)} is empty"
                
            except Exception as e:
                return False, f"Error reading directory: {str(e)}"
                
        except Exception as e:
            return False, f"Error processing directory: {str(e)}"

    def send_message(self, user_input):
        """Send user message and get AI response"""
        try:
            if not user_input.strip():
                self.logger.warning("Empty user input")
                return None
            
            messages = self.messages.copy()
            current_message = {"role": "user", "content": user_input}
            
            # Process directory references
            if '[[dir:' in user_input and ']]' in user_input:
                processed_parts = []
                remaining_text = user_input
                
                while '[[dir:' in remaining_text and ']]' in remaining_text:
                    start = remaining_text.find('[[dir:')
                    end = remaining_text.find(']]', start) + 2
                    
                    # Add text before the directory reference
                    if start > 0:
                        processed_parts.append(remaining_text[:start])
                    
                    # Process directory reference
                    dir_ref = remaining_text[start:end]
                    success, content = self._process_directory_reference(dir_ref)
                    if success:
                        processed_parts.append(content)
                    else:
                        processed_parts.append(f"[Error reading directory: {content}]")
                    
                    remaining_text = remaining_text[end:]
                
                # Add any remaining text
                if remaining_text:
                    processed_parts.append(remaining_text)
                
                # Join all parts
                current_message["content"] = '\n'.join(processed_parts)
            
            # Process file references
            elif '[[file:' in user_input and ']]' in user_input:
                processed_parts = []
                remaining_text = user_input
                
                while '[[file:' in remaining_text and ']]' in remaining_text:
                    start = remaining_text.find('[[file:')
                    end = remaining_text.find(']]', start) + 2
                    
                    # Add text before the file reference
                    if start > 0:
                        processed_parts.append(remaining_text[:start])
                    
                    # Process file reference
                    file_ref = remaining_text[start:end]
                    success, content = self._process_file_reference(file_ref)
                    if success:
                        processed_parts.append(content)
                    else:
                        processed_parts.append(f"[Error reading file: {content}]")
                    
                    remaining_text = remaining_text[end:]
                
                # Add any remaining text
                if remaining_text:
                    processed_parts.append(remaining_text)
                
                # Join all parts
                current_message["content"] = '\n'.join(processed_parts)

            # Process image references
            elif '[[img:' in user_input and ']]' in user_input:
                content = []
                remaining_text = user_input
                
                while '[[img:' in remaining_text and ']]' in remaining_text:
                    start = remaining_text.find('[[img:')
                    end = remaining_text.find(']]', start)
                    
                    if start > 0:
                        content.append({
                            "type": "text",
                            "text": remaining_text[:start].strip()
                        })
                    
                    img_ref = remaining_text[start+6:end].strip()
                    if img_ref.startswith('"') and img_ref.endswith('"'):
                        img_ref = img_ref[1:-1]
                    elif img_ref.startswith("'") and img_ref.endswith("'"):
                        img_ref = img_ref[1:-1]
                    
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
                    
                    remaining_text = remaining_text[end+2:]
                
                if remaining_text.strip():
                    content.append({
                        "type": "text",
                        "text": remaining_text.strip()
                    })
                
                current_message["content"] = content

            messages.append(current_message)
            self.messages.append(current_message)
            
            thinking_message = self.console.status(f"[bold yellow]{self.model_name} is thinking...[/bold yellow]")
            thinking_message.start()
            
            try:
                if self.provider == 'anthropic':
                    anthropic_messages = []
                    for msg in messages[1:]:
                        if isinstance(msg["content"], list):
                            anthropic_messages.append({
                                "role": "user" if msg["role"] == "user" else "assistant",
                                "content": msg["content"]
                            })
                        else:
                            anthropic_messages.append({
                                "role": "user" if msg["role"] == "user" else "assistant",
                                "content": msg["content"]
                            })

                    request_data = {
                        "model": self.model_id,
                        "messages": anthropic_messages,
                        "system": self.messages[0]['content'],
                        "max_tokens": 4096,
                        "temperature": 0.7
                    }

                    response = self.client.messages.create(**request_data)
                    log_api_response("Anthropic", request_data, response)
                    ai_response = response.content[0].text

                elif self.provider == 'openrouter':
                    request_data = {
                        "model": self.model_id,
                        "messages": messages,
                        "temperature": 0.7,
                        "max_tokens": 4096,
                        "stream": False
                    }

                    response = requests.post(
                        f"{self.api_base}/chat/completions",
                        headers=get_openrouter_headers(self.api_key),
                        json=request_data
                    )
                    response.raise_for_status()
                    data = response.json()
                    
                    log_api_response("OpenRouter", request_data, data)
                    ai_response = data['choices'][0]['message']['content'].strip()

                else:  # OpenAI
                    request_data = {
                        "model": self.model_id,
                        "messages": messages,
                        "temperature": 0.7,
                        "max_tokens": 4096,
                        "stream": False
                    }

                    response = self.client.chat.completions.create(**request_data)
                    log_api_response("OpenAI", request_data, response)
                    ai_response = response.choices[0].message.content.strip()

            except Exception as e:
                log_api_response(self.provider, request_data, None, error=e)
                raise
            finally:
                thinking_message.stop()
            
            self._display_response(ai_response)
            self.messages.append({"role": "assistant", "content": ai_response})
            
            return ai_response
        
        except Exception as e:
            self.logger.error(f"API Error: {e}", exc_info=True)
            self.console.print(f"[bold red]Error: {e}[/bold red]")
            return None

    def _get_colors(self):
        """Get colors from settings or use defaults"""
        try:
            if self.settings_manager:
                appearance = self.settings_manager.get_setting('appearance')
                return {
                    'ai_name': appearance.get('ai_name_color', '#A6E22E'),
                    'instruction_name': appearance.get('instruction_name_color', '#FFD700')
                }
            return {
                'ai_name': '#A6E22E',  # Default lime green
                'instruction_name': '#FFD700'  # Default gold
            }
        except Exception as e:
            self.logger.error(f"Error getting colors: {e}")
            return {
                'ai_name': '#A6E22E',  # Default lime green
                'instruction_name': '#FFD700'  # Default gold
            }

    def _display_response(self, response_text):
        """Format and display the AI response with enhanced markdown and code formatting"""
        if not response_text:
            return

        sections = []
        current_section = []
        lines = response_text.split('\n')
        
        for line in lines:
            if (not line.strip() and current_section) or \
               (line.strip().startswith(('•', '-', '*', '1.', '#')) and current_section):
                if current_section:
                    sections.append('\n'.join(current_section))
                    current_section = []
            if line.strip():
                current_section.append(line)
        
        if current_section:
            sections.append('\n'.join(current_section))
        
        if not sections:
            sections = [response_text]
        
        formatted_text = []
        for i, section in enumerate(sections):
            if i > 0:
                formatted_text.append("")
            formatted_text.append(section)
        
        final_text = '\n'.join(formatted_text)

        # Get colors from settings
        colors = self._get_colors()
        
        self.console.print(
            Panel(
                Markdown(final_text),
                title=f"[bold {colors['ai_name']}]{self.model_name}[/] [bold {colors['instruction_name']}][{self.instruction_name}][/]",
                border_style="bright_blue",
                padding=(1, 2),
                expand=True,
                highlight=True
            )
        )
    
    def save_chat(self, custom_name=None):
        """Save the chat conversation to both JSON and text files"""
        try:
            chats_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chats')
            os.makedirs(chats_dir, exist_ok=True)

            timestamp = self.start_time.strftime("%Y%m%d_%I%M%p")

            if self.provider == 'openrouter':
                company, model_name = self.model_id.split('/')
                company = sanitize_path(company)
                model_name = sanitize_path(model_name)
                chat_dir = os.path.join(chats_dir, 'openrouter', company, model_name)
            else:
                chat_dir = os.path.join(chats_dir, sanitize_path(self.provider))

            os.makedirs(chat_dir, exist_ok=True)

            if custom_name:
                custom_name = sanitize_path(custom_name)
                base_filename = f"{custom_name}_{timestamp}"
            else:
                base_filename = f"chat_{timestamp}"
            
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

            text_filepath = os.path.join(chat_dir, f"{base_filename}.txt")
            with open(text_filepath, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write(f"Chat with {self.model_name}\n")
                if custom_name:
                    f.write(f"Chat Name: {custom_name}\n")
                f.write(f"Model ID: {self.model_id}\n")
                f.write(f"Provider: {self.provider}\n")
                f.write(f"Date: {self.start_time.strftime('%Y-%m-%d %I:%M:%S %p')}\n")
                f.write("="*80 + "\n\n")

                for msg in self.messages[1:]:
                    role = msg['role'].upper()
                    content = msg['content']
                    
                    if role == "USER":
                        f.write("\nYOU:\n" + "-"*40 + "\n")
                    else:
                        f.write(f"\n{self.model_name}:\n" + "-"*40 + "\n")
                    
                    if isinstance(content, list):
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
                                    f.write("-"*80 + "\n")
                                    in_code_block = False
                                else:
                                    f.write("-"*80 + "\n")
                                    lang = line[3:].strip()
                                    if lang:
                                        f.write(f"Code ({lang}):\n")
                                    in_code_block = True
                            else:
                                if in_code_block:
                                    f.write("    " + line + "\n")
                                else:
                                    f.write(line + "\n")
                    
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
        """Main chat interaction loop"""
        self.logger.info(f"Starting chat loop for {self.model_name}")
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        def display_welcome():
            """Display welcome message and menu"""
            # Display ACT logo
            logo = """[bold cyan]
    ╔═══╗╔═══╗╔════╗
    ║╔═╗║║╔═╗║║╔╗╔╗║
    ║║ ║║║║ ╚╝╚╝║║╚╝
    ║╔═╗║║║ ╔╗  ║║  
    ║║ ║║║╚═╝║  ║║  
    ╚╝ ╚╝╚═══╝  ╚╝  
[/bold cyan]"""
            
            # Get colors from settings
            colors = self._get_colors()
            
            welcome_text = (
                f"{logo}\n"
                f"[bold {colors['ai_name']}]{self.model_name}[/] [bold {colors['instruction_name']}][{self.instruction_name}][/]\n\n"
                "📝 Type your message and press Enter to send\n"
                "🔗 Reference files and directories:\n"
                "   [[ file:example.py]]          - View single file contents\n"
                '   [[ file:"path/to/file.txt"]]  - Paths with spaces need quotes\n'
                "   [[ dir:folder]]               - List directory contents\n"
                '   [[ dir:"path/to/folder"]]     - Paths with spaces need quotes\n'
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
                    title="[bold white]✨ ACT - Advanced AI Chat Terminal ✨[/bold white]",
                    border_style="cyan",
                    padding=(1, 2)
                )
            )

        def display_main_menu():
            """Display the main ACT menu"""
            logo = """[bold cyan]
    ╔═══╗╔═══╗╔════╗
    ║╔═╗║║╔═╗║║╔╗╔╗║
    ║║ ║║║║ ╚╝╚╝║║╚╝
    ║╔═╗║║║ ╔╗  ║║  
    ║║ ║║║╚═╝║  ║║  
    ╚╝ ╚╝╚═══╝  ╚╝  
[/bold cyan]"""
            
            menu_text = (
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
            
            self.console.print(
                Panel(
                    menu_text,
                    title="[bold white]✨ ACT - Advanced AI Chat Terminal ✨[/bold white]",
                    border_style="cyan",
                    padding=(1, 2)
                )
            )
        
        # Clear screen and show welcome message at start
        os.system('cls' if os.name == 'nt' else 'clear')
        display_welcome()
        
        try:
            while True:
                try:
                    user_input = self.console.input("[bold yellow]You: [/bold yellow]")
                    
                    if user_input.lower() in ['exit', 'quit', 'bye']:
                        self.logger.info("Chat session ended by user")
                        self.console.print("[bold cyan]Goodbye![/bold cyan]")
                        # Clear screen and show main menu
                        os.system('cls' if os.name == 'nt' else 'clear')
                        display_main_menu()
                        break

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
                            os.system('cls' if os.name == 'nt' else 'clear')
                            self.messages = [self.messages[0]]  # Keep system message
                            display_welcome()
                            self.console.print("[bold green]Screen and chat history cleared![/bold green]")
                            continue
                        elif command == '/insert':
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
                            
                            if content_lines:
                                user_input = '\n'.join(content_lines)
                            else:
                                self.console.print("[yellow]No content provided, input cancelled[/yellow]")
                                continue
                        elif command == '/end':
                            self.logger.info("Chat session ended by user (/end command)")
                            self.console.print("[bold cyan]Chat session ended. Goodbye![/bold cyan]")
                            # Clear screen and show main menu
                            os.system('cls' if os.name == 'nt' else 'clear')
                            display_main_menu()
                            break
                        else:
                            self.console.print(f"[yellow]Unknown command: {command}[/yellow]")
                            continue
                    
                    # Process the message
                    self.send_message(user_input)
                
                except KeyboardInterrupt:
                    self.logger.warning("Chat interrupted by user")
                    self.console.print("\n[bold cyan]Chat interrupted. Exiting...[/bold cyan]")
                    # Clear screen and show main menu
                    os.system('cls' if os.name == 'nt' else 'clear')
                    display_main_menu()
                    break
                except Exception as e:
                    self.logger.error(f"Error in chat loop: {e}", exc_info=True)
                    self.console.print(f"[bold red]Error: {str(e)}[/bold red]")
                    continue
        
        except Exception as e:
            self.logger.error(f"Fatal error in chat loop: {e}", exc_info=True)
            self.console.print(f"[bold red]Fatal error: {e}[/bold red]")
            # Clear screen and show main menu
            os.system('cls' if os.name == 'nt' else 'clear')
            display_main_menu() 