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
                    'created_at': model.get('created_at', None),
                    'provider': 'openrouter'
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
    def __init__(self, model_config, logger, console, system_instruction=None, settings_manager=None, chroma_manager=None):
        """Initialize AI Chat with specific model configuration"""
        self.logger = logger
        self.console = console
        self.settings_manager = settings_manager
        self.chroma_manager = chroma_manager
        dotenv.load_dotenv()

        self.model_id = model_config['id']
        self.model_name = model_config['name']
        self.provider = model_config.get('provider', 'openai')
        self.max_tokens = model_config.get('max_tokens')
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
        """Send a message to the AI and get a response"""
        try:
            # Process file context if available
            if self.chroma_manager and self.chroma_manager.vectorstore:
                relevant_context = self.chroma_manager.search_context(user_input)
                if relevant_context:
                    context_message = {
                        "role": "system",
                        "content": "Here is some relevant context from the files:\n\n" + "\n\n".join(relevant_context)
                    }
                    messages = [self.messages[0], context_message] + self.messages[1:]
                else:
                    messages = self.messages.copy()
            else:
                messages = self.messages.copy()

            # Add user message
            current_message = {"role": "user", "content": user_input}
            
            # Process code blocks and file/directory references
            if "```" in user_input or "[[" in user_input:
                content = []
                remaining_text = user_input
                
                # Extract code blocks
                while "```" in remaining_text:
                    start = remaining_text.find("```")
                    if start == -1:
                        break
                    
                    # Add text before code block
                    if start > 0:
                        content.append({
                            "type": "text",
                            "text": remaining_text[:start].strip()
                        })
                    
                    # Find the end of the code block
                    end = remaining_text.find("```", start + 3)
                    if end == -1:
                        # No closing backticks, treat rest as text
                        content.append({
                            "type": "text",
                            "text": remaining_text[start:].strip()
                        })
                        break
                    
                    # Extract language (if specified) and code
                    code_block = remaining_text[start+3:end]
                    language = ""
                    if "\n" in code_block:
                        first_line = code_block[:code_block.find("\n")].strip()
                        if first_line and not first_line.startswith(" "):
                            language = first_line
                            code_block = code_block[code_block.find("\n")+1:]
                    
                    content.append({
                        "type": "code",
                        "language": language,
                        "code": code_block.strip()
                    })
                    
                    remaining_text = remaining_text[end+3:]
                
                # Process remaining text for file/directory references
                while "[[" in remaining_text:
                    start = remaining_text.find("[[")
                    if start == -1:
                        break
                    
                    # Add text before reference
                    if start > 0:
                        content.append({
                            "type": "text",
                            "text": remaining_text[:start].strip()
                        })
                    
                    # Find the end of the reference
                    end = remaining_text.find("]]", start)
                    if end == -1:
                        # No closing brackets, treat rest as text
                        content.append({
                            "type": "text",
                            "text": remaining_text[start:].strip()
                        })
                        break
                    
                    # Extract reference type and path
                    ref = remaining_text[start+2:end].strip()
                    if ":" in ref:
                        ref_type, ref_path = ref.split(":", 1)
                        ref_path = ref_path.strip('"').strip("'").strip()
                        
                        # Process image references
                        if ref_type == "img":
                            success, result = encode_image_to_base64(ref_path)
                            if success:
                                mime_type = get_image_mime_type(ref_path)
                                content.append({
                                    "type": "image",
                                    "image_url": f"data:{mime_type};base64,{result}"
                                })
                            else:
                                content.append({
                                    "type": "text",
                                    "text": f"Error processing image: {result}"
                                })
                        # Add file content to ChromaDB if available
                        elif ref_type == "file" and self.chroma_manager and self.chroma_manager.vectorstore:
                            try:
                                with open(ref_path, 'r', encoding='utf-8') as f:
                                    file_content = f.read()
                                    self.chroma_manager.add_file_to_store(ref_path, file_content)
                            except Exception as e:
                                self.logger.error(f"Error adding file to store: {e}")
                        
                        content.append({
                            "type": "reference",
                            "ref_type": ref_type,
                            "path": ref_path
                        })
                    else:
                        content.append({
                            "type": "text",
                            "text": remaining_text[start:end+2].strip()
                        })
                    
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
                        "temperature": 0.7
                    }
                    
                    # Add max_tokens only if configured
                    if self.max_tokens:
                        request_data["max_tokens"] = self.max_tokens

                    response = self.client.messages.create(**request_data)
                    log_api_response("Anthropic", request_data, response)
                    ai_response = response.content[0].text

                elif self.provider == 'openrouter':
                    # Format messages for OpenRouter API
                    formatted_messages = []
                    for msg in messages:
                        if isinstance(msg["content"], list):
                            # Handle messages with mixed content (text, images, etc.)
                            content_parts = []
                            for part in msg["content"]:
                                if part["type"] == "text":
                                    content_parts.append({"type": "text", "text": part["text"]})
                                elif part["type"] == "image":
                                    content_parts.append({"type": "image_url", "image_url": part["image_url"]})
                            formatted_messages.append({"role": msg["role"], "content": content_parts})
                        else:
                            # Handle simple text messages
                            formatted_messages.append({"role": msg["role"], "content": msg["content"]})

                    request_data = {
                        "model": self.model_id,
                        "messages": formatted_messages,
                        "temperature": 0.7,
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
                    
                    # Handle response based on whether it's a chat completion or image analysis
                    if "choices" in data:
                        ai_response = data['choices'][0]['message']['content'].strip()
                    else:
                        # Handle case where response doesn't have choices (e.g., image analysis)
                        ai_response = data.get('content', "No response content available").strip()
                    
                    # Get the generation ID from the response
                    generation_id = data.get('id')
                    if generation_id:
                        # Add a small delay to ensure cost data is available
                        time.sleep(0.5)
                        
                        # Try multiple times to get the cost data
                        max_retries = 3
                        retry_delay = 0.5
                        
                        for attempt in range(max_retries):
                            try:
                                generation_response = requests.get(
                                    f"{self.api_base}/generation?id={generation_id}",
                                    headers=get_openrouter_headers(self.api_key)
                                )
                                generation_response.raise_for_status()
                                generation_data = generation_response.json()
                                
                                if 'data' in generation_data and generation_data['data'].get('total_cost') is not None:
                                    # Store the accurate cost information
                                    self.last_total_cost = generation_data['data']['total_cost']
                                    self.last_tokens_prompt = generation_data['data'].get('tokens_prompt', 0)
                                    self.last_tokens_completion = generation_data['data'].get('tokens_completion', 0)
                                    self.last_native_tokens_prompt = generation_data['data'].get('native_tokens_prompt', 0)
                                    self.last_native_tokens_completion = generation_data['data'].get('native_tokens_completion', 0)
                                    
                                    # Store per-message stats
                                    msg_index = len(self.messages) - 1
                                    setattr(self, f'last_total_cost_{msg_index}', self.last_total_cost)
                                    setattr(self, f'last_tokens_prompt_{msg_index}', self.last_tokens_prompt)
                                    setattr(self, f'last_tokens_completion_{msg_index}', self.last_tokens_completion)
                                    setattr(self, f'last_native_tokens_prompt_{msg_index}', self.last_native_tokens_prompt)
                                    setattr(self, f'last_native_tokens_completion_{msg_index}', self.last_native_tokens_completion)
                                    break
                                else:
                                    if attempt < max_retries - 1:
                                        time.sleep(retry_delay)
                                        continue
                                    else:
                                        # If we still don't have cost data, try to calculate from usage
                                        if 'usage' in data:
                                            usage = data['usage']
                                            self.last_tokens_prompt = usage.get('prompt_tokens', 0)
                                            self.last_tokens_completion = usage.get('completion_tokens', 0)
                                            self.last_total_cost = 0  # We don't have accurate cost data
                            except Exception as e:
                                self.logger.error(f"Error fetching generation stats (attempt {attempt + 1}): {e}")
                                if attempt < max_retries - 1:
                                    time.sleep(retry_delay)
                                    continue
                                else:
                                    self.last_total_cost = 0
                                    if 'usage' in data:
                                        usage = data['usage']
                                        self.last_tokens_prompt = usage.get('prompt_tokens', 0)
                                        self.last_tokens_completion = usage.get('completion_tokens', 0)

                else:  # OpenAI
                    request_data = {
                        "model": self.model_id,
                        "messages": messages,
                        "temperature": 0.7,
                        "stream": False
                    }
                    
                    # Add max_tokens only if configured
                    if self.max_tokens:
                        request_data["max_tokens"] = self.max_tokens

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
                    'instruction_name': appearance.get('instruction_name_color', '#FFD700'),
                    'cost': appearance.get('cost_color', '#00FFFF')
                }
            return {
                'ai_name': '#A6E22E',  # Default lime green
                'instruction_name': '#FFD700',  # Default gold
                'cost': '#00FFFF'  # Default cyan
            }
        except Exception as e:
            self.logger.error(f"Error getting colors: {e}")
            return {
                'ai_name': '#A6E22E',  # Default lime green
                'instruction_name': '#FFD700',  # Default gold
                'cost': '#00FFFF'  # Default cyan
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
        
        # Create the footer with cost if it's an OpenRouter response
        footer = None
        if self.provider == 'openrouter':
            cost_parts = []
            if hasattr(self, 'last_total_cost'):
                if self.last_total_cost > 0:
                    cost_parts.append(f"Cost: ${self.last_total_cost:.6f}")
                elif self.last_total_cost == 0:
                    cost_parts.append("Cost: Free")
                else:
                    cost_parts.append("Cost: Not Found")
            else:
                cost_parts.append("Cost: Not Found")
            
            if hasattr(self, 'last_tokens_prompt') and hasattr(self, 'last_tokens_completion'):
                tokens = f"Tokens: {self.last_tokens_prompt}+{self.last_tokens_completion}"
                if hasattr(self, 'last_native_tokens_prompt') and hasattr(self, 'last_native_tokens_completion'):
                    if (self.last_native_tokens_prompt != self.last_tokens_prompt or 
                        self.last_native_tokens_completion != self.last_tokens_completion) and \
                        self.last_native_tokens_prompt > 0 and self.last_native_tokens_completion > 0:
                        tokens += f" (Native: {self.last_native_tokens_prompt}+{self.last_native_tokens_completion})"
                cost_parts.append(tokens)
            
            footer = f"[bold {colors['cost']}]{' | '.join(cost_parts)}[/]"
        
        self.console.print(
            Panel(
                Markdown(final_text),
                title=f"[bold {colors['ai_name']}]{self.model_name}[/] [bold {colors['instruction_name']}][{self.instruction_name}][/]",
                subtitle=footer,
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
            
            # Calculate total tokens and cost for OpenRouter
            total_prompt_tokens = 0
            total_completion_tokens = 0
            total_cost = 0
            if self.provider == 'openrouter':
                for i in range(1, len(self.messages), 2):
                    if hasattr(self, f'last_tokens_prompt_{i}'):
                        total_prompt_tokens += getattr(self, f'last_tokens_prompt_{i}', 0)
                        total_completion_tokens += getattr(self, f'last_tokens_completion_{i}', 0)
                        total_cost += getattr(self, f'last_total_cost_{i}', 0)

            # Calculate session duration
            duration = datetime.now() - self.start_time
            hours = duration.seconds // 3600
            minutes = (duration.seconds % 3600) // 60
            seconds = duration.seconds % 60
            duration_str = f"{hours}h {minutes}m {seconds}s"

            chat_data = {
                "model_name": self.model_name,
                "model_id": self.model_id,
                "provider": self.provider,
                "timestamp": self.start_time.isoformat(),
                "duration": duration_str,
                "system_instruction": {
                    "name": self.instruction_name,
                    "content": self.instruction_content
                },
                "messages": self.messages,
                "message_count": len(self.messages) - 1  # Subtract 1 for system message
            }

            # Add OpenRouter specific information
            if self.provider == 'openrouter':
                chat_data["usage"] = {
                    "total_prompt_tokens": total_prompt_tokens,
                    "total_completion_tokens": total_completion_tokens,
                    "total_tokens": total_prompt_tokens + total_completion_tokens,
                    "total_cost": total_cost
                }

            # Add agent information if available
            if self.chroma_manager and self.chroma_manager.vectorstore:
                chat_data["agent"] = {
                    "store": self.chroma_manager.store_name,
                    "embedding_model": self.chroma_manager.embedding_model_name
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
                f.write(f"Duration: {duration_str}\n")
                f.write(f"System Instruction: {self.instruction_name}\n")
                
                if self.provider == 'openrouter':
                    f.write(f"Total Tokens: {total_prompt_tokens + total_completion_tokens} ")
                    f.write(f"(Prompt: {total_prompt_tokens}, Completion: {total_completion_tokens})\n")
                    f.write(f"Total Cost: ${total_cost:.6f}\n")

                if self.chroma_manager and self.chroma_manager.vectorstore:
                    f.write(f"Agent Store: {self.chroma_manager.store_name}\n")
                    f.write(f"Embedding Model: {self.chroma_manager.embedding_model_name}\n")

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
                "   - /help - Display detailed help guide\n"
                "   - /info - Display chat session information\n"
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
        
        def exit_chat(message="Goodbye!"):
            """Exit chat with animation"""
            # Show goodbye message with fade effect
            self.console.print()  # Add a blank line for spacing
            with self.console.status("[bold cyan]👋[/bold cyan]", spinner="dots") as status:
                time.sleep(0.5)
                self.console.print(f"[bold cyan]{message}[/bold cyan]")
                time.sleep(1)
            
            # Clear screen and show main menu
            os.system('cls' if os.name == 'nt' else 'clear')
            display_main_menu()
        
        # Clear screen and show welcome message at start
        os.system('cls' if os.name == 'nt' else 'clear')
        display_welcome()
        
        try:
            while True:
                try:
                    user_input = self.console.input("[bold yellow]You: [/bold yellow]")
                    
                    if user_input.lower() in ['exit', 'quit', 'bye']:
                        self.logger.info("Chat session ended by user")
                        exit_chat("Thanks for chatting! See you next time!")
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
                            exit_chat("Chat session ended. Thanks for using ACT!")
                            break
                        elif command == '/help':
                            self._display_help()
                            continue
                        elif command == '/info':
                            self._display_info()
                            continue
                        else:
                            self.console.print(f"[yellow]Unknown command: {command}[/yellow]")
                            continue
                    
                    # Process the message
                    self.send_message(user_input)
                
                except KeyboardInterrupt:
                    self.logger.warning("Chat interrupted by user")
                    self.console.print("\n")  # Add newline for cleaner output
                    exit_chat("Chat interrupted. Thanks for using ACT!")
                    break
                except Exception as e:
                    self.logger.error(f"Error in chat loop: {e}", exc_info=True)
                    self.console.print(f"[bold red]Error: {str(e)}[/bold red]")
                    continue
        
        except Exception as e:
            self.logger.error(f"Fatal error in chat loop: {e}", exc_info=True)
            self.console.print(f"[bold red]Fatal error: {e}[/bold red]")
            exit_chat("Exiting due to error. Sorry for the inconvenience!") 

    def _display_help(self):
        """Display detailed help information"""
        help_text = (
            "[bold cyan]Available Commands:[/bold cyan]\n"
            "  [bold yellow]/help[/bold yellow]    - Display this help message\n"
            "  [bold yellow]/info[/bold yellow]    - Display chat session information\n"
            "  [bold yellow]/save[/bold yellow]    - Save chat history\n"
            "             Usage: /save [optional_name]\n"
            "  [bold yellow]/clear[/bold yellow]   - Clear screen and chat history\n"
            "  [bold yellow]/insert[/bold yellow]  - Enter multiline text\n"
            "             Type END to finish, CANCEL to abort\n"
            "  [bold yellow]/end[/bold yellow]     - End chat session\n\n"
            "[bold cyan]File & Directory References:[/bold cyan]\n"
            "  [bold yellow][[ file:path/to/file]][/bold yellow]\n"
            "    - View contents of a file\n"
            "    - Example: [[ file:example.py]] or [[ file:\"path with spaces.txt\"]]\n\n"
            "  [bold yellow][[ dir:path/to/directory]][/bold yellow]\n"
            "    - List contents of a directory\n"
            "    - Example: [[ dir:src]] or [[ dir:\"path with spaces\"]]\n\n"
            "  [bold yellow][[ codebase:path]][/bold yellow]\n"
            "    - View all code files in directory\n"
            "    - Example: [[ codebase:src/*.py]]\n\n"
            "[bold cyan]Image References:[/bold cyan]\n"
            "  [bold yellow][[ img:path/to/image]][/bold yellow]\n"
            "    - Include local image or URL\n"
            "    - Example: [[ img:image.jpg]] or [[ img:\"path with spaces.png\"]]\n"
            "    - Example: [[ img:https://example.com/image.jpg]]\n\n"
            "[bold cyan]Quick Exit:[/bold cyan]\n"
            "  Type 'exit', 'quit', or 'bye' to end the session\n"
            "  Press Ctrl+C to interrupt at any time"
        )
        
        self.console.print(
            Panel(
                help_text,
                title="[bold white]📚 ACT Help Guide[/bold white]",
                border_style="cyan",
                padding=(1, 2)
            )
        ) 

    def _display_info(self):
        """Display chat session information"""
        # Calculate session duration
        duration = datetime.now() - self.start_time
        hours = duration.seconds // 3600
        minutes = (duration.seconds % 3600) // 60
        seconds = duration.seconds % 60
        duration_str = f"{hours}h {minutes}m {seconds}s"

        # Get colors from settings
        colors = self._get_colors()

        # Build info sections
        model_info = [
            "[bold cyan]Model Information:[/bold cyan]",
            f"  Provider: [bold {colors['ai_name']}]{self.provider.upper()}[/]",
            f"  Model: [bold {colors['ai_name']}]{self.model_name}[/]",
            f"  Model ID: {self.model_id}",
            f"  Max Tokens: {self.max_tokens or 'Not Set'}"
        ]

        system_info = [
            "\n[bold cyan]System Information:[/bold cyan]",
            f"  Instruction: [bold {colors['instruction_name']}]{self.instruction_name}[/]",
            f"  Session Duration: {duration_str}",
            f"  Messages in Chat: {len(self.messages) - 1}"  # Subtract 1 for system message
        ]

        # Add OpenRouter specific information if applicable
        usage_info = []
        if self.provider == 'openrouter':
            usage_info.extend([
                "\n[bold cyan]Usage Information:[/bold cyan]"
            ])
            
            # Calculate total tokens and cost
            total_prompt_tokens = 0
            total_completion_tokens = 0
            total_cost = 0
            
            # Iterate through message history to sum up costs
            for i in range(1, len(self.messages), 2):  # Skip system message and go through user/assistant pairs
                if hasattr(self, f'last_tokens_prompt_{i}'):
                    total_prompt_tokens += getattr(self, f'last_tokens_prompt_{i}', 0)
                    total_completion_tokens += getattr(self, f'last_tokens_completion_{i}', 0)
                    total_cost += getattr(self, f'last_total_cost_{i}', 0)

            usage_info.extend([
                f"  Total Prompt Tokens: {total_prompt_tokens}",
                f"  Total Completion Tokens: {total_completion_tokens}",
                f"  Total Tokens: {total_prompt_tokens + total_completion_tokens}",
                f"  Total Cost: ${total_cost:.6f}" if total_cost > 0 else "  Total Cost: Free"
            ])

        # Add agent information if available
        agent_info = []
        if self.chroma_manager and self.chroma_manager.vectorstore:
            agent_info.extend([
                "\n[bold cyan]Agent Information:[/bold cyan]",
                f"  Store: {self.chroma_manager.store_name}",
                f"  Embedding Model: {self.chroma_manager.embedding_model_name}"
            ])

        # Combine all sections
        info_text = "\n".join(model_info + system_info + usage_info + agent_info)

        # Display in a panel
        self.console.print(
            Panel(
                info_text,
                title="[bold white]📊 Chat Session Information[/bold white]",
                border_style="cyan",
                padding=(1, 2)
            )
        ) 