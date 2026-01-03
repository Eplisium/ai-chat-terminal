from imports import *
from utils import log_api_response, encode_image_to_base64, get_image_mime_type, read_document_content, sanitize_path, normalize_path
from managers.tools_manager import ToolsManager
from chat.streaming_display import StreamingDisplay
import re
import glob


# Pricing dictionaries for cost calculation (rates per 1K tokens)
# OpenAI pricing - order matters for matching (more specific patterns first)
OPENAI_PRICING = [
    # o1 models
    {"pattern": "o1-mini", "input": 0.003, "output": 0.012},
    {"pattern": "o1", "input": 0.015, "output": 0.06, "exclude": "mini"},
    # GPT-4o models
    {"pattern": "gpt-4o-mini", "input": 0.00015, "output": 0.0006, "audio_input": 0.01, "audio_output": 0.02},
    {"pattern": "gpt-4o", "input": 0.0025, "output": 0.01, "audio_input": 0.10, "audio_output": 0.20, "exclude": "mini"},
    {"pattern": "chatgpt-4o-latest", "input": 0.005, "output": 0.015},
    # GPT-4 models
    {"pattern": "gpt-4-32k", "input": 0.06, "output": 0.12},
    {"pattern": "gpt-4-turbo", "input": 0.01, "output": 0.03},
    {"pattern": "gpt-4-0125-preview", "input": 0.01, "output": 0.03},
    {"pattern": "gpt-4-1106-preview", "input": 0.01, "output": 0.03},
    {"pattern": "gpt-4-vision-preview", "input": 0.01, "output": 0.03},
    {"pattern": "gpt-4", "input": 0.03, "output": 0.06},
    # GPT-3.5 models
    {"pattern": "gpt-3.5-turbo-16k", "input": 0.003, "output": 0.004},
    {"pattern": "gpt-3.5-turbo-0125", "input": 0.0005, "output": 0.0015},
    {"pattern": "gpt-3.5-turbo-1106", "input": 0.001, "output": 0.002},
    {"pattern": "gpt-3.5-turbo-instruct", "input": 0.0015, "output": 0.002},
    {"pattern": "gpt-3.5-turbo", "input": 0.0015, "output": 0.002},
    # Base models
    {"pattern": "davinci-002", "input": 0.002, "output": 0.002},
    {"pattern": "babbage-002", "input": 0.0004, "output": 0.0004},
]

# Anthropic pricing - order matters for matching (more specific patterns first)
ANTHROPIC_PRICING = [
    {"pattern": "claude-3-5-sonnet", "input": 0.003, "output": 0.015},
    {"pattern": "claude-3-5-haiku", "input": 0.0008, "output": 0.004},
    {"pattern": "claude-3-opus", "input": 0.015, "output": 0.075},
    {"pattern": "claude-3-sonnet", "input": 0.003, "output": 0.015},
    {"pattern": "claude-3-haiku", "input": 0.00025, "output": 0.00125},
]

# Default pricing for unknown models
DEFAULT_OPENAI_PRICING = {"input": 0.0005, "output": 0.0015}
DEFAULT_ANTHROPIC_PRICING = {"input": 0.003, "output": 0.015}


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
    
    def get_recent_models(self, models, limit=10):
        """Get the most recently added models (first models from API response)"""
        return models[:limit]
    
    def group_models_by_company(self, models, show_recent=False):
        """Group models by their company with option to show recent models first"""
        company_models = defaultdict(list)
        
        if show_recent:
            recent_models = self.get_recent_models(models)
            if recent_models:
                company_models['Recent'] = recent_models
        
        for model in models:
            model_id = model['id']
            if '/' in model_id:
                company = model_id.split('/')[0].title()
            else:
                company = 'Other'
            
            company_models[company].append(model)
        
        for company in company_models:
            if company != 'Recent':  # Don't sort Recent models as they're already in order
                company_models[company].sort(key=lambda x: x['name'].lower())
        
        return dict(company_models)

class AIChat:
    def __init__(self, model_config, logger, console, system_instruction=None, settings_manager=None, chroma_manager=None, stats_manager=None):
        """Initialize AI Chat with specific model configuration"""
        self.logger = logger
        self.console = console
        self.settings_manager = settings_manager
        self.chroma_manager = chroma_manager
        self.stats_manager = stats_manager
        self.session_id = None  # Track current session ID
        dotenv.load_dotenv()

        self.model_id = model_config['id']
        self.model_name = model_config['name']
        self.provider = model_config.get('provider', 'openai')
        self.max_tokens = model_config.get('max_tokens')
        self.start_time = datetime.now()
        self.last_save_path = None  # Track last save location
        self.last_save_name = None  # Track last used custom name
        
        # Initialize tools manager
        self.tools_manager = ToolsManager(logger=logger, settings_manager=settings_manager)
        
        # Get streaming and tools settings
        self.streaming_enabled = False
        self.tools_enabled = False
        if self.settings_manager:
            settings = self.settings_manager._load_settings()
            self.streaming_enabled = settings.get('streaming', {}).get('enabled', False)
            self.tools_enabled = settings.get('tools', {}).get('enabled', False)
        
        # Cache colors at session start to avoid repeated lookups
        self._cached_colors = self._get_colors()
        
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
        
        elif self.provider == 'custom':
            # Handle custom provider from model_config
            if 'custom_provider' not in model_config:
                self.logger.error("Custom provider details not found in model config")
                raise ValueError("Custom provider configuration is missing")
            
            custom_provider = model_config['custom_provider']
            self.api_base = custom_provider.get('base_url', '').rstrip('/')
            self.api_key = custom_provider.get('api_key', '')
            self.header_auth = custom_provider.get('header_auth', True)
            self.use_chat_completions_endpoint = custom_provider.get('use_chat_completions_endpoint', True)
            
            if not self.api_base:
                self.logger.error("Custom provider API base URL not specified")
                raise ValueError("Custom provider API base URL is required")
            
            self.logger.debug(f"Initializing custom provider client for {custom_provider.get('name')} with base URL: {self.api_base}")
            if self.api_key:
                self.logger.debug(f"Using API key: {mask_key(self.api_key)}")
        
        else:  # openai
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI library not installed. Please install 'openai'.")
            
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                self.logger.error("OpenAI API key not found")
                raise ValueError("OpenAI API key not found in .env file")
            
            self.logger.debug(f"Initializing OpenAI client with API key: {mask_key(api_key)}")
            self.client = OpenAI(api_key=api_key)

    def _extract_path_from_reference(self, reference):
        """Extract path from a reference, handling Windows drive letters correctly"""
        try:
            # Split only on the first colon that's not part of a Windows drive letter
            parts = reference.split(':', 1)
            if len(parts) < 2:
                return None, reference
            
            ref_type = parts[0].strip()
            
            # Handle Windows drive letters (e.g., C:)
            remaining = parts[1]
            if len(remaining) >= 2 and remaining[1] == ':':
                # This is a Windows path with drive letter
                drive = remaining[0]
                path = remaining[1:]  # Include the : and everything after
                return ref_type, drive + path
            
            return ref_type, remaining.strip()
            
        except Exception as e:
            self.logger.error(f"Error extracting path from reference: {e}", exc_info=True)
            return None, reference

    def _process_file_reference(self, file_ref):
        """Process a file reference and return its contents"""
        try:
            # Sanitize and normalize the path
            file_path = normalize_path(file_ref)
            
            # Check if file exists
            if not os.path.exists(file_path):
                return False, f"File not found: {file_path}"
            
            if not os.path.isfile(file_path):
                return False, f"Not a directory: {file_path}"
            
            # Read file contents
            try:
                success, content = read_document_content(file_path)
                if success:
                    return True, f"Contents of file {os.path.basename(file_path)}:\n```\n{content}\n```"
                else:
                    return False, f"Error reading file: {content}"
            except Exception as e:
                return False, f"Error reading file: {str(e)}"
                
        except Exception as e:
            return False, f"Error processing file: {str(e)}"

    def _process_directory_reference(self, dir_ref):
        """Process a directory reference and return its contents"""
        try:
            # Sanitize and normalize the path
            dir_path = normalize_path(dir_ref)
            
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

    def _preprocess_message(self, message):
        """Preprocess message to handle file, directory, and image references"""
        try:
            if not isinstance(message, str):
                return message

            # Regular expression to find all [[...]] references
            pattern = r'\[\[(.*?)\]\]'
            matches = re.findall(pattern, message)
            processed_message = message

            for match in matches:
                reference = match.strip()
                if not reference:
                    continue

                # Extract type and path using the new method
                ref_type, ref_path = self._extract_path_from_reference(reference)
                if not ref_type:
                    continue
                
                if ref_type == 'file':
                    success, content = self._process_file_reference(ref_path)
                    if success:
                        processed_message = processed_message.replace(f"[[{reference}]]", content)
                    else:
                        processed_message = processed_message.replace(f"[[{reference}]]", f"[Error: {content}]")
                
                elif ref_type == 'dir':
                    success, content = self._process_directory_reference(ref_path)
                    if success:
                        processed_message = processed_message.replace(f"[[{reference}]]", content)
                    else:
                        processed_message = processed_message.replace(f"[[{reference}]]", f"[Error: {content}]")
                
                elif ref_type == 'img':
                    try:
                        # Sanitize and normalize the path
                        img_path = normalize_path(ref_path)
                        # Handle both local files and URLs
                        success, base64_data = encode_image_to_base64(img_path)
                        if success:
                            mime_type = get_image_mime_type(img_path)
                            data_url = f"data:{mime_type};base64,{base64_data}"
                            # Replace the reference with a list containing text and image parts
                            return [
                                {"type": "text", "text": message.replace(f"[[{reference}]]", "")},
                                {"type": "image_url", "image_url": {"url": data_url}}
                            ]
                        else:
                            processed_message = processed_message.replace(f"[[{reference}]]", f"[Error loading image: {base64_data}]")
                    except Exception as e:
                        processed_message = processed_message.replace(f"[[{reference}]]", f"[Error processing image: {str(e)}]")
                
                elif ref_type == 'codebase':
                    try:
                        # Sanitize and normalize the path
                        code_path = normalize_path(ref_path)
                        # Use glob to find matching files
                        matching_files = glob.glob(code_path, recursive=True)
                        if not matching_files:
                            processed_message = processed_message.replace(f"[[{reference}]]", f"[No files found matching: {code_path}]")
                            continue

                        # Process each file
                        contents = []
                        for file_path in matching_files[:5]:  # Limit to first 5 files to avoid token limits
                            if os.path.isfile(file_path):
                                success, content = read_document_content(file_path)
                                if success:
                                    contents.append(f"File: {file_path}\n```\n{content}\n```")
                        
                        if contents:
                            replacement = "Contents of matching files:\n\n" + "\n\n".join(contents)
                            if len(matching_files) > 5:
                                replacement += f"\n\n[Note: Showing first 5 of {len(matching_files)} matching files]"
                            processed_message = processed_message.replace(f"[[{reference}]]", replacement)
                        else:
                            processed_message = processed_message.replace(f"[[{reference}]]", "[No readable files found]")
                    except Exception as e:
                        processed_message = processed_message.replace(f"[[{reference}]]", f"[Error processing codebase: {str(e)}]")

            return processed_message

        except Exception as e:
            self.logger.error(f"Error preprocessing message: {e}", exc_info=True)
            return message

    def _prepare_messages(self, user_input, processed_input):
        """Prepare messages with RAG context if available"""
        # Process file context if available (with visual feedback)
        if self.chroma_manager and self.chroma_manager.vectorstore:
            rag_status = self.console.status("[bold magenta]📚 Searching knowledge base...[/bold magenta]")
            rag_status.start()
            
            relevant_context = self.chroma_manager.search_context(user_input)
            self._rag_results_count = len(relevant_context) if relevant_context else 0
            
            rag_status.stop()
            if relevant_context:
                self.console.print(f"[green]📚 Found {len(relevant_context)} relevant document(s)[/green]")
            else:
                self.console.print("[dim]📚 No relevant context found[/dim]")
            
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

        # Add user message with processed content
        current_message = {"role": "user", "content": processed_input}
        messages.append(current_message)
        self.messages.append(current_message)
        
        return messages

    def _store_usage_stats(self, prompt_tokens, completion_tokens, cost=None, native_prompt=None, native_completion=None):
        """Store token usage statistics for the current message"""
        self.last_tokens_prompt = prompt_tokens
        self.last_tokens_completion = completion_tokens
        if cost is not None:
            self.last_total_cost = cost
        if native_prompt is not None:
            self.last_native_tokens_prompt = native_prompt
        if native_completion is not None:
            self.last_native_tokens_completion = native_completion
        
        msg_index = len(self.messages) - 1
        setattr(self, f'last_tokens_prompt_{msg_index}', prompt_tokens)
        setattr(self, f'last_tokens_completion_{msg_index}', completion_tokens)
        if cost is not None:
            setattr(self, f'last_total_cost_{msg_index}', cost)
        if native_prompt is not None:
            setattr(self, f'last_native_tokens_prompt_{msg_index}', native_prompt)
        if native_completion is not None:
            setattr(self, f'last_native_tokens_completion_{msg_index}', native_completion)

    def _process_tool_calls_streaming(self, tool_calls_list, messages, streaming_display):
        """Process tool calls and execute them, returning results and updated messages"""
        tool_messages = []
        tool_results = []
        
        for tool_call in tool_calls_list:
            streaming_display.start_tool_call(
                tool_call['function']['name'],
                tool_call['function'].get('arguments', '')
            )
            result = self.tools_manager.execute_tool(tool_call)
            tool_results.append(result)
            streaming_display.complete_tool_call(result[:100] if result else None)
            tool_messages.append({
                "role": "tool",
                "name": tool_call['function']['name'],
                "tool_call_id": tool_call.get('id', ''),
                "content": result
            })
        
        return tool_messages, tool_results

    def _process_tool_calls_non_streaming(self, tool_calls):
        """Process tool calls in non-streaming mode"""
        tool_messages = []
        tool_results = []
        
        for tool_call in tool_calls:
            # Normalize tool_call format
            if hasattr(tool_call, 'id'):
                # OpenAI SDK response object
                tc_dict = {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                }
            else:
                # Already a dict (OpenRouter)
                tc_dict = {
                    "id": tool_call.get('id', ''),
                    "type": "function",
                    "function": {
                        "name": tool_call['function']['name'],
                        "arguments": tool_call['function']['arguments']
                    }
                }
            
            result = self.tools_manager.execute_tool(tc_dict)
            tool_results.append(result)
            
            tool_messages.append({
                "role": "tool",
                "name": tc_dict['function']['name'],
                "tool_call_id": tc_dict.get('id', ''),
                "content": result
            })
        
        return tool_messages, tool_results

    def _handle_anthropic_request(self, messages, streaming_display, interrupted_flag):
        """Handle Anthropic API request"""
        partial_response = ""
        reasoning_content = ""
        
        # Format messages for Anthropic
        anthropic_messages = []
        for msg in messages[1:]:
            if isinstance(msg["content"], list):
                content_parts = []
                for part in msg["content"]:
                    if part["type"] == "text":
                        content_parts.append({"type": "text", "text": part["text"]})
                    elif part["type"] == "image_url":
                        content_parts.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": part["image_url"]["url"].split(";")[0].split(":")[1],
                                "data": part["image_url"]["url"].split(",")[1]
                            }
                        })
                anthropic_messages.append({
                    "role": "user" if msg["role"] == "user" else "assistant",
                    "content": content_parts
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
            "temperature": 0.7,
            "max_tokens": self.max_tokens if self.max_tokens else 4096,
            "stream": self.streaming_enabled
        }
        
        if self.streaming_enabled:
            response = self.client.messages.create(**request_data)
            stream_usage = None
            
            for chunk in response:
                if interrupted_flag[0]:
                    break
                if hasattr(chunk, 'type'):
                    if chunk.type == "content_block_start":
                        if hasattr(chunk, 'content_block') and hasattr(chunk.content_block, 'type'):
                            if chunk.content_block.type == "thinking":
                                streaming_display.set_phase("reasoning")
                            elif chunk.content_block.type == "text":
                                streaming_display.set_phase("responding")
                    elif chunk.type == "content_block_delta":
                        if hasattr(chunk, 'delta'):
                            if hasattr(chunk.delta, 'type') and chunk.delta.type == "thinking_delta":
                                reasoning_content += chunk.delta.thinking or ""
                                streaming_display.add_reasoning(chunk.delta.thinking or "")
                            elif hasattr(chunk.delta, 'text'):
                                partial_response += chunk.delta.text or ""
                                streaming_display.add_content(chunk.delta.text or "")
                    elif chunk.type == "message_start":
                        if hasattr(chunk, 'message') and hasattr(chunk.message, 'usage'):
                            stream_usage = {'input_tokens': chunk.message.usage.input_tokens, 'output_tokens': 0}
                    elif chunk.type == "message_delta":
                        if hasattr(chunk, 'usage') and hasattr(chunk.usage, 'output_tokens'):
                            if stream_usage:
                                stream_usage['output_tokens'] = chunk.usage.output_tokens
            
            if not interrupted_flag[0] and stream_usage:
                cost = self._calculate_anthropic_cost(stream_usage['input_tokens'], stream_usage['output_tokens'], self.model_id)
                self._store_usage_stats(stream_usage['input_tokens'], stream_usage['output_tokens'], cost)
        else:
            response = self.client.messages.create(**request_data)
            log_api_response("Anthropic", request_data, response)
            partial_response = response.content[0].text
            
            if hasattr(response, 'usage'):
                cost = self._calculate_anthropic_cost(response.usage.input_tokens, response.usage.output_tokens, self.model_id)
                self._store_usage_stats(response.usage.input_tokens, response.usage.output_tokens, cost)

        return partial_response

    def _handle_openrouter_request(self, messages, streaming_display, interrupted_flag):
        """Handle OpenRouter API request"""
        partial_response = ""
        reasoning_content = ""
        
        # Format messages
        formatted_messages = []
        for msg in messages:
            if isinstance(msg["content"], list):
                content_parts = []
                for part in msg["content"]:
                    if part["type"] == "text":
                        content_parts.append({"type": "text", "text": part["text"]})
                    elif part["type"] == "image_url":
                        content_parts.append({"type": "image_url", "image_url": part["image_url"]["url"]})
                formatted_messages.append({"role": msg["role"], "content": content_parts})
            else:
                formatted_messages.append({"role": msg["role"], "content": msg["content"]})

        request_data = {
            "model": self.model_id,
            "messages": formatted_messages,
            "temperature": 0.7,
            "stream": self.streaming_enabled
        }

        if self.tools_enabled:
            tools = self.tools_manager.get_enabled_tools()
            if tools:
                request_data["tools"] = tools
                request_data["tool_choice"] = "auto"

        data = None
        generation_id = None
        
        if self.streaming_enabled:
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=get_openrouter_headers(self.api_key),
                json=request_data,
                stream=True
            )
            response.raise_for_status()
            
            accumulated_tool_calls = {}
            finish_reason = None
            
            for line in response.iter_lines():
                if interrupted_flag[0]:
                    break
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        line_data = line[6:]
                        if line_data.strip() == '[DONE]':
                            continue
                        try:
                            chunk = json.loads(line_data)
                            if chunk.get('id') and not generation_id:
                                generation_id = chunk['id']
                            
                            if chunk.get('choices'):
                                choice = chunk['choices'][0]
                                if choice.get('finish_reason'):
                                    finish_reason = choice['finish_reason']
                                
                                delta = choice.get('delta', {})
                                
                                if delta.get('reasoning') or delta.get('reasoning_content'):
                                    reasoning_text = delta.get('reasoning') or delta.get('reasoning_content') or ''
                                    reasoning_content += reasoning_text
                                    streaming_display.add_reasoning(reasoning_text)
                                
                                if delta.get('content'):
                                    partial_response += delta['content']
                                    streaming_display.add_content(delta['content'])
                                
                                if delta.get('tool_calls'):
                                    for tc in delta['tool_calls']:
                                        idx = tc.get('index', 0)
                                        if idx not in accumulated_tool_calls:
                                            accumulated_tool_calls[idx] = {'id': '', 'type': 'function', 'function': {'name': '', 'arguments': ''}}
                                        if tc.get('id'):
                                            accumulated_tool_calls[idx]['id'] = tc['id']
                                        if tc.get('type'):
                                            accumulated_tool_calls[idx]['type'] = tc['type']
                                        if tc.get('function'):
                                            if tc['function'].get('name'):
                                                accumulated_tool_calls[idx]['function']['name'] = tc['function']['name']
                                                streaming_display.start_tool_call(tc['function']['name'])
                                            if tc['function'].get('arguments'):
                                                accumulated_tool_calls[idx]['function']['arguments'] += tc['function']['arguments']
                        except json.JSONDecodeError:
                            continue
            
            # Handle tool calls
            if accumulated_tool_calls and finish_reason == 'tool_calls' and not interrupted_flag[0]:
                tool_calls_list = [accumulated_tool_calls[i] for i in sorted(accumulated_tool_calls.keys())]
                
                assistant_message = {
                    "role": "assistant",
                    "content": partial_response if partial_response else None,
                    "tool_calls": tool_calls_list
                }
                messages.append(assistant_message)
                
                tool_messages, tool_results = self._process_tool_calls_streaming(tool_calls_list, messages, streaming_display)
                messages.extend(tool_messages)
                
                request_data["messages"] = messages
                request_data["stream"] = False
                request_data.pop("tools", None)
                request_data.pop("tool_choice", None)
                
                follow_up = requests.post(
                    f"{self.api_base}/chat/completions",
                    headers=get_openrouter_headers(self.api_key),
                    json=request_data
                )
                follow_up.raise_for_status()
                data = follow_up.json()
                generation_id = data.get('id')
                
                if 'choices' in data and len(data['choices']) > 0:
                    partial_response = data['choices'][0]['message']['content'].strip()
                else:
                    partial_response = "I apologize, but I encountered an issue while processing the tool results. Let me try to summarize what I found: " + " ".join(tool_results)
        else:
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=get_openrouter_headers(self.api_key),
                json=request_data
            )
            response.raise_for_status()
            data = response.json()
            log_api_response("OpenRouter", request_data, data)
            
            if data['choices'][0].get('message', {}).get('tool_calls'):
                initial_content = data['choices'][0]['message'].get('content')
                assistant_message = {
                    "role": "assistant",
                    "content": initial_content,
                    "tool_calls": [
                        {"id": tc.get('id', ''), "type": "function", "function": {"name": tc['function']['name'], "arguments": tc['function']['arguments']}}
                        for tc in data['choices'][0]['message']['tool_calls']
                    ]
                }
                messages.append(assistant_message)
                
                tool_messages, tool_results = self._process_tool_calls_non_streaming(data['choices'][0]['message']['tool_calls'])
                messages.extend(tool_messages)
                
                request_data["messages"] = messages
                request_data.pop("tools", None)
                request_data.pop("tool_choice", None)
                
                response = requests.post(
                    f"{self.api_base}/chat/completions",
                    headers=get_openrouter_headers(self.api_key),
                    json=request_data
                )
                response.raise_for_status()
                data = response.json()
                
                if 'choices' in data and len(data['choices']) > 0:
                    partial_response = data['choices'][0]['message']['content'].strip()
                else:
                    partial_response = "I apologize, but I encountered an issue while processing the tool results. Let me try to summarize what I found: " + " ".join(tool_results)
            else:
                partial_response = data['choices'][0]['message']['content'].strip()

        # Get token usage
        final_generation_id = generation_id or (data.get('id') if data else None)
        if final_generation_id and not interrupted_flag[0]:
            time.sleep(0.5)
            for attempt in range(3):
                try:
                    generation_response = requests.get(
                        f"{self.api_base}/generation?id={final_generation_id}",
                        headers=get_openrouter_headers(self.api_key)
                    )
                    generation_data = generation_response.json()
                    
                    if 'data' in generation_data and generation_data['data'].get('total_cost') is not None:
                        self._store_usage_stats(
                            generation_data['data'].get('tokens_prompt', 0),
                            generation_data['data'].get('tokens_completion', 0),
                            generation_data['data']['total_cost'],
                            generation_data['data'].get('native_tokens_prompt', 0),
                            generation_data['data'].get('native_tokens_completion', 0)
                        )
                        break
                except Exception:
                    if attempt < 2:
                        time.sleep(0.5)
                        continue

        return partial_response

    def _handle_custom_request(self, messages, streaming_display, interrupted_flag):
        """Handle custom provider API request"""
        partial_response = ""
        
        # Format messages
        formatted_messages = []
        for msg in messages:
            if isinstance(msg["content"], list):
                content_parts = []
                for part in msg["content"]:
                    if part["type"] == "text":
                        content_parts.append({"type": "text", "text": part["text"]})
                    elif part["type"] == "image_url":
                        content_parts.append({"type": "image_url", "image_url": part["image_url"]["url"]})
                formatted_messages.append({"role": msg["role"], "content": content_parts})
            else:
                formatted_messages.append({"role": msg["role"], "content": msg["content"]})

        request_data = {
            "model": self.model_id.split('/')[-1],
            "messages": formatted_messages,
            "temperature": 0.7,
            "stream": self.streaming_enabled
        }

        if self.max_tokens:
            request_data["max_tokens"] = self.max_tokens

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            if self.header_auth:
                headers["Authorization"] = f"Bearer {self.api_key}"
            else:
                request_data["api_key"] = self.api_key

        data = None
        stream_usage = None
        
        if self.streaming_enabled:
            response = requests.post(
                f"{self.api_base}{'/chat/completions' if self.use_chat_completions_endpoint else ''}",
                headers=headers,
                json=request_data,
                stream=True
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if interrupted_flag[0]:
                    break
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        line_data = line[6:]
                        if line_data.strip() == '[DONE]':
                            continue
                        try:
                            chunk = json.loads(line_data)
                            if chunk.get('usage'):
                                stream_usage = {
                                    'prompt_tokens': chunk['usage'].get('prompt_tokens', 0),
                                    'completion_tokens': chunk['usage'].get('completion_tokens', 0)
                                }
                            if chunk.get('choices') and chunk['choices'][0].get('delta', {}).get('content'):
                                content = chunk['choices'][0]['delta']['content']
                                partial_response += content
                                streaming_display.add_content(content)
                        except json.JSONDecodeError:
                            continue
        else:
            response = requests.post(
                f"{self.api_base}{'/chat/completions' if self.use_chat_completions_endpoint else ''}",
                headers=headers,
                json=request_data
            )
            response.raise_for_status()
            data = response.json()
            log_api_response("Custom Provider", request_data, data)
            
            if 'choices' in data and len(data['choices']) > 0:
                if 'message' in data['choices'][0]:
                    partial_response = data['choices'][0]['message']['content'].strip()
                elif 'content' in data['choices'][0]:
                    partial_response = data['choices'][0]['content'].strip()
                else:
                    self.logger.warning(f"Unexpected response format: {data}")
                    partial_response = "I apologize, but I received an unexpected response format from the server."
            else:
                self.logger.warning(f"No choices in response: {data}")
                partial_response = "I apologize, but I couldn't generate a proper response."

        # Try to get token usage
        if data and data.get('usage'):
            usage = data['usage']
            self.last_tokens_prompt = usage.get('prompt_tokens', 0)
            self.last_tokens_completion = usage.get('completion_tokens', 0)
            self.last_total_tokens = usage.get('total_tokens', 0)
            msg_index = len(self.messages) - 1
            setattr(self, f'last_tokens_prompt_{msg_index}', self.last_tokens_prompt)
            setattr(self, f'last_tokens_completion_{msg_index}', self.last_tokens_completion)
            setattr(self, f'last_total_tokens_{msg_index}', self.last_total_tokens)
        elif stream_usage and not interrupted_flag[0]:
            self.last_tokens_prompt = stream_usage['prompt_tokens']
            self.last_tokens_completion = stream_usage['completion_tokens']
            self.last_total_tokens = self.last_tokens_prompt + self.last_tokens_completion
            msg_index = len(self.messages) - 1
            setattr(self, f'last_tokens_prompt_{msg_index}', self.last_tokens_prompt)
            setattr(self, f'last_tokens_completion_{msg_index}', self.last_tokens_completion)
            setattr(self, f'last_total_tokens_{msg_index}', self.last_total_tokens)

        return partial_response

    def _handle_openai_request(self, messages, streaming_display, interrupted_flag):
        """Handle OpenAI API request"""
        partial_response = ""
        
        request_data = {
            "model": self.model_id,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": self.max_tokens if self.max_tokens else 4096,
            "stream": self.streaming_enabled
        }
        
        if self.streaming_enabled:
            request_data["stream_options"] = {"include_usage": True}

        if self.tools_enabled and self.tools_manager:
            tools = self.tools_manager.get_enabled_tools()
            if tools:
                request_data["tools"] = tools
                request_data["tool_choice"] = "auto"

        if self.streaming_enabled:
            response = self.client.chat.completions.create(**request_data)
            
            accumulated_tool_calls = {}
            finish_reason = None
            stream_usage = None
            
            for chunk in response:
                if interrupted_flag[0]:
                    break
                
                if hasattr(chunk, 'usage') and chunk.usage:
                    stream_usage = {
                        'prompt_tokens': chunk.usage.prompt_tokens,
                        'completion_tokens': chunk.usage.completion_tokens
                    }
                
                if not chunk.choices:
                    continue
                
                choice = chunk.choices[0]
                if choice.finish_reason:
                    finish_reason = choice.finish_reason
                
                delta = choice.delta
                
                if hasattr(delta, 'content') and delta.content:
                    partial_response += delta.content
                    streaming_display.add_content(delta.content)
                
                if hasattr(delta, 'tool_calls') and delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index if hasattr(tc, 'index') else 0
                        if idx not in accumulated_tool_calls:
                            accumulated_tool_calls[idx] = {'id': '', 'type': 'function', 'function': {'name': '', 'arguments': ''}}
                        if hasattr(tc, 'id') and tc.id:
                            accumulated_tool_calls[idx]['id'] = tc.id
                        if hasattr(tc, 'type') and tc.type:
                            accumulated_tool_calls[idx]['type'] = tc.type
                        if hasattr(tc, 'function') and tc.function:
                            if hasattr(tc.function, 'name') and tc.function.name:
                                accumulated_tool_calls[idx]['function']['name'] = tc.function.name
                                streaming_display.start_tool_call(tc.function.name)
                            if hasattr(tc.function, 'arguments') and tc.function.arguments:
                                accumulated_tool_calls[idx]['function']['arguments'] += tc.function.arguments
            
            # Handle tool calls
            if accumulated_tool_calls and finish_reason == 'tool_calls' and not interrupted_flag[0]:
                tool_calls_list = [accumulated_tool_calls[i] for i in sorted(accumulated_tool_calls.keys())]
                
                assistant_message = {
                    "role": "assistant",
                    "content": partial_response if partial_response else None,
                    "tool_calls": tool_calls_list
                }
                messages.append(assistant_message)
                
                tool_messages, tool_results = self._process_tool_calls_streaming(tool_calls_list, messages, streaming_display)
                messages.extend(tool_messages)
                
                request_data["messages"] = messages
                request_data["stream"] = False
                request_data.pop("tools", None)
                request_data.pop("tool_choice", None)
                request_data.pop("stream_options", None)
                
                follow_up = self.client.chat.completions.create(**request_data)
                log_api_response("OpenAI", request_data, follow_up)
                
                if follow_up.choices and len(follow_up.choices) > 0:
                    partial_response = follow_up.choices[0].message.content.strip()
                else:
                    partial_response = "I apologize, but I encountered an issue while processing the tool results. Let me try to summarize what I found: " + " ".join(tool_results)
                
                if hasattr(follow_up, 'usage'):
                    cost = self._calculate_openai_cost(follow_up.usage.prompt_tokens, follow_up.usage.completion_tokens, self.model_id)
                    self._store_usage_stats(follow_up.usage.prompt_tokens, follow_up.usage.completion_tokens, cost)
            elif not interrupted_flag[0] and stream_usage:
                cost = self._calculate_openai_cost(stream_usage['prompt_tokens'], stream_usage['completion_tokens'], self.model_id)
                self._store_usage_stats(stream_usage['prompt_tokens'], stream_usage['completion_tokens'], cost)
        else:
            response = self.client.chat.completions.create(**request_data)
            log_api_response("OpenAI", request_data, response)
            
            if response.choices[0].message.tool_calls:
                initial_content = response.choices[0].message.content
                assistant_message = {
                    "role": "assistant",
                    "content": initial_content,
                    "tool_calls": [
                        {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                        for tc in response.choices[0].message.tool_calls
                    ]
                }
                messages.append(assistant_message)
                
                tool_messages, tool_results = self._process_tool_calls_non_streaming(response.choices[0].message.tool_calls)
                messages.extend(tool_messages)
                
                request_data["messages"] = messages
                request_data.pop("tools", None)
                request_data.pop("tool_choice", None)
                
                response = self.client.chat.completions.create(**request_data)
                log_api_response("OpenAI", request_data, response)
                partial_response = response.choices[0].message.content.strip()
            else:
                partial_response = response.choices[0].message.content.strip()
            
            if hasattr(response, 'usage'):
                cost = self._calculate_openai_cost(response.usage.prompt_tokens, response.usage.completion_tokens, self.model_id)
                self._store_usage_stats(response.usage.prompt_tokens, response.usage.completion_tokens, cost)

        return partial_response

    def send_message(self, user_input):
        """Send a message to the AI model and get the response"""
        try:
            # Preprocess the message to handle file/dir/img references
            processed_input = self._preprocess_message(user_input)
            
            # Prepare messages with RAG context
            messages = self._prepare_messages(user_input, processed_input)
            
            # Initialize tool results
            self.last_tool_results = []
            
            # Check if the message is a command
            if isinstance(user_input, str):
                user_input_lower = user_input.strip().lower()
                if user_input_lower in ['bye', 'exit', 'quit', 'cya', 'adios', '/end', '/info', '/help', '/clear', '/save', '/insert']:
                    return None
            
            # Record sent message only when we're about to make the API call
            if self.stats_manager:
                self.stats_manager.record_chat(self.model_id, "sent", session_id=self.session_id)
            
            start_time = time.time()
            
            # Variables for response handling
            partial_response = ""
            interrupted_flag = [False]  # Use list to allow modification in nested function
            original_handler = signal.getsignal(signal.SIGINT)
            
            # Initialize streaming display or thinking indicator
            streaming_display = None
            stop_timer = None
            timer_thread = None
            thinking_message = None
            
            if self.streaming_enabled:
                streaming_display = StreamingDisplay(
                    self.console, 
                    self.model_name, 
                    self.instruction_name, 
                    self._cached_colors
                )
                streaming_display.start()
            else:
                thinking_message = self.console.status("")
                thinking_message.start()
                stop_timer = threading.Event()
                
                def update_thinking_message():
                    while not stop_timer.is_set():
                        elapsed = time.time() - start_time
                        minutes = int(elapsed // 60)
                        seconds = elapsed % 60
                        time_display = f"{seconds:.2f} secs"
                        if minutes > 0:
                            time_display = f"{minutes} min {seconds:.2f} secs"
                        thinking_message.update(
                            f"[bold yellow]{self.model_name} is thinking... ({time_display})[/bold yellow]"
                        )
                        time.sleep(0.1)
                
                timer_thread = threading.Thread(target=update_thinking_message)
                timer_thread.daemon = True
                timer_thread.start()

            def signal_handler(signum, frame):
                interrupted_flag[0] = True
                signal.signal(signal.SIGINT, original_handler)
            
            try:
                if self.streaming_enabled:
                    signal.signal(signal.SIGINT, signal_handler)
                
                # Route to appropriate provider handler
                if self.provider == 'anthropic':
                    partial_response = self._handle_anthropic_request(messages, streaming_display, interrupted_flag)
                elif self.provider == 'openrouter':
                    partial_response = self._handle_openrouter_request(messages, streaming_display, interrupted_flag)
                elif self.provider == 'custom':
                    partial_response = self._handle_custom_request(messages, streaming_display, interrupted_flag)
                else:  # Default provider (OpenAI)
                    partial_response = self._handle_openai_request(messages, streaming_display, interrupted_flag)

            finally:
                # Restore original signal handler if streaming was enabled
                if self.streaming_enabled:
                    signal.signal(signal.SIGINT, original_handler)
                    if streaming_display:
                        streaming_display.stop()
                else:
                    if stop_timer:
                        stop_timer.set()
                    if timer_thread:
                        timer_thread.join(timeout=1.0)
                    if thinking_message:
                        thinking_message.stop()
                
                # Calculate total response time
                response_time = time.time() - start_time
                self.last_response_time = response_time
                
                if interrupted_flag[0]:
                    self.console.print("\n[yellow]Message interrupted by user[/yellow]")
                
                # Record stats if available
                if self.stats_manager and hasattr(self, 'last_tokens_prompt'):
                    token_count = self.last_tokens_prompt + self.last_tokens_completion
                    cost = getattr(self, 'last_total_cost', 0)
                    
                    self.stats_manager.record_chat(
                        self.model_id,
                        "received",
                        token_count=token_count,
                        prompt_tokens=self.last_tokens_prompt,
                        completion_tokens=self.last_tokens_completion,
                        cost=cost,
                        session_id=self.session_id
                    )
            
            # Display and store the response
            if partial_response:
                self._display_response(partial_response.strip())
                self.messages.append({"role": "assistant", "content": partial_response.strip()})
            
            return partial_response.strip() if partial_response else None

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

        # Use cached colors
        colors = self._cached_colors
        
        # Format tool results if present
        if hasattr(self, 'last_tool_results') and self.last_tool_results:
            formatted_text = response_text
        else:
            # Format regular response
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
            
            formatted_text = '\n\n'.join(sections)
        
        # Create the footer with cost and response time
        footer_parts = []
        
        # Add response time
        if hasattr(self, 'last_response_time'):
            minutes = int(self.last_response_time // 60)
            seconds = self.last_response_time % 60
            time_display = f"{seconds:.2f} secs"
            if minutes > 0:
                time_display = f"{minutes} min {seconds:.2f} secs"
            response_time = f"Response Time: {time_display}"
            footer_parts.append(response_time)
        
        # Add cost information
        if hasattr(self, 'last_total_cost') and hasattr(self, 'last_tokens_prompt') and hasattr(self, 'last_tokens_completion'):
            if self.last_total_cost > 0:
                footer_parts.append(f"Cost: ${self.last_total_cost:.6f}")
            elif self.last_total_cost == 0:
                footer_parts.append("Cost: Free")
            else:
                footer_parts.append("Cost: Not Found")
            
            tokens = f"Tokens: {self.last_tokens_prompt}+{self.last_tokens_completion}"
            if self.provider == 'openrouter' and hasattr(self, 'last_native_tokens_prompt') and hasattr(self, 'last_native_tokens_completion'):
                if (self.last_native_tokens_prompt != self.last_tokens_prompt or 
                    self.last_native_tokens_completion != self.last_tokens_completion) and \
                    self.last_native_tokens_prompt > 0 and self.last_native_tokens_completion > 0:
                    tokens += f" (Native: {self.last_native_tokens_prompt}+{self.last_native_tokens_completion})"
            footer_parts.append(tokens)
        
        footer = f"[bold {colors['cost']}]{' | '.join(footer_parts)}[/]"
        
        self.console.print(
            Panel(
                Markdown(formatted_text),
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

            # Set up the chat directory path first
            if self.provider == 'openrouter':
                company, model_name = self.model_id.split('/')
                company = sanitize_path(company)
                model_name = sanitize_path(model_name)
                chat_dir = os.path.join(chats_dir, 'openrouter', company, model_name)
            else:
                chat_dir = os.path.join(chats_dir, sanitize_path(self.provider))
            os.makedirs(chat_dir, exist_ok=True)

            # Handle custom name logic
            if not custom_name and self.last_save_name:
                custom_name = self.last_save_name
            elif custom_name:
                self.last_save_name = custom_name

            # Sanitize custom name if it exists
            if custom_name:
                custom_name = sanitize_path(custom_name)

            # Generate timestamp and base filename
            timestamp = self.start_time.strftime("%Y%m%d_%I%M%p")
            base_filename = f"{custom_name}_{timestamp}" if custom_name else f"chat_{timestamp}"

            # Check if we should reuse existing files
            should_reuse_files = False
            if custom_name and self.last_save_path and os.path.exists(os.path.dirname(self.last_save_path)):
                last_save_name = os.path.splitext(os.path.basename(self.last_save_path))[0]
                if last_save_name.startswith(custom_name + "_"):
                    should_reuse_files = True

            # Set up file paths
            if should_reuse_files:
                json_filepath = self.last_save_path
                text_filepath = os.path.splitext(self.last_save_path)[0] + ".txt"
            else:
                json_filepath = os.path.join(chat_dir, f"{base_filename}.json")
                text_filepath = os.path.join(chat_dir, f"{base_filename}.txt")

            # Store the json filepath for future saves
            self.last_save_path = json_filepath

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

            # Add RAG information if available
            if self.chroma_manager and self.chroma_manager.store_name:
                chat_data["agent"] = {
                    "store": self.chroma_manager.store_name,
                    "model": self.chroma_manager.embedding_model_name
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
                
                if total_prompt_tokens > 0 or total_completion_tokens > 0:
                    f.write(f"Total Tokens: {total_prompt_tokens + total_completion_tokens} ")
                    f.write(f"(Prompt: {total_prompt_tokens}, Completion: {total_completion_tokens})\n")
                    if total_cost > 0:
                        f.write(f"Total Cost: ${total_cost:.6f}\n")
                    else:
                        f.write("Total Cost: Free\n")

                if self.chroma_manager and self.chroma_manager.store_name:
                    f.write(f"RAG Store: {self.chroma_manager.store_name}\n")
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
        """Main chat loop"""
        def display_welcome():
            # Start a new chat session
            if self.stats_manager:
                self.session_id = self.stats_manager.record_session_start({
                    'id': self.model_id,
                    'name': self.model_name,
                    'provider': self.provider
                })
                
                # Track this model as recently used
                self._update_model_as_recent()

            # Display ACT logo
            logo = """[bold cyan]
    ╔═══╗╔═══╗╔════╗
    ║╔═╗║║╔═╗║║╔╗╔╗║
    ║║ ║║║║ ╚╝╚╝║║╚╝
    ║╔═╗║║║ ╔╗  ║║  
    ║║ ║║║╚═╝║  ║║  
    ╚╝ ╚╝╚═══╝  ╚╝  
[/bold cyan]"""
            
            # Use cached colors
            colors = self._cached_colors
            
            # Get RAG status if enabled
            agent_status = ""
            if self.chroma_manager and self.chroma_manager.store_name:
                agent_status = f" [bold cyan]〈RAG Store: {self.chroma_manager.store_name}〉[/]"
            
            welcome_text = (
                f"{logo}\n"
                f"[bold {colors['ai_name']}]{self.model_name}[/] [bold {colors['instruction_name']}][{self.instruction_name}][/]{agent_status}\n\n"
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
                "   - /fav - Add/remove current model to/from favorites\n"
                "   - /save [name] - Save the chat history (optional custom name)\n"
                "   - /clear - Clear the screen and chat history\n"
                "   - /insert - Insert multiline text (end with END on new line)\n"
                "   - /end - End the chat session\n"
                "❌ Type 'exit', 'quit', or press Ctrl+C to end the session"
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
            try:
                # Add a blank line for spacing
                self.console.print()
                
                # Clear any active live displays
                if hasattr(self.console, '_live') and self.console._live:
                    try:
                        self.console._live.stop()
                    except:
                        pass
                    self.console._live = None
                
                # Show goodbye message with fade effect
                with self.console.status("[bold cyan]👋[/bold cyan]", spinner="dots") as status:
                    time.sleep(0.5)
                    self.console.print(f"[bold cyan]{message}[/bold cyan]")
                    time.sleep(1)
                
                # Clear screen and show main menu
                os.system('cls' if os.name == 'nt' else 'clear')
                display_main_menu()
            except Exception as e:
                # If anything fails, just print the message
                self.console.print(f"\n[bold cyan]{message}[/bold cyan]")
        
        # Clear screen and show welcome message at start
        os.system('cls' if os.name == 'nt' else 'clear')
        display_welcome()
        
        try:
            while True:
                try:
                    user_input = self.console.input("[bold yellow]You: [/bold yellow]")
                    
                    # Check for commands
                    if user_input.strip().startswith('/') or user_input.strip().lower() in ['bye', 'exit', 'quit', 'cya', 'adios']:
                        command = user_input.strip().lower()
                        
                        # Record command message
                        if self.stats_manager:
                            self.stats_manager.record_chat(self.model_id, "sent", is_command=True, session_id=self.session_id)
                        
                        if command in ['bye', 'exit', 'quit', 'cya', 'adios']:
                            self.logger.info(f"Chat session ended by user ({command} command)")
                            if self.stats_manager and self.session_id:
                                self.stats_manager.record_session_end(self.session_id)
                            exit_chat("Chat session ended. Thanks for using ACT!")
                            break
                        elif command.startswith('/save'):
                            # Extract name if provided
                            parts = command.split(maxsplit=1)
                            custom_name = parts[1] if len(parts) > 1 else None
                            self.save_chat(custom_name)
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
                            if self.stats_manager and self.session_id:
                                self.stats_manager.record_session_end(self.session_id)
                            exit_chat("Chat session ended. Thanks for using ACT!")
                            break
                        elif command == '/help':
                            self._display_help()
                            continue
                        elif command == '/info':
                            self._display_info()
                            continue
                        elif command == '/fav':
                            self._display_fav()
                            continue
                        else:
                            self.console.print(f"[yellow]Unknown command: {command}[/yellow]")
                            continue
                    
                    # Process the message
                    self.send_message(user_input)
                
                except KeyboardInterrupt:
                    self.logger.warning("Chat interrupted by user")
                    if self.stats_manager and self.session_id:
                        self.stats_manager.record_session_end(self.session_id)
                    self.console.print("\n")  # Add newline for cleaner output
                    exit_chat("Chat interrupted. Thanks for using ACT!")
                    break
                except Exception as e:
                    self.logger.error(f"Error in chat loop: {e}", exc_info=True)
                    if self.stats_manager and self.session_id:
                        self.stats_manager.record_session_end(self.session_id)
                    self.console.print(f"[bold red]Error: {str(e)}[/bold red]")
                    continue
        
        except Exception as e:
            self.logger.error(f"Fatal error in chat loop: {e}", exc_info=True)
            if self.stats_manager and self.session_id:
                self.stats_manager.record_session_end(self.session_id)
            self.console.print(f"[bold red]Fatal error: {e}[/bold red]")
            exit_chat("Exiting due to error")

    def _display_help(self):
        """Display detailed help information"""
        help_text = (
            "[bold cyan]Available Commands:[/bold cyan]\n"
            "  [bold yellow]/help[/bold yellow]    - Display this help message\n"
            "  [bold yellow]/info[/bold yellow]    - Display chat session information\n"
            "  [bold yellow]/fav[/bold yellow]     - Add/remove current model to/from favorites\n"
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

        # Use cached colors
        colors = self._cached_colors

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
        usage_info = ["\n[bold cyan]Usage Information:[/bold cyan]"]
        
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
        
        if total_prompt_tokens > 0 or total_completion_tokens > 0:
            usage_info.extend([
                f"  Total Prompt Tokens: {total_prompt_tokens}",
                f"  Total Completion Tokens: {total_completion_tokens}",
                f"  Total Tokens: {total_prompt_tokens + total_completion_tokens}",
                f"  Total Cost: ${total_cost:.6f}" if total_cost > 0 else "  Total Cost: Free"
            ])
        else:
            usage_info.append("  No token usage data available")

        # Add RAG information if available
        agent_info = []
        if self.chroma_manager and self.chroma_manager.store_name:
            agent_info.extend([
                "\n[bold cyan]RAG Information:[/bold cyan]",
                f"Store: {self.chroma_manager.store_name}",
                f"Embedding Model: {self.chroma_manager.embedding_model_name}"
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

    def _calculate_openai_cost(self, prompt_tokens, completion_tokens, model_id):
        """Calculate cost for OpenAI models using pricing dictionary lookup"""
        for pricing in OPENAI_PRICING:
            if pricing["pattern"] in model_id:
                # Check exclusion pattern if present
                if "exclude" in pricing and pricing["exclude"] in model_id:
                    continue
                # Use audio pricing if applicable
                if "audio" in model_id and "audio_input" in pricing:
                    input_rate = pricing["audio_input"]
                    output_rate = pricing["audio_output"]
                else:
                    input_rate = pricing["input"]
                    output_rate = pricing["output"]
                return (input_rate * prompt_tokens / 1000) + (output_rate * completion_tokens / 1000)
        
        # Default pricing for unknown models
        return (DEFAULT_OPENAI_PRICING["input"] * prompt_tokens / 1000) + \
               (DEFAULT_OPENAI_PRICING["output"] * completion_tokens / 1000) 

    def _calculate_anthropic_cost(self, prompt_tokens, completion_tokens, model_id):
        """Calculate cost for Anthropic models using pricing dictionary lookup"""
        for pricing in ANTHROPIC_PRICING:
            if pricing["pattern"] in model_id:
                return (pricing["input"] * prompt_tokens / 1000) + (pricing["output"] * completion_tokens / 1000)
        
        # Default pricing for unknown models
        return (DEFAULT_ANTHROPIC_PRICING["input"] * prompt_tokens / 1000) + \
               (DEFAULT_ANTHROPIC_PRICING["output"] * completion_tokens / 1000) 

    def _display_fav(self):
        """Add or remove current model from favorites"""
        try:
            # Create favorites.json path
            favorites_path = os.path.join(os.path.dirname(__file__), 'favorites.json')
            
            # Load existing favorites
            if os.path.exists(favorites_path):
                with open(favorites_path, 'r') as f:
                    favorites = json.load(f)['favorites']
            else:
                favorites = []
            
            # Format display name
            display_name = self.model_name
            if self.provider == 'openrouter' and display_name.startswith(f"{self.model_id.split('/')[0].title()}: "):
                # Name already has provider prefix, use it as is
                display_name = self.model_name
            elif self.provider == 'openrouter' and '/' in self.model_id:
                # Add provider prefix if not present
                company = self.model_id.split('/')[0].title()
                display_name = f"{company}: {display_name}"
            
            # Check if model is already in favorites
            model_id = self.model_id
            is_favorite = any(f['id'] == model_id for f in favorites)
            
            if is_favorite:
                # Remove from favorites
                favorites = [f for f in favorites if f['id'] != model_id]
                self.console.print(f"[yellow]Removed {display_name} from favorites[/yellow]")
            else:
                # For OpenRouter models, get description from API
                if self.provider == 'openrouter':
                    try:
                        response = requests.get(
                            f"https://openrouter.ai/api/v1/models",
                            headers=get_openrouter_headers(self.api_key)
                        )
                        if response.status_code == 200:
                            data = response.json()
                            if 'data' in data:
                                for model in data['data']:
                                    if model['id'] == model_id:
                                        description = model.get('description')
                                        break
                                else:
                                    description = f"{display_name} ({self.provider})"
                        else:
                            description = f"{display_name} ({self.provider})"
                    except:
                        description = f"{display_name} ({self.provider})"
                else:
                    # For other providers, try to get description from models.json
                    description = None
                    models_path = os.path.join(os.path.dirname(__file__), 'models.json')
                    if os.path.exists(models_path):
                        try:
                            with open(models_path, 'r') as f:
                                models = json.load(f)['models']
                                for model in models:
                                    if model['id'] == model_id:
                                        description = model.get('description')
                                        break
                        except:
                            pass
                    if not description:
                        description = f"{display_name} ({self.provider})"
                
                # Add to favorites
                favorite = {
                    'id': model_id,
                    'name': display_name,
                    'provider': self.provider,
                    'description': description
                }
                favorites.append(favorite)
                self.console.print(f"[green]Added {display_name} to favorites[/green]")
            
            # Save updated favorites
            with open(favorites_path, 'w') as f:
                json.dump({'favorites': favorites}, f, indent=4)
            
        except Exception as e:
            self.logger.error(f"Error managing favorites: {e}", exc_info=True)
            self.console.print(f"[bold red]Error managing favorites: {e}[/bold red]")

    def _execute_tool_call(self, tool_call):
        """Execute a tool call and return the result"""
        return self.tools_manager.execute_tool(tool_call)

    def _update_model_as_recent(self):
        """Update this model as the most recently used model in models.json"""
        try:
            # Create a model config object
            model_config = {
                'id': self.model_id,
                'name': self.model_name,
                'description': f"{self.model_name} ({self.provider})",
                'context_window': 4096,  # Default value
                'max_tokens': self.max_tokens or 4096,
                'provider': self.provider,
                'recent': True  # Mark as recent
            }
            
            # Load the current models.json file
            models_path = os.path.join(os.path.dirname(__file__), 'models.json')
            with open(models_path, 'r') as f:
                models_data = json.load(f)
            
            # Remove the recent flag from any existing models
            for model in models_data['models']:
                if 'recent' in model:
                    del model['recent']
            
            # Check if the model is already in models.json
            model_exists = False
            for model in models_data['models']:
                if model['id'] == self.model_id:
                    model['recent'] = True
                    model_exists = True
                    break
            
            # If it's a new model, add it to models.json
            if not model_exists:
                models_data['models'].append(model_config)
            
            # Save the updated models.json file
            with open(models_path, 'w') as f:
                json.dump(models_data, f, indent=4)
            
        except Exception as e:
            # Just log the error but don't interrupt the chat
            if hasattr(self, 'logger'):
                self.logger.error(f"Error updating recent model: {e}", exc_info=True)