from imports import *
from utils import setup_logging
from managers import (
    SettingsManager,
    SystemInstructionsManager,
    DataManager,
    ChromaManager,
    StatsManager,
    ToolsManager
)
from chat import AIChat, OpenRouterAPI, get_openrouter_headers
from typing import Dict, Any, List

class AIChatApp:
    def __init__(self, logger, console):
        """Initialize AI Chat Application"""
        self.logger = logger
        self.console = console
        
        # Initialize file paths with detailed logging
        self.logger.debug("Initializing file paths...")
        self.settings_file = os.path.join(os.path.dirname(__file__), 'settings.json')
        self.custom_providers_file = os.path.join(os.path.dirname(__file__), 'custom_providers.json')
        self.logger.debug(f"Settings file path: {self.settings_file}")
        self.logger.debug(f"Custom providers file path: {self.custom_providers_file}")
        
        # Initialize managers
        self.instructions_manager = SystemInstructionsManager(logger, console)
        self.settings_manager = SettingsManager(logger, console)
        self.data_manager = DataManager(logger, console)
        self.chroma_manager = ChromaManager(logger, console)
        self.stats_manager = StatsManager(logger, console)
        
        # Load models from JSON
        try:
            models_path = os.path.join(os.path.dirname(__file__), 'models.json')
            self.logger.debug(f"Loading models from: {models_path}")
            
            if not os.path.exists(models_path):
                self.logger.error(f"models.json not found at: {models_path}")
                raise FileNotFoundError(f"models.json not found at: {models_path}")
                
            with open(models_path, 'r', encoding='utf-8') as f:
                self.models_config = json.load(f)['models']
                self.logger.info(f"Successfully loaded {len(self.models_config)} models")
            
            # Load custom providers
            self.custom_providers = self._load_custom_providers()
            
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
            
            # Try to load last used store
            settings = self._load_settings()
            agent_enabled = settings.get('agent', {}).get('enabled', False)
            if agent_enabled:
                last_store = settings.get('agent', {}).get('last_store')
                if last_store:
                    if self.chroma_manager.load_store(last_store):
                        self.logger.info(f"Loaded last used store: {last_store}")
                    else:
                        self.logger.warning(f"Failed to load last store: {last_store}")
            
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
        model_id = model_config['id']
        provider = model_config.get('provider', 'unknown')
        display_name = model_config['name']
        
        # For OpenRouter models, handle provider prefix
        if provider == 'openrouter' and '/' in model_id:
            # If display name already has a proper format (Provider: Name), use it as is
            if ':' in display_name and not display_name.startswith('meta-'):
                # Keep the display name as is since it's already properly formatted
                pass
            else:
                # Extract company name from model ID and format display name
                company = model_id.split('/')[0].title()
                # Remove any existing company/provider prefix to avoid duplication
                if ':' in display_name:
                    display_name = display_name.split(':', 1)[1].strip()
                # Add company prefix
                display_name = f"{company}: {display_name}"
        
        if not any(f['id'] == model_id for f in self.favorites):
            # For OpenRouter models, use the description from the API response
            if provider == 'openrouter':
                description = model_config.get('description', f"{display_name}")
                self.console.print(f"[green]Added {display_name} to favorites[/green]")
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
                    description = display_name
            
            favorite = {
                'id': model_id,
                'name': display_name,
                'provider': provider,
                'description': description
            }
            self.favorites.append(favorite)
            self.save_favorites()
        else:
            self.console.print(f"[yellow]{display_name} is already in favorites[/yellow]")

    def remove_from_favorites(self, model_id):
        """Remove a model from favorites"""
        self.favorites = [f for f in self.favorites if f['id'] != model_id]
        self.save_favorites()

    def sort_favorites(self, sort_key="name"):
        """Sort favorites list based on the given key"""
        if sort_key == "name":
            self.favorites.sort(key=lambda x: x['name'].lower())
        elif sort_key == "provider":
            self.favorites.sort(key=lambda x: (x['provider'].lower(), x['name'].lower()))
        self.save_favorites()

    def reload_favorites(self):
        """Reload favorites from file"""
        if os.path.exists(self.favorites_path):
            with open(self.favorites_path, 'r') as f:
                self.favorites = json.load(f)['favorites']
        else:
            self.favorites = []

    def manage_favorites(self):
        """Display favorites management menu"""
        # Reload favorites from file
        self.reload_favorites()
        
        if not self.favorites:
            self.console.print("[yellow]No favorite models yet[/yellow]")
            return

        sort_choices = [
            ("Sort by Name", "name"),
            ("Sort by Provider", "provider"),
            ("Back to Favorites", "back")
        ]

        while True:
            choices = [(f"{f['name']} ({f['provider']})", f) for f in self.favorites]
            choices.extend([
                ("═══ Sort Options ═══", None),
                ("Sort Favorites", "sort"),
                ("═══ Navigation ═══", None),
                ("Back", "back")
            ])

            questions = [
                inquirer.List('favorite',
                    message="Select favorite to manage or choose an action",
                    choices=choices,
                    carousel=True
                ),
            ]

            answer = inquirer.prompt(questions)
            if not answer or answer['favorite'] == "back":
                return

            if answer['favorite'] == "sort":
                sort_question = [
                    inquirer.List('sort_option',
                        message="Select sorting method",
                        choices=sort_choices,
                        carousel=True
                    ),
                ]
                
                sort_answer = inquirer.prompt(sort_question)
                if sort_answer and sort_answer['sort_option'] != "back":
                    self.sort_favorites(sort_answer['sort_option'])
                    self.console.print(f"[green]Favorites sorted by {sort_answer['sort_option']}[/green]")
                continue

            if isinstance(answer['favorite'], dict):
                selected = answer['favorite']
                
                # Display model description in a panel with enhanced styling
                self.console.print()  # Add spacing
                self.console.print(Panel(
                    f"[bold white]Provider:[/bold white] [cyan]{selected['provider'].upper()}[/cyan]\n\n"
                    f"[bold white]Description:[/bold white]\n[cyan]{selected.get('description', 'No description available')}[/cyan]",
                    title=f"[bold cyan]★ {selected['name']}[/bold cyan]",
                    title_align="left",
                    border_style="bright_blue",
                    padding=(1, 2),
                    highlight=True
                ))
                self.console.print()  # Add spacing
                
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
                    continue

                if action_answer['action'] == "chat":
                    if self.start_chat(selected):
                        return  # Return to main menu after chat ends
                elif action_answer['action'] == "remove":
                    self.remove_from_favorites(selected['id'])
                    self.console.print(f"[green]Removed {selected['name']} from favorites[/green]")
                    if not self.favorites:
                        return

    def select_openrouter_model(self):
        """Display nested menu for OpenRouter model selection"""
        try:
            if not self.openrouter:
                self.console.print("[bold red]OpenRouter API key not found. Please add it to your .env file.[/bold red]")
                return None
            
            self.console.print("[cyan]Fetching available models...[/cyan]")
            try:
                models = self.openrouter.fetch_models()
                self.console.print(f"[green]Found {len(models)} available models[/green]")
            except Exception as e:
                self.console.print(f"[bold red]Error fetching models: {e}[/bold red]")
                return None
            
            # Pass show_recent=True to display recent models first
            grouped_models = self.openrouter.group_models_by_company(models, show_recent=True)
            if not grouped_models:
                self.console.print("[bold yellow]No models available[/bold yellow]")
                return None
            
            companies = list(grouped_models.keys())
            # Make sure Recent is first if it exists
            if 'Recent' in companies:
                companies.remove('Recent')
                companies.sort()
                companies.insert(0, 'Recent')
            else:
                companies.sort()
            
            self.console.print(f"\n[cyan]Available Companies ({len(companies)}):[/cyan]")
            for company in companies:
                models = grouped_models[company]
                if company == 'Recent':
                    self.console.print(f"[bold cyan]{company}[/bold cyan] ({len(models)} most recently added models)")
                else:
                    # Count favorited models instead of featured ones
                    starred_models = sum(1 for m in models if any(f['id'] == m['id'] for f in self.favorites))
                    self.console.print(
                        f"[cyan]{company}: {len(models)} models "
                        f"({starred_models} ★)[/cyan]"
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
            company_models = grouped_models[selected_company]
            model_choices = []
            
            self.console.print(f"\n[cyan]Models from {selected_company}:[/cyan]")
            
            for model in company_models:
                name = model['name'].split(':')[-1].strip()
                
                try:
                    context_length = int(model.get('context_length', 0))
                    context = f"{context_length // 1000}K" if context_length else "Unknown"
                except (ValueError, TypeError):
                    context = "Unknown"
                
                try:
                    price = model.get('pricing', {}).get('prompt')
                    if price is None:
                        price_str = "Price N/A"
                    else:
                        price_float = float(price) if isinstance(price, str) else price
                        price_str = f"${price_float:.6f}"
                except (ValueError, TypeError):
                    price_str = "Price N/A"
                
                # Use star for favorited models instead of featured models
                starred = "★ " if any(f['id'] == model['id'] for f in self.favorites) else ""
                model_info = f"{starred}{name} (Context: {context}, {price_str})"
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
                return self.select_openrouter_model()
            
            return model_answer['model']
        
        except Exception as e:
            self.logger.error(f"Error selecting OpenRouter model: {e}", exc_info=True)
            self.console.print(f"[bold red]Error in OpenRouter model selection: {e}[/bold red]")
            return None

    def manage_instructions(self):
        """Display system instructions management menu"""
        while True:
            current_name = self.instructions_manager.get_current_name()
            instructions = self.instructions_manager.list_instructions()
            
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
                name_question = [
                    inquirer.Text('name',
                        message="Enter instruction name",
                        validate=lambda _, x: len(x) > 0
                    )
                ]
                
                name_answer = inquirer.prompt(name_question)
                if not name_answer:
                    continue
                
                os.system('cls' if os.name == 'nt' else 'clear')
                
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
                    content = '\n'.join(content_lines)
                    
                    self.console.print("\n[bold cyan]Preview of your instruction:[/bold cyan]")
                    self.console.print(
                        Panel(
                            content,
                            title=f"[bold white]{name_answer['name']}[/bold white]",
                            border_style="cyan"
                        )
                    )
                    
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
                
                instruction_choices = [
                    (f"{i['name']}", i['name']) 
                    for i in instructions 
                    if i['name'] != 'Default'
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
                
                instruction_choices = [
                    (f"{i['name']}", i) for i in instructions
                ]
                instruction_choices.append(("Back", None))
                
                view_question = [
                    inquirer.List('instruction',
                        message="Select instruction to view",
                        choices=instruction_choices,
                        carousel=True
                    ),
                ]
                
                view_answer = inquirer.prompt(view_question)
                if view_answer and view_answer['instruction']:
                    instruction = view_answer['instruction']
                    is_selected = instruction['name'] == current_name
                    self.console.print(
                        Panel(
                            f"[bold]Content:[/bold]\n{instruction['content']}",
                            title=f"[{'green' if is_selected else 'white'}]{instruction['name']}{'  [Selected]' if is_selected else ''}[/]",
                            border_style="green" if is_selected else "white"
                        )
                    )
                    self.console.input("\nPress Enter to continue...")

    def manage_settings(self):
        """Display settings management menu"""
        while True:
            choices = [
                ("=== Application Settings ===", None),
                ("🔍 Appearance Settings", "appearance"),
                ("🔍 Codebase Search Settings", "codebase"),
                ("📊 ACT Statistics", "statistics"),
                ("=== Data Management ===", None),
                ("🗑️ Clear All Logs", "clear_logs"),
                ("🗑️ Clear Chat History", "clear_chats"),
                ("🗑️ Clear Statistics DB", "clear_db"),
                ("=== Navigation ===", None),
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
            elif answer['setting'] == "appearance":
                self.settings_manager.manage_appearance_settings()
            elif answer['setting'] == "statistics":
                self.manage_statistics()
            elif answer['setting'] == "clear_logs":
                confirm = inquirer.confirm("Are you sure you want to clear all logs?", default=False)
                if confirm:
                    self.data_manager.clear_logs()
            elif answer['setting'] == "clear_chats":
                confirm = inquirer.confirm("Are you sure you want to clear all chat history?", default=False)
                if confirm:
                    self.data_manager.clear_chats()
            elif answer['setting'] == "clear_db":
                confirm = inquirer.confirm("Are you sure you want to clear all statistics data? This cannot be undone.", default=False)
                if confirm:
                    self.stats_manager.clear_db()
                    self.console.print("[green]Statistics database cleared successfully![/green]")

    def manage_ai_settings(self):
        """Display AI settings management menu"""
        while True:
            settings = self._load_settings()
            agent_enabled = settings.get('agent', {}).get('enabled', False)
            streaming_enabled = settings.get('streaming', {}).get('enabled', False)
            tools_enabled = settings.get('tools', {}).get('enabled', False)
            agent_active = (
                agent_enabled and 
                self.chroma_manager and 
                self.chroma_manager.vectorstore is not None and
                self.chroma_manager.store_name is not None
            )

            # Get agent status with icons
            if agent_active:
                agent_status = f"🟢 Active - Store: {self.chroma_manager.store_name}"
            elif agent_enabled:
                agent_status = "🟡 Enabled (No Store Selected)"
            else:
                agent_status = "⭕ Disabled"

            # Get streaming status
            streaming_status = "🟢 Enabled" if streaming_enabled else "⭕ Disabled"

            # Get tools status
            tools_status = "🟢 Enabled" if tools_enabled else "⭕ Disabled"

            # Get current instruction name
            current_instruction = self.instructions_manager.get_current_name()

            choices = [
                ("═══ AI Settings ═══", None),
                (f"🤖 RAG           〈{agent_status}〉", "agent"),
                (f"🤖 Streaming Mode   〈{streaming_status}〉", "streaming"),
                (f"🤖️ AI Tools         〈{tools_status}〉", "tools"),
                (f"🤖 System Instructions 〈{current_instruction}〉", "instructions"),
                ("📝 Model Context Settings", "contexts"),
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

            if answer['setting'] == "agent":
                self.manage_agent_settings(agent_status)
            elif answer['setting'] == "streaming":
                # Toggle streaming mode
                settings = self._load_settings()
                if 'streaming' not in settings:
                    settings['streaming'] = {}
                settings['streaming']['enabled'] = not streaming_enabled
                self._save_settings(settings)
                new_status = "enabled" if not streaming_enabled else "disabled"
                icon = "🟢" if not streaming_enabled else "⭕"
                self.console.print(f"{icon} Streaming mode {new_status}")
            elif answer['setting'] == "tools":
                self.manage_tools_settings()
            elif answer['setting'] == "instructions":
                self.manage_instructions()
            elif answer['setting'] == "contexts":
                self.manage_model_contexts()

    def manage_tools_settings(self):
        """Manage AI tools settings"""
        while True:
            settings = self._load_settings()
            tools_settings = settings.get('tools', {})
            tools_enabled = tools_settings.get('enabled', False)
            
            # Initialize tools manager if not already done
            if not hasattr(self, 'tools_manager'):
                self.tools_manager = ToolsManager(self.logger, self.settings_manager)
            
            available_tools = self.tools_manager.available_tools

            choices = [
                ("═══ AI Tools Settings ═══", None),
                (f"Tools Status: {'🟢 Enabled' if tools_enabled else '⭕ Disabled'}", None),
                ("Toggle Tools", "toggle"),
                ("═══ Available Tools ═══", None)
            ]

            # Add available tools with their status
            for tool_name, tool_info in available_tools.items():
                status = "✓" if tool_info.get('enabled', True) else "✗"
                choices.append((
                    f"{tool_name.title()}: {status} - {tool_info.get('description', 'No description')}",
                    f"toggle_{tool_name}"
                ))

            choices.extend([
                ("═══ Navigation ═══", None),
                ("Back", "back")
            ])

            questions = [
                inquirer.List('action',
                    message="Manage AI Tools",
                    choices=choices,
                    carousel=True
                ),
            ]

            answer = inquirer.prompt(questions)
            if not answer or answer['action'] == "back":
                break

            if answer['action'] is None:
                continue

            if answer['action'] == "toggle":
                # Toggle overall tools feature
                tools_settings['enabled'] = not tools_enabled
                settings['tools'] = tools_settings
                self._save_settings(settings)
                new_status = "enabled" if not tools_enabled else "disabled"
                icon = "🟢" if not tools_enabled else "⭕"
                self.console.print(f"{icon} AI Tools {new_status}")
            
            elif answer['action'].startswith("toggle_"):
                # Toggle individual tool
                tool_name = answer['action'][7:]  # Remove "toggle_" prefix
                if tool_name in available_tools:
                    current_state = available_tools[tool_name].get('enabled', True)
                    success = self.tools_manager.enable_tool(tool_name, not current_state)
                    if success:
                        new_state = "enabled" if not current_state else "disabled"
                        icon = "🟢" if not current_state else "⭕"
                        self.console.print(f"{icon} Tool '{tool_name}' {new_state}")
                    else:
                        self.console.print(f"[red]Failed to change tool state for '{tool_name}'[/red]")

    def manage_agent_settings(self, agent_status):
        """Manage RAG settings"""
        while True:
            settings = self._load_settings()
            agent_settings = settings.get('agent', {})
            agent_enabled = agent_settings.get('enabled', False)
            stores = self.chroma_manager.list_stores()  # Get list of stores
            
            current_store = "None"
            if self.chroma_manager and self.chroma_manager.store_name:
                current_store = self.chroma_manager.store_name

            choices = [
                ("═══ RAG Settings ═══", None),
                (f"Current Status: {self._get_agent_status_display()}", None),
                (f"Current Store: {current_store}", None),
                ("Select Embedding Model", "model"),
                ("Toggle RAG", "toggle"),
                ("Test Embeddings", "test_embeddings"),
                ("ChromaDB Settings", "chromadb_settings"),
            ]

            if agent_enabled:
                choices.extend([
                    ("═══ Store Management ═══", None),
                    ("Create New Store", "create"),
                    ("Select Store", "select"),
                    ("Delete Store", "delete"),
                ])

                if self.chroma_manager and self.chroma_manager.store_name:
                    choices.extend([
                        ("Manage Store", "manage_store"),
                        ("Test Search", "test"),
                    ])

            choices.extend([
                ("═══ Navigation ═══", None),
                ("Back", "back")
            ])

            questions = [
                inquirer.List('action',
                    message="Manage RAG Settings",
                    choices=choices,
                    carousel=True
                ),
            ]

            answer = inquirer.prompt(questions)
            if not answer or answer['action'] == "back":
                break

            if answer['action'] == "toggle":
                # Toggle overall RAG feature
                new_status = not agent_enabled  # Store new status
                settings['agent']['enabled'] = new_status
                
                # Save last used store when disabling
                if agent_enabled and self.chroma_manager.store_name:
                    settings['agent']['last_store'] = self.chroma_manager.store_name
                
                # Save settings before initializing embeddings
                self._save_settings(settings)
                
                if not new_status:  # Disabling RAG
                    # Unload the store first if one is loaded
                    if self.chroma_manager.store_name:
                        self.chroma_manager.unload_store()
                    # Clear embeddings
                    self.chroma_manager.embeddings = None
                    self.chroma_manager.embedding_model_name = None
                    icon = "⭕"
                    self.console.print(f"{icon} RAG disabled")
                else:  # Enabling RAG
                    # Initialize embeddings first
                    success = self.chroma_manager.initialize_embeddings()
                    if success:
                        icon = "🟡"
                        self.console.print(f"{icon} RAG enabled")
                        # Try to load last used store if available
                        last_store = settings.get('agent', {}).get('last_store')
                        if last_store and last_store in self.chroma_manager.list_stores():
                            if self.chroma_manager.load_store(last_store):
                                self.console.print(f"[green]Loaded last used store: {last_store}[/green]")
                    else:
                        # If initialization fails, revert the settings
                        settings['agent']['enabled'] = False
                        self._save_settings(settings)
                        self.console.print("[red]Failed to initialize embeddings. RAG will remain disabled.[/red]")
                continue  # Continue the loop instead of returning

            elif answer['action'] == "test_embeddings":
                if not self.chroma_manager:
                    self.console.print("[yellow]Please enable RAG first[/yellow]")
                    continue
                self.chroma_manager.test_embeddings()

            elif answer['action'] == "manage_store":
                self.manage_store()

            elif answer['action'] == "chromadb_settings":
                if not self.chroma_manager:
                    self.console.print("[yellow]Please enable RAG first[/yellow]")
                    continue
                
                settings = self._load_settings()
                chromadb_settings = settings.get('chromadb', {})
                
                while True:
                    config_choices = [
                        ("=== ChromaDB Settings ===", None),
                        (f"Auto Add Files: {'✓' if chromadb_settings.get('auto_add_files', True) else '✗'}", "auto_add"),
                        (f"Max File Size (MB): {chromadb_settings.get('max_file_size_mb', 5)}", "max_size"),
                        (f"Search Results Limit: {chromadb_settings.get('search_results_limit', 10)}", "search_limit"),
                        ("Manage Exclude Patterns", "exclude"),
                        ("Manage File Types", "file_types"),
                        ("Back", "back")
                    ]
                    
                    config_question = [
                        inquirer.List('config_action',
                            message="Select setting to modify",
                            choices=config_choices,
                            carousel=True
                        ),
                    ]
                    
                    config_answer = inquirer.prompt(config_question)
                    if not config_answer or config_answer['config_action'] == "back":
                        break
                    
                    if config_answer['config_action'] == "auto_add":
                        current = chromadb_settings.get('auto_add_files', True)
                        chromadb_settings['auto_add_files'] = not current
                        self.console.print(f"[green]Auto add files {'enabled' if not current else 'disabled'}[/green]")
                    
                    elif config_answer['config_action'] == "max_size":
                        size_question = [
                            inquirer.Text('size',
                                message="Enter maximum file size in MB",
                                validate=lambda _, x: x.isdigit() and int(x) > 0,
                                default=str(chromadb_settings.get('max_file_size_mb', 5))
                            )
                        ]
                        
                        size_answer = inquirer.prompt(size_question)
                        if size_answer:
                            chromadb_settings['max_file_size_mb'] = int(size_answer['size'])
                            self.console.print("[green]Max file size updated[/green]")
                    
                    elif config_answer['config_action'] == "search_limit":
                        limit_question = [
                            inquirer.Text('limit',
                                message="Enter maximum number of search results",
                                validate=lambda _, x: x.isdigit() and int(x) > 0,
                                default=str(chromadb_settings.get('search_results_limit', 10))
                            )
                        ]
                        
                        limit_answer = inquirer.prompt(limit_question)
                        if limit_answer:
                            chromadb_settings['search_results_limit'] = int(limit_answer['limit'])
                            self.console.print("[green]Search results limit updated[/green]")
                    
                    elif config_answer['config_action'] == "exclude":
                        patterns = chromadb_settings.get('exclude_patterns', [])
                        while True:
                            pattern_choices = [
                                (f"Remove: {pattern}", f"remove_{pattern}") for pattern in patterns
                            ]
                            pattern_choices.extend([
                                ("Add New Pattern", "add"),
                                ("Back", "back")
                            ])
                            
                            pattern_question = [
                                inquirer.List('pattern_action',
                                    message="Manage exclude patterns",
                                    choices=pattern_choices,
                                    carousel=True
                                ),
                            ]
                            
                            pattern_answer = inquirer.prompt(pattern_question)
                            if not pattern_answer or pattern_answer['pattern_action'] == "back":
                                break
                            
                            if pattern_answer['pattern_action'] == "add":
                                new_pattern = inquirer.text(message="Enter new exclude pattern")
                                if new_pattern and new_pattern not in patterns:
                                    patterns.append(new_pattern)
                                    self.console.print(f"[green]Added pattern: {new_pattern}[/green]")
                            elif pattern_answer['pattern_action'].startswith("remove_"):
                                pattern = pattern_answer['pattern_action'][7:]
                                patterns.remove(pattern)
                                self.console.print(f"[green]Removed pattern: {pattern}[/green]")
                        
                        chromadb_settings['exclude_patterns'] = patterns
                    
                    elif config_answer['config_action'] == "file_types":
                        file_types = chromadb_settings.get('file_types', [])
                        while True:
                            type_choices = [
                                (f"Remove: {ft}", f"remove_{ft}") for ft in file_types
                            ]
                            type_choices.extend([
                                ("Add New File Type", "add"),
                                ("Back", "back")
                            ])
                            
                            type_question = [
                                inquirer.List('type_action',
                                    message="Manage file types",
                                    choices=type_choices,
                                    carousel=True
                                ),
                            ]
                            
                            type_answer = inquirer.prompt(type_question)
                            if not type_answer or type_answer['type_action'] == "back":
                                break
                            
                            if type_answer['type_action'] == "add":
                                new_type = inquirer.text(
                                    message="Enter new file type (e.g., .py)",
                                    validate=lambda _, x: x.startswith('.')
                                )
                                if new_type and new_type not in file_types:
                                    file_types.append(new_type)
                                    self.console.print(f"[green]Added file type: {new_type}[/green]")
                            elif type_answer['type_action'].startswith("remove_"):
                                ft = type_answer['type_action'][7:]
                                file_types.remove(ft)
                                self.console.print(f"[green]Removed file type: {ft}[/green]")
                        
                        chromadb_settings['file_types'] = file_types
                    
                    # Save settings after each change
                    settings['chromadb'] = chromadb_settings
                    self._save_settings(settings)

            elif answer['action'] == "create":
                name_question = [
                    inquirer.Text('name',
                        message="Enter store name",
                        validate=lambda _, x: len(x) > 0
                    )
                ]

                name_answer = inquirer.prompt(name_question)
                if name_answer:
                    if self.chroma_manager.create_store(name_answer['name']):
                        if inquirer.confirm("Would you like to process a directory now?", default=True):
                            dir_questions = [
                                inquirer.Text('directory',
                                    message="Enter directory path to process",
                                    validate=lambda _, x: os.path.exists(x.strip('"\''))
                                ),
                                inquirer.Confirm('include_subdirs',
                                    message="Include subdirectories?",
                                    default=True
                                ),
                            ]
                            dir_answer = inquirer.prompt(dir_questions)
                            if dir_answer:
                                directory_path = dir_answer['directory'].strip('"\'')
                                include_subdirs = dir_answer['include_subdirs']
                                self.chroma_manager.process_directory(directory_path, include_subdirs=include_subdirs)
                    else:
                        self.console.print("[red]Failed to create store[/red]")

            elif answer['action'] == "select" and stores:
                store_choices = [
                    ("None (Disable Store)", "none"),  # Add None option at the top
                ]
                store_choices.extend((store, store) for store in stores)
                store_choices.append(("Back", None))

                store_question = [
                    inquirer.List('store',
                        message="Select Store",
                        choices=store_choices,
                        carousel=True
                    ),
                ]

                store_answer = inquirer.prompt(store_question)
                if store_answer:
                    if store_answer['store'] == "none":
                        self.chroma_manager.unload_store()
                    elif store_answer['store']:
                        if self.chroma_manager.load_store(store_answer['store']):
                            if inquirer.confirm("Would you like to refresh the store to check for new files?", default=False):
                                self.chroma_manager.refresh_store()

            elif answer['action'] == "delete" and stores:
                store_choices = [(store, store) for store in stores]
                store_choices.append(("Back", None))

                store_question = [
                    inquirer.List('store',
                        message="Select Store to Delete",
                        choices=store_choices,
                        carousel=True
                    ),
                ]

                store_answer = inquirer.prompt(store_question)
                if store_answer and store_answer['store']:
                    confirm = inquirer.confirm(
                        f"Are you sure you want to delete store '{store_answer['store']}'?",
                        default=False
                    )
                    if confirm:
                        if self.chroma_manager.delete_store(store_answer['store']):
                            self.console.print(f"[green]Deleted store: {store_answer['store']}[/green]")
                        else:
                            self.console.print("[red]Failed to delete store[/red]")

            elif answer['action'] == "test":
                query = inquirer.text(message="Enter a test query")
                if query:
                    self.console.print("\n[cyan]Searching for relevant context...[/cyan]")
                    results = self.chroma_manager.search_context(query)
                    if results:
                        self.console.print("\n[green]Found relevant files:[/green]")
                        for i, result in enumerate(results, 1):
                            self.console.print(Panel(
                                result,
                                title=f"[bold cyan]Result {i}[/bold cyan]",
                                border_style="cyan"
                            ))
                    else:
                        self.console.print("[yellow]No relevant context found[/yellow]")
                    self.console.input("\nPress Enter to continue...")

            elif answer['action'] == "model":
                if not self.chroma_manager:
                    self.chroma_manager = ChromaManager(self.logger, self.console)
                self.chroma_manager.select_embedding_model()

    def manage_store(self):
        """Manage current store contents and operations"""
        if not self.chroma_manager or not self.chroma_manager.store_name:
            self.console.print("[yellow]Please select a store first[/yellow]")
            return

        while True:
            # Get current store info
            store_name = self.chroma_manager.store_name
            doc_count = self.chroma_manager.vectorstore._collection.count() if self.chroma_manager.vectorstore else 0

            choices = [
                ("═══ Store Information ═══", None),
                (f"Current Store: {store_name}", None),
                (f"Document Count: {doc_count}", None),
                ("═══ Content Management ═══", None),
                ("View Store Contents", "view"),
                ("Add Files", "add_files"),
                ("Add Directory", "add_directory"),
                ("Refresh Store", "refresh"),
                ("═══ Document Management ═══", None),
                ("List All Documents", "list_docs"),
                ("Delete Documents", "delete_docs"),
                ("Update Document", "update_doc"),
                ("Clear All Documents", "clear_store"),
                ("═══ Navigation ═══", None),
                ("Back", "back")
            ]

            questions = [
                inquirer.List('action',
                    message="Manage Store",
                    choices=choices,
                    carousel=True
                ),
            ]

            answer = inquirer.prompt(questions)
            if not answer or answer['action'] == "back":
                break

            if answer['action'] == "view":
                # Show store contents with search
                query = inquirer.text(message="Enter search query (or press Enter to see all)")
                if query is not None:  # Allow empty query
                    self.console.print("\n[cyan]Searching store contents...[/cyan]")
                    results = self.chroma_manager.search_context(query if query else "", k=50)
                    if results:
                        self.console.print("\n[green]Store contents:[/green]")
                        for i, result in enumerate(results, 1):
                            self.console.print(Panel(
                                result,
                                title=f"[bold cyan]Result {i}[/bold cyan]",
                                border_style="cyan"
                            ))
                    else:
                        self.console.print("[yellow]No matching documents found[/yellow]")
                    self.console.input("\nPress Enter to continue...")

            elif answer['action'] == "list_docs":
                documents = self.chroma_manager.get_all_documents()
                if documents:
                    self.console.print("\n[green]All Documents in Store:[/green]")
                    for doc in documents:
                        self.console.print(Panel(
                            f"[bold white]ID:[/bold white] {doc['id']}\n\n"
                            f"[bold white]Content:[/bold white]\n{doc['content']}\n\n"
                            f"[bold white]Metadata:[/bold white]\n{json.dumps(doc['metadata'], indent=2)}",
                            title=f"[bold cyan]Document[/bold cyan]",
                            border_style="cyan"
                        ))
                    self.console.input("\nPress Enter to continue...")
                else:
                    self.console.print("[yellow]No documents found in store[/yellow]")

            elif answer['action'] == "delete_docs":
                documents = self.chroma_manager.get_all_documents()
                if not documents:
                    self.console.print("[yellow]No documents found in store[/yellow]")
                    continue

                # Create choices for documents
                doc_choices = []
                for doc in documents:
                    # Get first line or truncate content for display
                    content_preview = doc['content'].split('\n')[0][:100] + "..."
                    doc_choices.append((
                        f"ID: {doc['id']} - {content_preview}",
                        doc['id']
                    ))
                doc_choices.append(("Back", None))

                # Allow multiple selection
                doc_question = [
                    inquirer.Checkbox('docs',
                        message="Select documents to delete (space to select, enter to confirm)",
                        choices=doc_choices[:-1]  # Exclude "Back" option
                    )
                ]

                doc_answer = inquirer.prompt(doc_question)
                if doc_answer and doc_answer['docs']:
                    if inquirer.confirm("Are you sure you want to delete these documents?", default=False):
                        self.chroma_manager.delete_documents(doc_answer['docs'])

            elif answer['action'] == "update_doc":
                documents = self.chroma_manager.get_all_documents()
                if not documents:
                    self.console.print("[yellow]No documents found in store[/yellow]")
                    continue

                # Create choices for documents
                doc_choices = []
                for doc in documents:
                    # Get first line or truncate content for display
                    content_preview = doc['content'].split('\n')[0][:100] + "..."
                    doc_choices.append((
                        f"ID: {doc['id']} - {content_preview}",
                        doc
                    ))
                doc_choices.append(("Back", None))

                # Select document to update
                doc_question = [
                    inquirer.List('doc',
                        message="Select document to update",
                        choices=doc_choices,
                        carousel=True
                    )
                ]

                doc_answer = inquirer.prompt(doc_question)
                if doc_answer and doc_answer['doc']:
                    doc = doc_answer['doc']
                    self.console.print(Panel(
                        doc['content'],
                        title=f"[bold cyan]Current Content - ID: {doc['id']}[/bold cyan]",
                        border_style="cyan"
                    ))
                    
                    self.console.print(
                        "\n[bold cyan]Enter new content below:[/bold cyan]\n"
                        "• You can paste multiple lines of text\n"
                        "• Press [bold]Enter[/bold] twice to start a new line\n"
                        "• Type [bold]END[/bold] on a new line and press Enter to finish\n"
                        "• Type [bold]CANCEL[/bold] on a new line to cancel"
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
                        self.console.print("\n[yellow]Update cancelled[/yellow]")
                        continue

                    if content_lines:
                        new_content = '\n'.join(content_lines)
                        if inquirer.confirm("Update this document?", default=True):
                            self.chroma_manager.update_document(doc['id'], new_content)

            elif answer['action'] == "clear_store":
                if inquirer.confirm(
                    "Are you sure you want to remove ALL documents from the store?",
                    default=False
                ):
                    self.chroma_manager.clear_store()

            elif answer['action'] == "add_files":
                files_question = [
                    inquirer.Text('files',
                        message="Enter file paths (comma-separated)",
                    )
                ]
                files_answer = inquirer.prompt(files_question)
                if files_answer:
                    files = [f.strip() for f in files_answer['files'].split(',')]
                    self.chroma_manager.refresh_store(files=files)

            elif answer['action'] == "add_directory":
                dir_question = [
                    inquirer.Text('directory',
                        message="Enter directory path to process",
                        default="."
                    ),
                    inquirer.Confirm('include_subdirs',
                        message="Include subdirectories?",
                        default=True
                    )
                ]
                dir_answer = inquirer.prompt(dir_question)
                if dir_answer:
                    self.chroma_manager.process_directory(
                        dir_answer['directory'],
                        include_subdirs=dir_answer['include_subdirs']
                    )

            elif answer['action'] == "refresh":
                if inquirer.confirm("Are you sure you want to refresh the store? This will reprocess all files.", default=True):
                    self.chroma_manager.refresh_store()

    def manage_model_contexts(self):
        """Display and manage context window settings for models"""
        while True:
            # Group models by provider
            providers = {}
            for model in self.models_config:
                provider = model.get('provider', 'unknown')
                if provider not in providers:
                    providers[provider] = []
                providers[provider].append(model)

            # Create choices list with provider headers and models
            choices = []
            for provider in sorted(providers.keys()):
                choices.append((f"=== {provider.upper()} Models ===", None))
                for model in providers[provider]:
                    context_window = model.get('context_window', 'N/A')
                    max_tokens = model.get('max_tokens', 'N/A')
                    choices.append((
                        f"{model['name']} (Context: {context_window}, Max Tokens: {max_tokens})",
                        model
                    ))
            
            choices.extend([
                ("=== Navigation ===", None),
                ("Back", "back")
            ])

            questions = [
                inquirer.List('model',
                    message="Select model to configure context",
                    choices=choices,
                    carousel=True
                ),
            ]

            answer = inquirer.prompt(questions)
            if not answer or answer['model'] == "back":
                break

            if answer['model']:
                model = answer['model']
                current_context = model.get('context_window', 0)
                current_max_tokens = model.get('max_tokens', 0)

                self.console.print(f"\n[cyan]Current settings for {model['name']}:[/cyan]")
                self.console.print(f"Context Window: {current_context}")
                self.console.print(f"Max Tokens: {current_max_tokens}\n")

                context_question = [
                    inquirer.Text('context',
                        message="Enter new context window size (or press Enter to keep current)",
                        default=str(current_context)
                    ),
                    inquirer.Text('max_tokens',
                        message="Enter new max tokens (or press Enter to keep current)",
                        default=str(current_max_tokens)
                    )
                ]

                context_answer = inquirer.prompt(context_question)
                if context_answer:
                    try:
                        new_context = int(context_answer['context'])
                        new_max_tokens = int(context_answer['max_tokens'])

                        # Update the model in models_config
                        for m in self.models_config:
                            if m['id'] == model['id']:
                                m['context_window'] = new_context
                                m['max_tokens'] = new_max_tokens
                                break

                        # Save updated config to models.json
                        models_path = os.path.join(os.path.dirname(__file__), 'models.json')
                        with open(models_path, 'w') as f:
                            json.dump({'models': self.models_config}, f, indent=4)

                        self.console.print(f"[green]Successfully updated context settings for {model['name']}[/green]")
                    except ValueError:
                        self.console.print("[red]Invalid input. Please enter numeric values.[/red]")
                    except Exception as e:
                        self.console.print(f"[red]Error updating settings: {e}[/red]")

    def manage_file_context(self):
        """Manage file context settings and ChromaDB stores"""
        while True:
            stores = self.chroma_manager.list_stores()
            current_store = self.chroma_manager.store_name

            choices = [
                ("=== File Context Settings ===", None),
                (f"Current Store: {current_store or 'None'}", None),
                ("Create New Store", "create"),
            ]

            if stores:
                choices.extend([
                    ("Select Store", "select"),
                    ("Delete Store", "delete"),
                ])

            if current_store:
                choices.extend([
                    ("Test Search", "test"),
                ])

            choices.extend([
                ("=== Configuration ===", None),
                ("Select Embedding Model", "model"),
                ("=== Navigation ===", None),
                ("Back", "back")
            ])

            questions = [
                inquirer.List('action',
                    message="Manage File Context",
                    choices=choices,
                    carousel=True
                ),
            ]

            answer = inquirer.prompt(questions)
            if not answer or answer['action'] == "back":
                break

            if answer['action'] == "create":
                name_question = [
                    inquirer.Text('name',
                        message="Enter store name",
                        validate=lambda _, x: len(x) > 0
                    )
                ]

                name_answer = inquirer.prompt(name_question)
                if name_answer:
                    if self.chroma_manager.create_store(name_answer['name']):
                        # Ask if user wants to process a directory
                        if inquirer.confirm("Would you like to process a directory now?", default=True):
                            dir_questions = [
                                inquirer.Text('directory',
                                    message="Enter directory path to process",
                                    validate=lambda _, x: os.path.exists(x.strip('"\''))
                                ),
                                inquirer.Confirm('include_subdirs',
                                    message="Include subdirectories?",
                                    default=True
                                ),
                            ]
                            dir_answer = inquirer.prompt(dir_questions)
                            if dir_answer:
                                directory_path = dir_answer['directory'].strip('"\'')
                                include_subdirs = dir_answer['include_subdirs']
                                self.chroma_manager.process_directory(directory_path, include_subdirs=include_subdirs)
                    else:
                        self.console.print("[red]Failed to create store[/red]")

            elif answer['action'] == "select" and stores:
                store_choices = [
                    ("None (Disable Store)", "none"),  # Add None option at the top
                ]
                store_choices.extend((store, store) for store in stores)
                store_choices.append(("Back", None))

                store_question = [
                    inquirer.List('store',
                        message="Select Store",
                        choices=store_choices,
                        carousel=True
                    ),
                ]

                store_answer = inquirer.prompt(store_question)
                if store_answer:
                    if store_answer['store'] == "none":
                        self.chroma_manager.unload_store()
                    elif store_answer['store']:
                        if self.chroma_manager.load_store(store_answer['store']):
                            if inquirer.confirm("Would you like to refresh the store to check for new files?", default=False):
                                self.chroma_manager.refresh_store()

            elif answer['action'] == "delete" and stores:
                store_choices = [(store, store) for store in stores]
                store_choices.append(("Back", None))

                store_question = [
                    inquirer.List('store',
                        message="Select Store to Delete",
                        choices=store_choices,
                        carousel=True
                    ),
                ]

                store_answer = inquirer.prompt(store_question)
                if store_answer and store_answer['store']:
                    confirm = inquirer.confirm(
                        f"Are you sure you want to delete store '{store_answer['store']}'?",
                        default=False
                    )
                    if confirm:
                        if self.chroma_manager.delete_store(store_answer['store']):
                            self.console.print(f"[green]Deleted store: {store_answer['store']}[/green]")
                        else:
                            self.console.print("[red]Failed to delete store[/red]")

            elif answer['action'] == "model":
                if not self.chroma_manager:
                    self.chroma_manager = ChromaManager(self.logger, self.console)
                self.chroma_manager.select_embedding_model()

    def _load_custom_providers(self) -> List[Dict]:
        """Load custom providers from file"""
        try:
            if os.path.exists(self.custom_providers_file):
                with open(self.custom_providers_file, 'r', encoding='utf-8') as f:
                    return json.load(f)['providers']
            else:
                # Create default empty custom providers file
                with open(self.custom_providers_file, 'w', encoding='utf-8') as f:
                    json.dump({'providers': []}, f, indent=4)
                return []
        except Exception as e:
            self.logger.error(f"Error loading custom providers: {e}")
            return []

    def _save_custom_providers(self, providers: List[Dict]) -> None:
        """Save custom providers to file"""
        try:
            with open(self.custom_providers_file, 'w', encoding='utf-8') as f:
                json.dump({'providers': providers}, f, indent=4)
            # Refresh the in-memory list
            self.custom_providers = providers
        except Exception as e:
            self.logger.error(f"Error saving custom providers: {e}")

    def manage_custom_providers(self):
        """Manage custom API providers"""
        while True:
            # Display current custom providers
            if self.custom_providers:
                self.console.print("\n[bold cyan]Current Custom Providers:[/bold cyan]")
                for i, provider in enumerate(self.custom_providers):
                    self.console.print(f"[cyan]{i+1}.[/cyan] [bold]{provider['name']}[/bold] - {provider['base_url']}")
            else:
                self.console.print("\n[yellow]No custom providers configured[/yellow]")
            
            # Management options
            choices = [
                ("Add New Provider", "add"),
                ("Edit Provider", "edit"),
                ("Remove Provider", "remove"),
                ("Back", "back")
            ]

            questions = [
                inquirer.List('action',
                    message="Manage Custom Providers",
                    choices=choices,
                    carousel=True
                ),
            ]

            answer = inquirer.prompt(questions)
            if not answer or answer['action'] == "back":
                break

            if answer['action'] == "add":
                self._add_custom_provider()
            elif answer['action'] == "edit" and self.custom_providers:
                self._edit_custom_provider()
            elif answer['action'] == "remove" and self.custom_providers:
                self._remove_custom_provider()

    def _add_custom_provider(self):
        """Add a new custom API provider"""
        try:
            # Gather provider information
            provider_questions = [
                inquirer.Text('name',
                    message="Provider name",
                    validate=lambda _, x: bool(x.strip())
                ),
                inquirer.Text('base_url',
                    message="API base URL (e.g., http://localhost:8000/v1)",
                    validate=lambda _, x: bool(x.strip())
                ),
                inquirer.Text('api_key',
                    message="API key (leave empty if not required)"
                ),
                inquirer.Confirm('header_auth',
                    message="Use Authorization header for API key?",
                    default=True
                ),
                inquirer.Confirm('use_chat_completions_endpoint',
                    message="Append '/chat/completions' to base URL?",
                    default=True
                )
            ]
            
            provider_answers = inquirer.prompt(provider_questions)
            if not provider_answers:
                return
            
            # Gather model information
            models_list = []
            while True:
                self.console.print("\n[cyan]Add models for this provider[/cyan]")
                model_questions = [
                    inquirer.Text('id',
                        message="Model ID (as expected by the API)",
                        validate=lambda _, x: bool(x.strip())
                    ),
                    inquirer.Text('name',
                        message="Display name",
                        validate=lambda _, x: bool(x.strip())
                    ),
                    inquirer.Text('description',
                        message="Description",
                        default="Custom model"
                    ),
                    inquirer.Text('context_window',
                        message="Context window size",
                        validate=lambda _, x: x.isdigit() and int(x) > 0,
                        default="4096"
                    ),
                    inquirer.Text('max_tokens',
                        message="Max output tokens",
                        validate=lambda _, x: x.isdigit() and int(x) > 0,
                        default="2048"
                    ),
                    inquirer.Confirm('add_another',
                        message="Add another model?",
                        default=False
                    )
                ]
                
                model_answers = inquirer.prompt(model_questions)
                if not model_answers:
                    break
                
                models_list.append({
                    "id": model_answers['id'],
                    "name": model_answers['name'],
                    "description": model_answers['description'],
                    "context_window": int(model_answers['context_window']),
                    "max_tokens": int(model_answers['max_tokens'])
                })
                
                if not model_answers['add_another']:
                    break
            
            # Create the provider entry
            new_provider = {
                "name": provider_answers['name'],
                "base_url": provider_answers['base_url'].rstrip('/'),
                "api_key": provider_answers['api_key'],
                "header_auth": provider_answers['header_auth'],
                "use_chat_completions_endpoint": provider_answers['use_chat_completions_endpoint'],
                "models": models_list
            }
            
            # Add to custom providers list
            self.custom_providers.append(new_provider)
            self._save_custom_providers(self.custom_providers)
            
            self.console.print(f"[green]Added custom provider: {new_provider['name']}[/green]")
            
        except Exception as e:
            self.logger.error(f"Error adding custom provider: {e}", exc_info=True)
            self.console.print(f"[bold red]Error adding custom provider: {e}[/bold red]")

    def _edit_custom_provider(self):
        """Edit an existing custom API provider"""
        try:
            # Select provider to edit
            provider_choices = [(p['name'], i) for i, p in enumerate(self.custom_providers)]
            provider_choices.append(("Back", None))
            
            provider_question = [
                inquirer.List('provider',
                    message="Select provider to edit",
                    choices=provider_choices,
                    carousel=True
                )
            ]
            
            provider_answer = inquirer.prompt(provider_question)
            if not provider_answer or provider_answer['provider'] is None:
                return
            
            provider_index = provider_answer['provider']
            current_provider = self.custom_providers[provider_index]
            
            # Edit provider settings
            provider_questions = [
                inquirer.Text('name',
                    message="Provider name",
                    default=current_provider['name'],
                    validate=lambda _, x: bool(x.strip())
                ),
                inquirer.Text('base_url',
                    message="API base URL",
                    default=current_provider['base_url'],
                    validate=lambda _, x: bool(x.strip())
                ),
                inquirer.Text('api_key',
                    message="API key (leave empty if not required)",
                    default=current_provider['api_key']
                ),
                inquirer.Confirm('header_auth',
                    message="Use Authorization header for API key?",
                    default=current_provider.get('header_auth', True)
                ),
                inquirer.Confirm('use_chat_completions_endpoint',
                    message="Append '/chat/completions' to base URL?",
                    default=current_provider.get('use_chat_completions_endpoint', True)
                )
            ]
            
            provider_answers = inquirer.prompt(provider_questions)
            if not provider_answers:
                return
            
            # Update provider information
            self.custom_providers[provider_index].update({
                "name": provider_answers['name'],
                "base_url": provider_answers['base_url'].rstrip('/'),
                "api_key": provider_answers['api_key'],
                "header_auth": provider_answers['header_auth'],
                "use_chat_completions_endpoint": provider_answers['use_chat_completions_endpoint']
            })
            
            # Manage models
            model_action_question = [
                inquirer.List('action',
                    message="Manage models for this provider",
                    choices=[
                        ("Keep existing models", "keep"),
                        ("Edit existing models", "edit"),
                        ("Replace all models", "replace")
                    ],
                    carousel=True
                )
            ]
            
            model_action_answer = inquirer.prompt(model_action_question)
            if not model_action_answer:
                return
            
            if model_action_answer['action'] == "edit":
                self._edit_provider_models(provider_index)
            elif model_action_answer['action'] == "replace":
                # Clear models and add new ones
                self.custom_providers[provider_index]['models'] = []
                while True:
                    self.console.print("\n[cyan]Add models for this provider[/cyan]")
                    model_questions = [
                        inquirer.Text('id',
                            message="Model ID (as expected by the API)",
                            validate=lambda _, x: bool(x.strip())
                        ),
                        inquirer.Text('name',
                            message="Display name",
                            validate=lambda _, x: bool(x.strip())
                        ),
                        inquirer.Text('description',
                            message="Description",
                            default="Custom model"
                        ),
                        inquirer.Text('context_window',
                            message="Context window size",
                            validate=lambda _, x: x.isdigit() and int(x) > 0,
                            default="4096"
                        ),
                        inquirer.Text('max_tokens',
                            message="Max output tokens",
                            validate=lambda _, x: x.isdigit() and int(x) > 0,
                            default="2048"
                        )
                    ]
                    
                    model_answers = inquirer.prompt(model_questions)
                    if not model_answers:
                        break
                    
                    self.custom_providers[provider_index]['models'].append({
                        "id": model_answers['id'],
                        "name": model_answers['name'],
                        "description": model_answers['description'],
                        "context_window": int(model_answers['context_window']),
                        "max_tokens": int(model_answers['max_tokens'])
                    })
                    
                    if not model_answers['add_another']:
                        break
            
            # Save changes
            self._save_custom_providers(self.custom_providers)
            self.console.print(f"[green]Updated custom provider: {provider_answers['name']}[/green]")
            
        except Exception as e:
            self.logger.error(f"Error editing custom provider: {e}", exc_info=True)
            self.console.print(f"[bold red]Error editing custom provider: {e}[/bold red]")

    def _edit_provider_models(self, provider_index):
        """Edit models for a specific provider"""
        provider = self.custom_providers[provider_index]
        models = provider['models']
        
        while True:
            # Display current models
            self.console.print(f"\n[bold cyan]Models for {provider['name']}:[/bold cyan]")
            for i, model in enumerate(models):
                self.console.print(f"[cyan]{i+1}.[/cyan] [bold]{model['name']}[/bold] ({model['id']})")
            
            model_choices = [
                ("Add new model", "add"),
                ("Edit model", "edit"),
                ("Remove model", "remove"),
                ("Done", "done")
            ]
            
            model_action = inquirer.prompt([
                inquirer.List('action',
                    message="Model management",
                    choices=model_choices,
                    carousel=True
                )
            ])
            
            if not model_action or model_action['action'] == "done":
                break
            
            if model_action['action'] == "add":
                # Add new model
                model_questions = [
                    inquirer.Text('id',
                        message="Model ID (as expected by the API)",
                        validate=lambda _, x: bool(x.strip())
                    ),
                    inquirer.Text('name',
                        message="Display name",
                        validate=lambda _, x: bool(x.strip())
                    ),
                    inquirer.Text('description',
                        message="Description",
                        default="Custom model"
                    ),
                    inquirer.Text('context_window',
                        message="Context window size",
                        validate=lambda _, x: x.isdigit() and int(x) > 0,
                        default="4096"
                    ),
                    inquirer.Text('max_tokens',
                        message="Max output tokens",
                        validate=lambda _, x: x.isdigit() and int(x) > 0,
                        default="2048"
                    )
                ]
                
                model_answers = inquirer.prompt(model_questions)
                if model_answers:
                    models.append({
                        "id": model_answers['id'],
                        "name": model_answers['name'],
                        "description": model_answers['description'],
                        "context_window": int(model_answers['context_window']),
                        "max_tokens": int(model_answers['max_tokens'])
                    })
            
            elif model_action['action'] == "edit" and models:
                # Select model to edit
                model_choices = [(f"{m['name']} ({m['id']})", i) for i, m in enumerate(models)]
                model_choices.append(("Back", None))
                
                model_select = inquirer.prompt([
                    inquirer.List('model',
                        message="Select model to edit",
                        choices=model_choices,
                        carousel=True
                    )
                ])
                
                if model_select and model_select['model'] is not None:
                    model_index = model_select['model']
                    current_model = models[model_index]
                    
                    # Edit model
                    model_edit = inquirer.prompt([
                        inquirer.Text('id',
                            message="Model ID",
                            default=current_model['id'],
                            validate=lambda _, x: bool(x.strip())
                        ),
                        inquirer.Text('name',
                            message="Display name",
                            default=current_model['name'],
                            validate=lambda _, x: bool(x.strip())
                        ),
                        inquirer.Text('description',
                            message="Description",
                            default=current_model['description']
                        ),
                        inquirer.Text('context_window',
                            message="Context window size",
                            default=str(current_model['context_window']),
                            validate=lambda _, x: x.isdigit() and int(x) > 0
                        ),
                        inquirer.Text('max_tokens',
                            message="Max output tokens",
                            default=str(current_model['max_tokens']),
                            validate=lambda _, x: x.isdigit() and int(x) > 0
                        )
                    ])
                    
                    if model_edit:
                        models[model_index] = {
                            "id": model_edit['id'],
                            "name": model_edit['name'],
                            "description": model_edit['description'],
                            "context_window": int(model_edit['context_window']),
                            "max_tokens": int(model_edit['max_tokens'])
                        }
            
            elif model_action['action'] == "remove" and models:
                # Select model to remove
                model_choices = [(f"{m['name']} ({m['id']})", i) for i, m in enumerate(models)]
                model_choices.append(("Back", None))
                
                model_select = inquirer.prompt([
                    inquirer.List('model',
                        message="Select model to remove",
                        choices=model_choices,
                        carousel=True
                    )
                ])
                
                if model_select and model_select['model'] is not None:
                    model_index = model_select['model']
                    confirm = inquirer.confirm(f"Remove model '{models[model_index]['name']}'?", default=False)
                    if confirm:
                        del models[model_index]
        
        # Save changes
        self.custom_providers[provider_index]['models'] = models
        self._save_custom_providers(self.custom_providers)

    def _remove_custom_provider(self):
        """Remove a custom API provider"""
        try:
            # Select provider to remove
            provider_choices = [(p['name'], i) for i, p in enumerate(self.custom_providers)]
            provider_choices.append(("Back", None))
            
            provider_question = [
                inquirer.List('provider',
                    message="Select provider to remove",
                    choices=provider_choices,
                    carousel=True
                )
            ]
            
            provider_answer = inquirer.prompt(provider_question)
            if not provider_answer or provider_answer['provider'] is None:
                return
            
            provider_index = provider_answer['provider']
            provider_name = self.custom_providers[provider_index]['name']
            
            # Confirm removal
            confirm = inquirer.confirm(f"Are you sure you want to remove '{provider_name}'?", default=False)
            if confirm:
                del self.custom_providers[provider_index]
                self._save_custom_providers(self.custom_providers)
                self.console.print(f"[green]Removed custom provider: {provider_name}[/green]")
            
        except Exception as e:
            self.logger.error(f"Error removing custom provider: {e}", exc_info=True)
            self.console.print(f"[bold red]Error removing custom provider: {e}[/bold red]")

    def display_main_menu(self):
        """Display the main menu for model selection"""
        self.logger.info("Displaying main menu")
        
        while True:
            try:
                # Reload favorites
                self.reload_favorites()
                
                # Check API availability first
                openai_available = bool(os.getenv('OPENAI_API_KEY'))
                openrouter_available = bool(os.getenv('OPENROUTER_API_KEY'))
                anthropic_available = bool(os.getenv('ANTHROPIC_API_KEY'))
                
                # Check if RAG is enabled and active
                settings = self._load_settings()
                agent_enabled = settings.get('agent', {}).get('enabled', False)
                agent_active = (
                    agent_enabled and 
                    self.chroma_manager and 
                    self.chroma_manager.vectorstore is not None and
                    self.chroma_manager.store_name is not None
                )
                
                # Get RAG status for main menu
                agent_info = f" 〈{self._get_agent_status_display()}〉"
                
                # Check for other enabled AI features
                streaming_enabled = settings.get('streaming', {}).get('enabled', False)
                tools_enabled = settings.get('tools', {}).get('enabled', False)
                
                # Build AI features display string for main menu
                ai_features_info = self._get_ai_features_display()
                
                # Update main menu choices with RAG status
                main_choices = [
                    ("═══ Select Your AI Provider ═══", None),
                ]
                
                # Add Recent Model option if a recent model exists
                recent_model = next((m for m in self.models_config if m.get('recent', False)), None)
                if recent_model:
                    recent_provider = recent_model.get('provider', '').upper()
                    main_choices.append((f"🕒 Recent Model      〈{recent_model['name']} ({recent_provider})〉", "recent"))
                
                # Add Favorites option
                main_choices.append(("★ Favorite Models   〈Your preferred AI companions〉", "favorites"))
                
                # OpenAI status
                if openai_available:
                    if agent_enabled:
                        status = "🟢"  # Green for normal operation with RAG
                        status_info = " 〈RAG: Enabled〉"
                    else:
                        status = "🟢"  # Green for normal operation
                        status_info = ""
                    main_choices.append((f"{status} OpenAI Models    〈GPT-4, GPT-3.5 & More〉{status_info}", "openai"))
                else:
                    main_choices.append(("○ OpenAI Models    〈API Key Required〉", None))
                
                # OpenRouter status - Always green when available
                if openrouter_available:
                    if agent_enabled:
                        status_info = " 〈RAG: Enabled〉"
                    else:
                        status_info = ""
                    main_choices.append((f"🟢 OpenRouter Models 〈Multiple Providers〉{status_info}", "openrouter"))
                else:
                    main_choices.append(("○ OpenRouter Models 〈API Key Required〉", None))
                
                # Anthropic status - Now supports RAG mode
                if anthropic_available:
                    if agent_enabled:
                        status = "🟢"  # Green for normal operation with RAG
                        status_info = " 〈RAG: Enabled〉"
                    else:
                        status = "🟢"  # Green for normal operation
                        status_info = ""
                    main_choices.append((f"{status} Anthropic Models 〈Claude & More〉{status_info}", "anthropic"))
                else:
                    main_choices.append(("○ Anthropic Models 〈API Key Required〉", None))
                
                # Add Custom Providers section
                if self.custom_providers:
                    custom_count = len(self.custom_providers)
                    model_count = sum(len(p.get('models', [])) for p in self.custom_providers)
                    main_choices.append((f"🟢 Custom Providers  〈{custom_count} Providers, {model_count} Models〉", "custom"))
                
                # Add Custom Provider Management option
                main_choices.append(("⚙️ Manage Custom APIs 〈Add/Edit Custom Providers〉", "manage_custom"))
                
                main_choices.extend([
                    ("═══ System Settings ═══", None),
                    (f"⚙️ AI Settings       〈Configure AI Behavior〉{ai_features_info}", "ai_settings"),
                    ("⚙️ Application Settings 〈Configure App Behavior〉", "settings"),
                    ("═══ Application ═══", None),
                    ("✖ Exit Application    〈Close ACT〉", "exit")
                ])
                
                # Only show logo on first display
                if not hasattr(self, '_menu_displayed'):
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
                    self._menu_displayed = True

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
                
                questions = [
                    inquirer.List('provider',
                        message="Select an option:",
                        choices=main_choices,
                        carousel=True
                    ),
                ]
                
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

                if answer['provider'] == "ai_settings":
                    if self.manage_ai_settings():
                        continue  # Refresh menu if settings changed
                elif answer['provider'] == "settings":
                    self.manage_settings()
                elif answer['provider'] == "favorites":
                    self.manage_favorites()
                elif answer['provider'] == "manage_custom":
                    self.manage_custom_providers()
                elif answer['provider'] == "custom":
                    self._select_custom_provider_model()
                elif answer['provider'] == "recent":
                    # Find the recent model
                    recent_model = next((m for m in self.models_config if m.get('recent', False)), None)
                    if recent_model:
                        # Check if model is in favorites
                        is_favorite = any(f['id'] == recent_model['id'] for f in self.favorites)
                        star = "★ " if is_favorite else ""
                        self.console.print(f"\n[cyan]Recent Model:[/cyan] {star}{recent_model['name']}")
                        if self._handle_model_selection(recent_model):
                            continue  # Return to main menu
                    else:
                        self.console.print("[yellow]No recent model found[/yellow]")
                        continue
                elif answer['provider'] == "openai" and openai_available:
                    # Filter OpenAI models
                    openai_models = [m for m in self.models_config if m.get('provider', '').lower() == 'openai']
                    model_choices = []
                    for m in openai_models:
                        # Check if model is in favorites
                        is_favorite = any(f['id'] == m['id'] for f in self.favorites)
                        star = "★ " if is_favorite else ""
                        model_choices.append((f"{star}{m['name']} - {m.get('description', 'No description')}", m))
                    model_choices.append(("Back", None))
                    
                    model_question = [
                        inquirer.List('model',
                            message="Select OpenAI Model",
                            choices=model_choices,
                            carousel=True
                        ),
                    ]
                    
                    model_answer = inquirer.prompt(model_question)
                    if model_answer and model_answer['model']:
                        if self._handle_model_selection(model_answer['model']):
                            continue  # Return to main menu
                elif answer['provider'] == "openrouter" and openrouter_available:
                    selected_model = self.select_openrouter_model()
                    if selected_model:
                        if self._handle_model_selection(selected_model):
                            continue  # Return to main menu
                elif answer['provider'] == "anthropic" and anthropic_available:
                    # Filter Anthropic models
                    anthropic_models = [m for m in self.models_config if m.get('provider', '').lower() == 'anthropic']
                    model_choices = []
                    for m in anthropic_models:
                        # Check if model is in favorites
                        is_favorite = any(f['id'] == m['id'] for f in self.favorites)
                        star = "★ " if is_favorite else ""
                        model_choices.append((f"{star}{m['name']} - {m.get('description', 'No description')}", m))
                    model_choices.append(("Back", None))
                    
                    model_question = [
                        inquirer.List('model',
                            message="Select Anthropic Model",
                            choices=model_choices,
                            carousel=True
                        ),
                    ]
                    
                    model_answer = inquirer.prompt(model_question)
                    if model_answer and model_answer['model']:
                        if self._handle_model_selection(model_answer['model']):
                            continue  # Return to main menu
                # ... rest of the menu handling ...

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

    def _select_custom_provider_model(self):
        """Display menu for selecting models from custom providers"""
        if not self.custom_providers:
            self.console.print("[yellow]No custom providers configured[/yellow]")
            return False

        # Group by provider
        provider_choices = [(p['name'], i) for i, p in enumerate(self.custom_providers)]
        provider_choices.append(("Back", None))
        
        provider_question = [
            inquirer.List('provider',
                message="Select Custom Provider",
                choices=provider_choices,
                carousel=True
            )
        ]
        
        provider_answer = inquirer.prompt(provider_question)
        if not provider_answer or provider_answer['provider'] is None:
            return False
        
        provider_index = provider_answer['provider']
        provider = self.custom_providers[provider_index]
        
        # Show models for selected provider
        models = provider.get('models', [])
        if not models:
            self.console.print(f"[yellow]No models configured for {provider['name']}[/yellow]")
            return False
        
        model_choices = []
        for m in models:
            # Check if model is in favorites
            full_id = f"custom/{provider['name']}/{m['id']}"
            is_favorite = any(f['id'] == full_id for f in self.favorites)
            star = "★ " if is_favorite else ""
            model_choices.append((
                f"{star}{m['name']} - {m.get('description', 'No description')}",
                m
            ))
        model_choices.append(("Back", None))
        
        model_question = [
            inquirer.List('model',
                message=f"Select {provider['name']} Model",
                choices=model_choices,
                carousel=True
            )
        ]
        
        model_answer = inquirer.prompt(model_question)
        if model_answer and model_answer['model']:
            # Create a complete model config with provider details
            model_config = model_answer['model'].copy()
            model_config['provider'] = 'custom'
            model_config['custom_provider'] = provider
            model_config['id'] = f"custom/{provider['name']}/{model_config['id']}"
            
            return self._handle_model_selection(model_config)
        
        return False
        
    def _handle_model_selection(self, selected_model):
        """Handle the model selection and subsequent actions"""
        # Check if model is in favorites
        is_favorite = any(f['id'] == selected_model['id'] for f in self.favorites)
        icon = "★ " if is_favorite else ""
        
        # Display model description in a panel with enhanced styling
        self.console.print()  # Add spacing
        self.console.print(Panel(
            f"[bold white]Provider:[/bold white] [cyan]{selected_model.get('provider', 'unknown').upper()}[/cyan]\n\n"
            f"[bold white]Description:[/bold white]\n[cyan]{selected_model.get('description', 'No description available')}[/cyan]",
            title=f"[bold cyan]{icon} {selected_model['name']}[/bold cyan]",
            title_align="left",
            border_style="bright_blue",
            padding=(1, 2),
            highlight=True
        ))
        self.console.print()  # Add spacing
        
        # Adjust action choices based on favorite status
        action_choices = [("Start Chat", "chat")]
        if is_favorite:
            action_choices.append(("Remove from Favorites", "unfavorite"))
        else:
            action_choices.append(("Add to Favorites", "favorite"))
        action_choices.append(("Back", "back"))

        action_question = [
            inquirer.List('action',
                message=f"Manage {selected_model['name']}",
                choices=action_choices,
                carousel=True
            ),
        ]

        action_answer = inquirer.prompt(action_question)
        if not action_answer:
            return False

        if action_answer['action'] == "chat":
            if self.start_chat(selected_model):
                return True  # Signal to return to main menu
        elif action_answer['action'] == "favorite":
            self.add_to_favorites(selected_model)
        elif action_answer['action'] == "unfavorite":
            self.remove_from_favorites(selected_model['id'])
            self.console.print(f"[yellow]Removed {selected_model['name']} from favorites[/yellow]")
        return False

    def start_chat(self, model_config):
        """Start chat with selected model"""
        try:
            system_instruction = self.instructions_manager.get_selected_instruction()
            
            # Record model and instruction usage
            self.stats_manager.record_model_usage(model_config)
            if system_instruction:
                self.stats_manager.record_instruction_usage(system_instruction['name'])
            
            # Enable file context for all providers when RAG is enabled
            settings = self._load_settings()
            agent_enabled = settings.get('agent', {}).get('enabled', False)
            enable_file_context = (
                agent_enabled and 
                self.chroma_manager and 
                self.chroma_manager.vectorstore is not None and
                self.chroma_manager.store_name is not None
            )
            
            # Save as recent model
            self._save_recent_model(model_config)
            
            chat = AIChat(
                model_config,
                self.logger,
                self.console,
                system_instruction,
                self.settings_manager,
                chroma_manager=self.chroma_manager if enable_file_context else None,
                stats_manager=self.stats_manager  # Pass stats_manager to AIChat
            )
            chat.chat_loop()
            return True
        except Exception as e:
            self.logger.error(f"Error starting chat: {e}", exc_info=True)
            self.console.print(f"[bold red]Error starting chat: {e}[/bold red]")
            return False

    def _save_recent_model(self, model_config):
        """Save the current model as the recent model in models.json"""
        try:
            # Make a copy of the model config
            recent_model = model_config.copy()
            
            # Add or set the recent flag
            recent_model['recent'] = True
            
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
                if model['id'] == recent_model['id']:
                    model['recent'] = True
                    model_exists = True
                    break
            
            # If the model is not in models.json (e.g., OpenRouter models), add it
            if not model_exists and recent_model.get('provider') == 'openrouter':
                # Format OpenRouter model for storage
                formatted_model = {
                    'id': recent_model['id'],
                    'name': recent_model['name'],
                    'description': recent_model.get('description', 'No description available'),
                    'context_window': recent_model.get('context_length', 4096),
                    'max_tokens': 4096,  # Default value if not specified
                    'provider': 'openrouter',
                    'recent': True
                }
                models_data['models'].append(formatted_model)
            
            # Save the updated models.json file
            with open(models_path, 'w') as f:
                json.dump(models_data, f, indent=4)
            
            self.logger.info(f"Saved {recent_model['name']} as recent model")
            
            # Update the models_config in memory
            self.models_config = models_data['models']
            
        except Exception as e:
            self.logger.error(f"Error saving recent model: {e}", exc_info=True)
            self.console.print(f"[bold red]Error saving recent model: {e}[/bold red]")

    def _load_settings(self) -> Dict:
        """Load settings from file"""
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                    
                    # Ensure agent settings exist with defaults
                    if 'agent' not in settings:
                        settings['agent'] = {
                            'enabled': False
                        }
                    
                    # Ensure streaming settings exist with defaults
                    if 'streaming' not in settings:
                        settings['streaming'] = {
                            'enabled': False
                        }
                    
                    # Load tools configuration from tools_config.json
                    tools_config_file = os.path.join(os.path.dirname(__file__), 'tools', 'tools_config.json')
                    if os.path.exists(tools_config_file):
                        with open(tools_config_file, 'r', encoding='utf-8') as f:
                            tools_config = json.load(f)
                            if 'tools' not in settings:
                                settings['tools'] = {
                                    'enabled': False,
                                    'available_tools': tools_config['tools']
                                }
                            else:
                                # Update available tools while preserving enabled status
                                available_tools = settings['tools'].get('available_tools', {})
                                for tool_name, tool_info in tools_config['tools'].items():
                                    if tool_name not in available_tools:
                                        available_tools[tool_name] = tool_info
                                    else:
                                        # Update tool info but preserve enabled status
                                        enabled_status = available_tools[tool_name].get('enabled', True)
                                        available_tools[tool_name] = tool_info
                                        available_tools[tool_name]['enabled'] = enabled_status
                                
                                # Remove tools that no longer exist
                                for tool_name in list(available_tools.keys()):
                                    if tool_name not in tools_config['tools']:
                                        del available_tools[tool_name]
                                
                                settings['tools']['available_tools'] = available_tools
                    else:
                        # Default tools settings if tools_config.json doesn't exist
                        if 'tools' not in settings:
                            settings['tools'] = {
                                'enabled': False,
                                'available_tools': {}
                            }
                    
                    self._save_settings(settings)
                    return settings
            
            # Default settings if settings file doesn't exist
            return {
                'agent': {
                    'enabled': False
                },
                'streaming': {
                    'enabled': False
                },
                'tools': {
                    'enabled': False,
                    'available_tools': {}
                }
            }
        except Exception as e:
            self.logger.error(f"Error loading settings: {e}")
            return {
                'agent': {'enabled': False},
                'streaming': {'enabled': False},
                'tools': {'enabled': False}
            }

    def _save_settings(self, settings: Dict) -> None:
        """Save settings to file"""
        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=4)
        except Exception as e:
            self.logger.error(f"Error saving settings: {e}")

    def _get_agent_status_display(self):
        """Get formatted RAG status for display"""
        settings = self._load_settings()
        agent_enabled = settings.get('agent', {}).get('enabled', False)
        agent_active = (
            agent_enabled and 
            self.chroma_manager and 
            self.chroma_manager.vectorstore is not None and
            self.chroma_manager.store_name is not None
        )

        if agent_active:
            return f"🟢 Active - Store: {self.chroma_manager.store_name}"
        elif agent_enabled:
            return "🟡 Enabled (No Store Selected)"
        else:
            return "⭕ Disabled"

    def manage_statistics(self):
        """Display ACT statistics"""
        while True:
            # Get formatted statistics table
            stats_table = self.stats_manager.format_stats_display()
            
            # Display the statistics
            self.console.print()  # Add some spacing
            self.console.print(stats_table)
            self.console.print()  # Add some spacing

            # Navigation options
            choices = [
                ("Back", "back")
            ]

            questions = [
                inquirer.List('action',
                    message="Select action",
                    choices=choices,
                    carousel=True
                ),
            ]

            answer = inquirer.prompt(questions)
            if not answer or answer['action'] == "back":
                break

    def _get_ai_features_display(self):
        """Get formatted AI features display string"""
        settings = self._load_settings()
        agent_enabled = settings.get('agent', {}).get('enabled', False)
        streaming_enabled = settings.get('streaming', {}).get('enabled', False)
        tools_enabled = settings.get('tools', {}).get('enabled', False)
        
        # Check if RAG is active (enabled + store selected)
        agent_active = (
            agent_enabled and 
            self.chroma_manager and 
            self.chroma_manager.vectorstore is not None and
            self.chroma_manager.store_name is not None
        )
        
        ai_features = []
        if agent_enabled:
            if agent_active:
                ai_features.append("🟢 RAG (Active)")
            else:
                ai_features.append("🟡 RAG (Enabled)")
        if streaming_enabled:
            ai_features.append("🟢 Streaming")
        if tools_enabled:
            ai_features.append("🟢 Tools")
        
        if not ai_features:
            return " 〈No features enabled〉"
        
        return f" 〈{', '.join(ai_features)}〉"

def main():
    """Main application entry point"""
    logger, console = setup_logging()
    
    try:
        env_path = os.path.join(os.path.dirname(__file__), '.env')
        if not os.path.exists(env_path):
            logger.error("Missing .env file")
            console.print("[bold red]Create a .env file with API keys[/bold red]")
            sys.exit(1)
        
        app = AIChatApp(logger, console)
        app.display_main_menu()
    
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}", exc_info=True)
        console.print(f"[bold red]Critical error: {e}[/bold red]")
        sys.exit(1)

if __name__ == "__main__":
    main()