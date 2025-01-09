from imports import *
from utils import setup_logging
from managers import (
    SettingsManager,
    SystemInstructionsManager,
    DataManager,
    ChromaManager,
    StatsManager
)
from chat import AIChat, OpenRouterAPI, get_openrouter_headers
from typing import Dict, Any

class AIChatApp:
    def __init__(self, logger, console):
        """Initialize AI Chat Application"""
        self.logger = logger
        self.console = console
        
        # Initialize file paths
        self.settings_file = os.path.join(os.path.dirname(__file__), 'settings.json')
        
        # Initialize managers
        self.instructions_manager = SystemInstructionsManager(logger, console)
        self.settings_manager = SettingsManager(logger, console)
        self.data_manager = DataManager(logger, console)
        self.chroma_manager = ChromaManager(logger, console)
        self.stats_manager = StatsManager(logger, console)
        
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
        model_id = f"{model_config.get('provider', 'unknown')}:{model_config['id']}"
        
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

    def sort_favorites(self, sort_key="name"):
        """Sort favorites list based on the given key"""
        if sort_key == "name":
            self.favorites.sort(key=lambda x: x['name'].lower())
        elif sort_key == "provider":
            self.favorites.sort(key=lambda x: (x['provider'].lower(), x['name'].lower()))
        self.save_favorites()

    def manage_favorites(self):
        """Display favorites management menu"""
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
            
            grouped_models = self.openrouter.group_models_by_company(models)
            if not grouped_models:
                self.console.print("[bold yellow]No models available[/bold yellow]")
                return None
            
            companies = list(grouped_models.keys())
            companies.sort()
            
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
                
                featured = "⭐ " if model.get('top_provider', False) else ""
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
                
                for instruction in instructions:
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

    def manage_ai_settings(self):
        """Display AI settings management menu"""
        while True:
            settings = self._load_settings()
            agent_enabled = settings.get('agent', {}).get('enabled', False)
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

            # Get current instruction name
            current_instruction = self.instructions_manager.get_current_name()

            choices = [
                ("═══ AI Settings ═══", None),
                (f"🤖 Agent           〈{agent_status}〉", "agent"),
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
            elif answer['setting'] == "instructions":
                self.manage_instructions()
            elif answer['setting'] == "contexts":
                self.manage_model_contexts()

    def manage_agent_settings(self, agent_status):
        """Manage Agent settings"""
        while True:
            settings = self._load_settings()
            agent_settings = settings.get('agent', {})
            agent_enabled = agent_settings.get('enabled', False)
            
            current_store = "None"
            if self.chroma_manager and self.chroma_manager.store_name:
                current_store = self.chroma_manager.store_name

            choices = [
                ("═══ Agent Settings ═══", None),
                (f"Current Status: {self._get_agent_status_display()}", None),
                (f"Current Store: {current_store}", None),
                ("Toggle Agent", "toggle"),
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
                        ("Test Search", "test"),
                        ("Refresh Store", "refresh"),
                        ("Select Embedding Model", "model"),
                    ])

            choices.extend([
                ("═══ Navigation ═══", None),
                ("Back", "back")
            ])

            questions = [
                inquirer.List('action',
                    message="Manage Agent Settings",
                    choices=choices,
                    carousel=True
                ),
            ]

            answer = inquirer.prompt(questions)
            if not answer or answer['action'] == "back":
                break

            if answer['action'] == "toggle":
                settings['agent']['enabled'] = not agent_enabled
                # Save last used store when disabling
                if agent_enabled and self.chroma_manager.store_name:
                    settings['agent']['last_store'] = self.chroma_manager.store_name
                self._save_settings(settings)
                new_status = "enabled" if not agent_enabled else "disabled"
                icon = "🟡" if not agent_enabled else "⭕"
                self.console.print(f"{icon} Agent {new_status}")
                continue  # Continue the loop instead of returning

            elif answer['action'] == "test_embeddings":
                if not self.chroma_manager:
                    self.console.print("[yellow]Please enable the agent first[/yellow]")
                    continue
                self.chroma_manager.test_embeddings()

            elif answer['action'] == "chromadb_settings":
                if not self.chroma_manager:
                    self.console.print("[yellow]Please enable the agent first[/yellow]")
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

            elif answer['action'] == "refresh":
                if not self.chroma_manager or not self.chroma_manager.store_name:
                    self.console.print("[yellow]Please select a store first[/yellow]")
                    continue

                refresh_choices = [
                    ("Refresh Using Last Directory", "last"),
                    ("Refresh Different Directory", "directory"),
                    ("Refresh Specific Files", "files"),
                    ("Back", None)
                ]

                refresh_question = [
                    inquirer.List('refresh_type',
                        message="Select Refresh Type",
                        choices=refresh_choices,
                        carousel=True
                    ),
                ]

                refresh_answer = inquirer.prompt(refresh_question)
                if refresh_answer and refresh_answer['refresh_type']:
                    if refresh_answer['refresh_type'] == "last":
                        self.chroma_manager.refresh_store()
                    elif refresh_answer['refresh_type'] == "directory":
                        dir_question = [
                            inquirer.Text('directory',
                                message="Enter directory path to refresh",
                                default="."
                            )
                        ]
                        dir_answer = inquirer.prompt(dir_question)
                        if dir_answer:
                            self.chroma_manager.refresh_store(directory_path=dir_answer['directory'])
                    else:  # files
                        files_question = [
                            inquirer.Text('files',
                                message="Enter file paths (comma-separated)",
                            )
                        ]
                        files_answer = inquirer.prompt(files_question)
                        if files_answer:
                            files = [f.strip() for f in files_answer['files'].split(',')]
                            self.chroma_manager.refresh_store(files=files)

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
                            dir_question = [
                                inquirer.Text('directory',
                                    message="Enter directory path to process",
                                    default="."
                                )
                            ]
                            dir_answer = inquirer.prompt(dir_question)
                            if dir_answer:
                                self.chroma_manager.process_directory(dir_answer['directory'], force_refresh=True)

            elif answer['action'] == "select":
                stores = self.chroma_manager.list_stores()
                if not stores:
                    self.console.print("[yellow]No stores available[/yellow]")
                    continue

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

            elif answer['action'] == "delete":
                stores = self.chroma_manager.list_stores()
                if not stores:
                    self.console.print("[yellow]No stores available[/yellow]")
                    continue

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
                self.chroma_manager.select_embedding_model()

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
                            dir_question = [
                                inquirer.Text('directory',
                                    message="Enter directory path to process",
                                    default="."
                                )
                            ]
                            dir_answer = inquirer.prompt(dir_question)
                            if dir_answer:
                                self.chroma_manager.process_directory(dir_answer['directory'], force_refresh=True)
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
                self.chroma_manager.select_embedding_model()

    def display_main_menu(self):
        """Display the main menu for model selection"""
        self.logger.info("Displaying main menu")
        
        while True:
            try:
                # Check API availability first
                openai_available = bool(os.getenv('OPENAI_API_KEY'))
                openrouter_available = bool(os.getenv('OPENROUTER_API_KEY'))
                anthropic_available = bool(os.getenv('ANTHROPIC_API_KEY'))
                
                # Check if Agent is enabled and active
                settings = self._load_settings()
                agent_enabled = settings.get('agent', {}).get('enabled', False)
                agent_active = (
                    agent_enabled and 
                    self.chroma_manager and 
                    self.chroma_manager.vectorstore is not None and
                    self.chroma_manager.store_name is not None
                )
                
                # Get agent status for main menu
                agent_info = f" 〈{self._get_agent_status_display()}〉"
                
                # Update main menu choices with agent status
                main_choices = [
                    ("═══ Select Your AI Provider ═══", None),
                    ("★ Favorite Models   〈Your preferred AI companions〉", "favorites"),
                ]
                
                # OpenAI status - Green when agent is enabled
                if openai_available:
                    if agent_enabled:
                        status = "🟢"  # Green for normal operation with agent
                        status_info = " 〈Agent: Enabled〉"
                    else:
                        status = "🟢"  # Green for normal operation
                        status_info = ""
                    main_choices.append((f"{status} OpenAI Models    〈GPT-4, GPT-3.5 & More〉{status_info}", "openai"))
                else:
                    main_choices.append(("○ OpenAI Models    〈API Key Required〉", None))
                
                # OpenRouter status - Always green when available
                if openrouter_available:
                    if agent_enabled:
                        status_info = " 〈Agent: Enabled〉"
                    else:
                        status_info = ""
                    main_choices.append((f"🟢 OpenRouter Models 〈Multiple Providers〉{status_info}", "openrouter"))
                else:
                    main_choices.append(("○ OpenRouter Models 〈API Key Required〉", None))
                
                # Anthropic status - Now supports agent mode
                if anthropic_available:
                    if agent_enabled:
                        status = "🟢"  # Green for normal operation with agent
                        status_info = " 〈Agent: Enabled〉"
                    else:
                        status = "🟢"  # Green for normal operation
                        status_info = ""
                    main_choices.append((f"{status} Anthropic Models 〈Claude & More〉{status_info}", "anthropic"))
                else:
                    main_choices.append(("○ Anthropic Models 〈API Key Required〉", None))
                
                main_choices.extend([
                    ("═══ System Settings ═══", None),
                    (f"⚙️ AI Settings       〈Configure AI Behavior〉{agent_info}", "ai_settings"),
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
                elif answer['provider'] == "openai" and openai_available:
                    # Filter OpenAI models
                    openai_models = [m for m in self.models_config if m.get('provider', '').lower() == 'openai']
                    model_choices = [(f"{m['name']} - {m.get('description', 'No description')}", m) for m in openai_models]
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
                    model_choices = [(f"{m['name']} - {m.get('description', 'No description')}", m) for m in anthropic_models]
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
    
    def _handle_model_selection(self, selected_model):
        """Handle the model selection and subsequent actions"""
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
                if self.start_chat(selected_model):
                    return True  # Signal to return to main menu
            elif action_answer['action'] == "favorite":
                self.add_to_favorites(selected_model)
        return False

    def start_chat(self, model_config):
        """Start chat with selected model"""
        try:
            system_instruction = self.instructions_manager.get_selected_instruction()
            
            # Record model and instruction usage
            self.stats_manager.record_model_usage(model_config)
            if system_instruction:
                self.stats_manager.record_instruction_usage(system_instruction['name'])
            
            # Enable file context for all providers when agent is enabled
            settings = self._load_settings()
            agent_enabled = settings.get('agent', {}).get('enabled', False)
            enable_file_context = (
                agent_enabled and
                self.chroma_manager and
                self.chroma_manager.vectorstore is not None
            )
            
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
                        self._save_settings(settings)
                    return settings
            return {
                'agent': {
                    'enabled': False
                }
            }
        except Exception as e:
            self.logger.error(f"Error loading settings: {e}")
            return {'agent': {'enabled': False}}

    def _save_settings(self, settings: Dict) -> None:
        """Save settings to file"""
        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=4)
        except Exception as e:
            self.logger.error(f"Error saving settings: {e}")

    def _get_agent_status_display(self):
        """Get formatted agent status for display"""
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