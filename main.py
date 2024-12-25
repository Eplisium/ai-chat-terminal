from imports import *
from utils import setup_logging
from managers import SettingsManager, SystemInstructionsManager
from chat import AIChat, OpenRouterAPI, get_openrouter_headers

class AIChatApp:
    def __init__(self, logger, console):
        """Initialize AI Chat Application"""
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

    def manage_favorites(self):
        """Display favorites management menu"""
        if not self.favorites:
            self.console.print("[yellow]No favorite models yet[/yellow]")
            return

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

    def manage_ai_settings(self):
        """Display AI settings management menu"""
        while True:
            choices = [
                ("=== AI Settings ===", None),
                ("🤖 System Instructions", "instructions"),
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
        """Display the main menu for model selection"""
        self.logger.info("Displaying main menu")
        
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
                openai_available = bool(os.getenv('OPENAI_API_KEY'))
                openrouter_available = bool(os.getenv('OPENROUTER_API_KEY'))
                anthropic_available = bool(os.getenv('ANTHROPIC_API_KEY'))
                
                main_choices = [
                    ("═══ Select Your AI Provider ═══", None),
                    ("★ Favorite Models   〈Your preferred AI companions〉", "favorites"),
                ]
                
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
                
                main_choices.extend([
                    ("═══ System Settings ═══", None),
                    ("⚙️ AI Settings       〈Configure AI Behavior〉", "ai_settings"),
                    ("⚙️ Application Settings 〈Configure App Behavior〉", "settings"),
                    ("═══ Application ═══", None),
                    ("✖ Exit Application    〈Close ACT〉", "exit")
                ])

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

                if selected_provider == "openai":
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
                    selected_model = self.select_openrouter_model()
                    if selected_model:
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
                self.start_chat(selected_model)
            elif action_answer['action'] == "favorite":
                self.add_to_favorites(selected_model)

    def start_chat(self, model_config):
        """Start chat with selected model"""
        try:
            system_instruction = self.instructions_manager.get_selected_instruction()
            chat = AIChat(model_config, self.logger, self.console, system_instruction, self.settings_manager)
            chat.chat_loop()
        except Exception as e:
            self.logger.error(f"Error starting chat: {e}", exc_info=True)
            self.console.print(f"[bold red]Error starting chat: {e}[/bold red]")

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