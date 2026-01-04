from imports import *

class SettingsManager:
    """Class to manage application settings"""
    def __init__(self, logger, console):
        self.logger = logger
        self.console = console
        self.settings_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'settings.json')
        self.settings = self._load_settings()

    def _load_settings(self):
        """Load settings from file"""
        default_settings = {
            'timezone': {
                'preferred_timezone': None,  # Will use system timezone if None
                'format': 'natural',  # 'natural' or 'technical'
            },
            'codebase_search': {
                'file_types': ['.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h', '.cs', '.php', '.rb', '.go', '.rs', '.swift', '.kt', '.scala', '.html', '.css', '.scss', '.sass', '.less', '.vue', '.sql', '.md', '.txt', '.json', '.yaml', '.yml', '.toml'],
                'enabled_types': ['.py'],  # Default to only Python files
                'search_subdirectories': True,
                'max_file_size_mb': 10,
                'exclude_patterns': ['node_modules', 'venv', '.git', '__pycache__', 'build', 'dist']
            },
            'appearance': {
                'ai_name_color': '#A6E22E',  # Default lime green
                'instruction_name_color': '#FFD700',  # Default gold
                'cost_color': '#00FFFF'  # Default cyan
            },
            'chromadb': {
                'default_store': None,
                'auto_add_files': True,
                'max_file_size_mb': 5,
                'exclude_patterns': ['node_modules', 'venv', '.git', '__pycache__', 'build', 'dist', 'chroma_stores'],
                'file_types': ['.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h', '.cs', '.php', '.rb', '.md', '.txt']
            },
            'model_updates': {
                'check_on_startup': False,
                'last_check': None
            }
        }
        
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    loaded_settings = json.load(f)
                    # Merge loaded settings with defaults to ensure all keys exist
                    for category, values in default_settings.items():
                        if category not in loaded_settings:
                            loaded_settings[category] = values
                        elif isinstance(values, dict):
                            for key, value in values.items():
                                if key not in loaded_settings[category]:
                                    loaded_settings[category][key] = value
                    return loaded_settings
            return default_settings
        except Exception as e:
            self.logger.error(f"Error loading settings: {e}")
            return default_settings

    def _save_settings(self):
        """Save settings to file"""
        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=4)
        except Exception as e:
            self.logger.error(f"Error saving settings: {e}")

    def get_setting(self, category, default=None):
        """Get a category of settings"""
        try:
            return self.settings.get(category, default if default is not None else {})
        except Exception as e:
            self.logger.error(f"Error getting setting {category}: {e}")
            return default if default is not None else {}

    def update_setting(self, category, key, value):
        """Update a specific setting value"""
        try:
            if category not in self.settings:
                self.settings[category] = {}
            self.settings[category][key] = value
            self._save_settings()
            return True
        except Exception as e:
            self.logger.error(f"Error updating setting {category}.{key}: {e}")
            return False

    def manage_appearance_settings(self):
        """Manage appearance settings"""
        while True:
            try:
                appearance_settings = self.get_setting('appearance')
                
                # Ensure default values exist
                if not appearance_settings:
                    appearance_settings = {
                        'ai_name_color': '#A6E22E',
                        'instruction_name_color': '#FFD700',
                        'cost_color': '#00FFFF'
                    }
                    self.settings['appearance'] = appearance_settings
                    self._save_settings()

                choices = [
                    ("=== Appearance Settings ===", None),
                    (f"AI Name Color: {appearance_settings.get('ai_name_color', '#A6E22E')}", "ai_color"),
                    (f"Instruction Name Color: {appearance_settings.get('instruction_name_color', '#FFD700')}", "instruction_color"),
                    (f"Cost Information Color: {appearance_settings.get('cost_color', '#00FFFF')}", "cost_color"),
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

                if answer['action'] in ["ai_color", "instruction_color", "cost_color"]:
                    color_choices = [
                        ("Lime Green (#A6E22E)", "#A6E22E"),
                        ("Gold (#FFD700)", "#FFD700"),
                        ("Cyan (#00FFFF)", "#00FFFF"),
                        ("Magenta (#FF00FF)", "#FF00FF"),
                        ("Orange (#FFA500)", "#FFA500"),
                        ("Purple (#800080)", "#800080"),
                        ("Red (#FF0000)", "#FF0000"),
                        ("Blue (#0000FF)", "#0000FF"),
                        ("Custom (Enter HEX)", "custom"),
                        ("Back", None)
                    ]

                    color_question = [
                        inquirer.List('color',
                            message="Select color",
                            choices=color_choices,
                            carousel=True
                        ),
                    ]

                    color_answer = inquirer.prompt(color_question)
                    if color_answer and color_answer['color']:
                        if color_answer['color'] == "custom":
                            custom_question = [
                                inquirer.Text('hex',
                                    message="Enter HEX color code (e.g., #FF0000)",
                                    validate=lambda _, x: x.startswith('#') and len(x) == 7 and all(c in '0123456789ABCDEFabcdef' for c in x[1:])
                                )
                            ]
                            custom_answer = inquirer.prompt(custom_question)
                            if custom_answer:
                                color = custom_answer['hex']
                            else:
                                continue
                        else:
                            color = color_answer['color']

                        # Preview the color
                        self.console.print(f"\nPreview: [bold {color}]Sample Text[/]")
                        
                        confirm = [
                            inquirer.Confirm('confirm',
                                message="Apply this color?",
                                default=True
                            ),
                        ]
                        
                        if inquirer.prompt(confirm)['confirm']:
                            setting_key = {
                                "ai_color": "ai_name_color",
                                "instruction_color": "instruction_name_color",
                                "cost_color": "cost_color"
                            }[answer['action']]
                            
                            if 'appearance' not in self.settings:
                                self.settings['appearance'] = {}
                            self.settings['appearance'][setting_key] = color
                            self._save_settings()
                            self.console.print(f"[green]Color updated successfully![/green]")
            except Exception as e:
                self.logger.error(f"Error in appearance settings: {e}")
                self.console.print(f"[bold red]Error: {e}[/bold red]")
                continue

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

    def manage_timezone_settings(self):
        """Manage timezone settings"""
        while True:
            try:
                timezone_settings = self.get_setting('timezone', {
                    'preferred_timezone': None,
                    'format': 'natural'
                })

                current_timezone = timezone_settings.get('preferred_timezone', 'System Default')
                current_format = timezone_settings.get('format', 'natural')

                choices = [
                    ("=== Timezone Settings ===", None),
                    (f"Preferred Timezone: {current_timezone}", "timezone"),
                    (f"Time Format: {'Natural' if current_format == 'natural' else 'Technical'}", "format"),
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

                if answer['action'] == "timezone":
                    # Get common timezones for easier selection
                    common_timezones = [
                        "System Default",
                        "America/New_York",
                        "America/Los_Angeles",
                        "America/Chicago",
                        "Europe/London",
                        "Europe/Paris",
                        "Asia/Tokyo",
                        "Asia/Shanghai",
                        "Australia/Sydney",
                        "Custom"
                    ]

                    tz_choices = [(tz, tz) for tz in common_timezones]
                    tz_question = [
                        inquirer.List('timezone',
                            message="Select timezone",
                            choices=tz_choices,
                            carousel=True
                        ),
                    ]

                    tz_answer = inquirer.prompt(tz_question)
                    if tz_answer:
                        selected_tz = tz_answer['timezone']
                        
                        if selected_tz == "Custom":
                            custom_question = [
                                inquirer.Text('custom_tz',
                                    message="Enter timezone (e.g., America/New_York)",
                                    validate=lambda _, x: x in pytz.all_timezones
                                )
                            ]
                            custom_answer = inquirer.prompt(custom_question)
                            if custom_answer:
                                selected_tz = custom_answer['custom_tz']
                            else:
                                continue
                        
                        if selected_tz == "System Default":
                            selected_tz = None
                        
                        self.update_setting('timezone', 'preferred_timezone', selected_tz)
                        self.console.print("[green]Timezone updated successfully![/green]")

                elif answer['action'] == "format":
                    format_choices = [
                        ("Natural (e.g., 'It's 12:52 AM in New York')", "natural"),
                        ("Technical (e.g., 'Current time: 12:52 AM (America/New_York)')", "technical")
                    ]

                    format_question = [
                        inquirer.List('format',
                            message="Select time format",
                            choices=format_choices,
                            carousel=True
                        ),
                    ]

                    format_answer = inquirer.prompt(format_question)
                    if format_answer:
                        self.update_setting('timezone', 'format', format_answer['format'])
                        self.console.print("[green]Time format updated successfully![/green]")

            except Exception as e:
                self.logger.error(f"Error in timezone settings: {e}")
                self.console.print(f"[bold red]Error: {e}[/bold red]")
                continue

    def manage_chromadb_settings(self):
        """Manage ChromaDB settings"""
        while True:
            settings = self._load_settings()
            chromadb_settings = settings.get('chromadb', {})
            
            choices = [
                ("═══ ChromaDB Settings ═══", None),
                (f"Embedding Model: {chromadb_settings.get('embedding_model', 'text-embedding-3-small')}", "embedding_model"),
                (f"Auto-Add Files: {chromadb_settings.get('auto_add_files', True)}", "auto_add"),
                (f"Max File Size: {chromadb_settings.get('max_file_size_mb', 5)}MB", "max_size"),
                ("Manage File Types", "file_types"),
                ("Manage Exclude Patterns", "exclude"),
                ("Test Embeddings", "test_embeddings"),
                ("═══ Navigation ═══", None),
                ("Back", "back")
            ]

            questions = [
                inquirer.List('action',
                    message="Manage ChromaDB Settings",
                    choices=choices,
                    carousel=True
                ),
            ]

            answer = inquirer.prompt(questions)
            if not answer or answer['action'] == "back":
                break

            elif answer['action'] == "test_embeddings":
                # Get ChromaManager class to test embeddings
                from .chroma_manager import ChromaManager
                chroma_manager = ChromaManager(self.logger, self.console)
                chroma_manager.test_embeddings()

            elif answer['action'] == "embedding_model":
                # Get ChromaManager class to access available models
                from .chroma_manager import ChromaManager
                
                current_model = chromadb_settings.get('embedding_model', 'text-embedding-3-small')
                choices = []
                
                for model_name, info in ChromaManager.EMBEDDING_MODELS.items():
                    is_current = "✓ " if model_name == current_model else "  "
                    choices.append((
                        f"{is_current}{model_name} - {info['description']} ({info['dimensions']} dimensions)",
                        model_name
                    ))
                choices.append(("Back", None))

                model_question = [
                    inquirer.List('model',
                        message="Select OpenAI Embedding Model",
                        choices=choices,
                        carousel=True
                    ),
                ]

                model_answer = inquirer.prompt(model_question)
                if model_answer and model_answer['model']:
                    self.update_setting('chromadb', 'embedding_model', model_answer['model'])
                    self.console.print(f"[green]Embedding model updated to: {model_answer['model']}[/green]")
                    self.console.print("[yellow]Note: You'll need to reload any open stores for the change to take effect[/yellow]") 