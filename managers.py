from imports import *

class SettingsManager:
    """Class to manage application settings"""
    def __init__(self, logger, console):
        self.logger = logger
        self.console = console
        self.settings_file = os.path.join(os.path.dirname(__file__), 'settings.json')
        self.settings = self._load_settings()

    def _load_settings(self):
        """Load settings from file"""
        default_settings = {
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
        """Get the currently selected instruction"""
        selected = self.instructions['selected']
        for instruction in self.instructions['instructions']:
            if instruction['name'] == selected:
                return {
                    'name': instruction['name'],
                    'content': instruction['content']
                }
        # Return default if nothing is selected
        return {
            'name': 'Default',
            'content': "You are a helpful AI assistant. Provide clear, concise, and helpful responses."
        }

    def list_instructions(self):
        """List all available instructions"""
        return self.instructions['instructions']

    def get_current_name(self):
        """Get the name of the currently selected instruction"""
        return self.instructions['selected'] 

class DataManager:
    """Class to manage application data operations"""
    def __init__(self, logger, console):
        self.logger = logger
        self.console = console
        self.base_dir = os.path.dirname(__file__)

    def clear_logs(self):
        """Clear all log files from the logs directory"""
        try:
            logs_dir = os.path.join(self.base_dir, 'logs')
            if os.path.exists(logs_dir):
                cleared = False
                for file in os.listdir(logs_dir):
                    if file.endswith('.log'):
                        file_path = os.path.join(logs_dir, file)
                        try:
                            os.remove(file_path)
                            cleared = True
                        except Exception as e:
                            self.logger.error(f"Error deleting log file {file}: {e}")
                
                if cleared:
                    self.console.print("[green]Successfully cleared all log files[/green]")
                else:
                    self.console.print("[yellow]No log files found to clear[/yellow]")
            else:
                self.console.print("[yellow]No logs directory found[/yellow]")
        except Exception as e:
            self.logger.error(f"Error clearing logs: {e}")
            self.console.print("[red]Error clearing logs[/red]")

    def clear_chats(self):
        """Clear all chat history files and subdirectories from the chats directory"""
        try:
            chats_dir = os.path.join(self.base_dir, 'chats')
            if os.path.exists(chats_dir):
                cleared = False
                # First, walk through all directories and files
                for root, dirs, files in os.walk(chats_dir, topdown=False):
                    # Delete all files in current directory
                    for file in files:
                        file_path = os.path.join(root, file)
                        try:
                            os.remove(file_path)
                            cleared = True
                        except Exception as e:
                            self.logger.error(f"Error deleting chat file {file}: {e}")
                    
                    # Delete all subdirectories except the main chats directory
                    if root != chats_dir:
                        try:
                            os.rmdir(root)
                            cleared = True
                        except Exception as e:
                            self.logger.error(f"Error deleting directory {root}: {e}")
                
                if cleared:
                    self.console.print("[green]Successfully cleared all chat history and subdirectories[/green]")
                else:
                    self.console.print("[yellow]No chat files or directories found to clear[/yellow]")
            else:
                self.console.print("[yellow]No chats directory found[/yellow]")
        except Exception as e:
            self.logger.error(f"Error clearing chats: {e}")
            self.console.print("[red]Error clearing chat history[/red]") 