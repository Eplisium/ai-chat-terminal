from imports import *

class ModelsManager:
    """Manager for fetching and updating AI models from provider APIs"""
    
    # Model ID prefixes to filter for chat-capable models
    OPENAI_CHAT_PREFIXES = ('gpt-', 'chatgpt-', 'o1-', 'o3-', 'o4-')
    
    def __init__(self, logger, console):
        self.logger = logger
        self.console = console
        self.models_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models.json')
        self.settings_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'settings.json')
        
        # Load API keys
        dotenv.load_dotenv()
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
    
    def _load_models(self) -> dict:
        """Load current models from models.json"""
        try:
            if os.path.exists(self.models_file):
                with open(self.models_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {"models": []}
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            return {"models": []}
    
    def _save_models(self, models_data: dict):
        """Save models to models.json"""
        try:
            with open(self.models_file, 'w', encoding='utf-8') as f:
                json.dump(models_data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Error saving models: {e}")
            raise
    
    def _load_settings(self) -> dict:
        """Load settings from settings.json"""
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            self.logger.error(f"Error loading settings: {e}")
            return {}
    
    def _save_settings(self, settings: dict):
        """Save settings to settings.json"""
        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=4, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Error saving settings: {e}")
    
    def fetch_openai_models(self) -> tuple[list, str | None]:
        """
        Fetch available models from OpenAI API
        Returns: (list of model dicts, error message or None)
        """
        if not self.openai_api_key:
            return [], "OpenAI API key not found"
        
        try:
            response = requests.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {self.openai_api_key}"},
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            raw_models = []
            
            self.logger.debug(f"OpenAI API returned {len(data.get('data', []))} total models")
            
            for model in data.get('data', []):
                model_id = model.get('id', '')
                
                # Filter for chat-capable models
                if not any(model_id.startswith(prefix) for prefix in self.OPENAI_CHAT_PREFIXES):
                    continue
                
                # Skip non-chat models
                skip_patterns = [
                    'embedding', 'audio', 'realtime', 'search', 'tts', 
                    'whisper', 'dall-e', 'moderation', 'instruct', 'image'
                ]
                if any(x in model_id.lower() for x in skip_patterns):
                    continue
                
                raw_models.append({
                    'id': model_id,
                    'name': self._format_model_name(model_id, 'openai'),
                    'provider': 'openai',
                    'created': model.get('created')
                })
            
            # Deduplicate: prefer non-dated versions, keep only the most recent dated version per family
            models = self._deduplicate_openai_models(raw_models)
            
            self.logger.info(f"Fetched {len(models)} OpenAI chat models (from {len(raw_models)} raw)")
            return models, None
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Failed to fetch OpenAI models: {str(e)}"
            self.logger.error(error_msg)
            return [], error_msg
        except Exception as e:
            error_msg = f"Error processing OpenAI models: {str(e)}"
            self.logger.error(error_msg)
            return [], error_msg
    
    def _deduplicate_openai_models(self, models: list) -> list:
        """
        Deduplicate OpenAI models - keep only the most useful versions.
        For each model family, prefer the base model or the latest dated version.
        """
        import re
        
        # Group models by their base family
        families = {}
        
        for model in models:
            model_id = model['id']
            
            # Extract base model family (remove date suffixes like -2024-11-20, -0125, etc.)
            # Pattern: base-name-YYYYMMDD or base-name-MMDD
            base_id = re.sub(r'-\d{4}-\d{2}-\d{2}$', '', model_id)  # Remove YYYY-MM-DD
            base_id = re.sub(r'-\d{4}$', '', base_id)  # Remove MMDD (like -0125)
            base_id = re.sub(r'-\d{6,8}$', '', base_id)  # Remove YYYYMMDD
            
            if base_id not in families:
                families[base_id] = []
            families[base_id].append(model)
        
        # For each family, pick the best model
        result = []
        for base_id, family_models in families.items():
            if len(family_models) == 1:
                result.append(family_models[0])
            else:
                # Prefer the base model (shortest ID that matches base_id exactly)
                base_model = None
                dated_models = []
                
                for m in family_models:
                    if m['id'] == base_id:
                        base_model = m
                    else:
                        dated_models.append(m)
                
                if base_model:
                    # Use the base model
                    result.append(base_model)
                elif dated_models:
                    # No base model, use the most recently created one
                    dated_models.sort(key=lambda x: x.get('created', 0), reverse=True)
                    result.append(dated_models[0])
        
        return result
    
    def fetch_anthropic_models(self) -> tuple[list, str | None]:
        """
        Fetch available models from Anthropic API with pagination support
        Returns: (list of model dicts, error message or None)
        """
        if not self.anthropic_api_key:
            return [], "Anthropic API key not found"
        
        try:
            models = []
            after_id = None
            
            while True:
                # Build request params
                params = {"limit": 100}
                if after_id:
                    params["after_id"] = after_id
                
                response = requests.get(
                    "https://api.anthropic.com/v1/models",
                    headers={
                        "x-api-key": self.anthropic_api_key,
                        "anthropic-version": "2023-06-01"
                    },
                    params=params,
                    timeout=30
                )
                response.raise_for_status()
                
                data = response.json()
                
                self.logger.debug(f"Anthropic API page returned {len(data.get('data', []))} models, has_more={data.get('has_more', False)}")
                
                for model in data.get('data', []):
                    model_id = model.get('id', '')
                    display_name = model.get('display_name', '')
                    
                    models.append({
                        'id': model_id,
                        'name': display_name or self._format_model_name(model_id, 'anthropic'),
                        'provider': 'anthropic',
                        'created_at': model.get('created_at')
                    })
                
                # Check for more pages
                if data.get('has_more', False) and data.get('last_id'):
                    after_id = data.get('last_id')
                else:
                    break
            
            self.logger.info(f"Fetched {len(models)} Anthropic models")
            return models, None
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Failed to fetch Anthropic models: {str(e)}"
            self.logger.error(error_msg)
            return [], error_msg
        except Exception as e:
            error_msg = f"Error processing Anthropic models: {str(e)}"
            self.logger.error(error_msg)
            return [], error_msg
    
    def _format_model_name(self, model_id: str, provider: str) -> str:
        """Format a model ID into a human-readable name"""
        name = model_id
        
        # Remove common suffixes and format
        if provider == 'openai':
            # gpt-4o-2024-11-20 -> GPT-4o
            parts = name.split('-')
            if parts[0].lower() == 'gpt':
                # Handle GPT models
                if len(parts) >= 2:
                    version = parts[1]
                    variant = parts[2] if len(parts) > 2 and not parts[2].isdigit() else ''
                    if variant and not variant.startswith('202'):
                        name = f"GPT-{version} {variant.title()}"
                    else:
                        name = f"GPT-{version}"
            elif parts[0].lower() == 'chatgpt':
                name = f"ChatGPT-{'-'.join(parts[1:])}"
            elif parts[0].lower() in ('o1', 'o3', 'o4'):
                name = model_id.upper().replace('-', ' ')
        
        elif provider == 'anthropic':
            # claude-3-5-sonnet-20241022 -> Claude 3.5 Sonnet
            if 'claude' in name.lower():
                parts = name.split('-')
                # Find version numbers and variant
                version_parts = []
                variant = None
                for i, part in enumerate(parts):
                    if part == 'claude':
                        continue
                    elif part.isdigit() or (len(part) == 1 and part.isdigit()):
                        version_parts.append(part)
                    elif part.startswith('202'):  # Date suffix
                        break
                    elif part not in ['latest']:
                        variant = part.title()
                
                version = '.'.join(version_parts) if version_parts else ''
                if variant:
                    name = f"Claude {version} {variant}" if version else f"Claude {variant}"
                else:
                    name = f"Claude {version}" if version else "Claude"
        
        return name.strip()
    
    def _merge_models(self, existing_models: list, new_models: list, provider: str) -> tuple[list, list, list]:
        """
        Merge new models with existing ones
        Returns: (merged_models, added_models, updated_models)
        """
        import re
        
        # Create lookup by ID for existing models
        existing_by_id = {m['id']: m for m in existing_models}
        
        # Also create a lookup by base model family to avoid adding variants of existing models
        def get_base_family(model_id: str) -> str:
            """Extract the base model family from an ID"""
            base_id = re.sub(r'-\d{4}-\d{2}-\d{2}$', '', model_id)  # Remove YYYY-MM-DD
            base_id = re.sub(r'-\d{4}$', '', base_id)  # Remove MMDD
            base_id = re.sub(r'-\d{6,8}$', '', base_id)  # Remove YYYYMMDD
            return base_id.lower()
        
        existing_families = set()
        for m in existing_models:
            if m.get('provider') == provider:
                existing_families.add(get_base_family(m['id']))
        
        added = []
        updated = []
        
        for new_model in new_models:
            model_id = new_model['id']
            model_family = get_base_family(model_id)
            
            # Check if exact ID exists
            if model_id in existing_by_id:
                # Model exists - check if we need to update anything
                existing = existing_by_id[model_id]
                
                # Preserve user customizations (recent, favorite flags, description, context_window, max_tokens)
                # Only update if the new model has additional info
                changes_made = False
                
                # Don't overwrite existing name if it's been customized
                if existing.get('name') == model_id and new_model.get('name') != model_id:
                    existing['name'] = new_model['name']
                    changes_made = True
                
                if changes_made:
                    updated.append(existing)
            # Check if a model from the same family already exists
            elif model_family in existing_families:
                # Skip - we already have a model from this family
                self.logger.debug(f"Skipping {model_id} - family {model_family} already exists")
                continue
            else:
                # Truly new model - add it
                added.append(new_model.copy())
                existing_families.add(model_family)
        
        # Build final list - preserve order, add new models at provider's section end
        merged = list(existing_models)  # Start with existing
        
        # Find the position to insert (after last model of this provider)
        if added:
            last_provider_idx = -1
            for i, model in enumerate(merged):
                if model.get('provider') == provider:
                    last_provider_idx = i
            
            if last_provider_idx >= 0:
                # Insert after the last model of this provider
                for i, new_model in enumerate(added):
                    merged.insert(last_provider_idx + 1 + i, new_model)
            else:
                # No existing models for this provider, append at end
                merged.extend(added)
        
        return merged, added, updated
    
    def check_for_updates(self, show_progress: bool = True) -> dict:
        """
        Check for model updates from both OpenAI and Anthropic
        Returns: Summary dict with added, updated, and error counts
        """
        summary = {
            'openai_added': [],
            'openai_updated': [],
            'openai_error': None,
            'anthropic_added': [],
            'anthropic_updated': [],
            'anthropic_error': None,
            'total_added': 0,
            'total_updated': 0
        }
        
        # Load current models
        models_data = self._load_models()
        current_models = models_data.get('models', [])
        
        if show_progress:
            self.console.print("[dim]Checking for model updates...[/dim]")
        
        # Fetch OpenAI models
        if self.openai_api_key:
            if show_progress:
                self.console.print("[dim]  → Fetching OpenAI models...[/dim]")
            openai_models, openai_error = self.fetch_openai_models()
            
            if openai_error:
                summary['openai_error'] = openai_error
            elif openai_models:
                current_models, added, updated = self._merge_models(
                    current_models, openai_models, 'openai'
                )
                summary['openai_added'] = added
                summary['openai_updated'] = updated
        else:
            summary['openai_error'] = "API key not configured"
        
        # Fetch Anthropic models
        if self.anthropic_api_key:
            if show_progress:
                self.console.print("[dim]  → Fetching Anthropic models...[/dim]")
            anthropic_models, anthropic_error = self.fetch_anthropic_models()
            
            if anthropic_error:
                summary['anthropic_error'] = anthropic_error
            elif anthropic_models:
                current_models, added, updated = self._merge_models(
                    current_models, anthropic_models, 'anthropic'
                )
                summary['anthropic_added'] = added
                summary['anthropic_updated'] = updated
        else:
            summary['anthropic_error'] = "API key not configured"
        
        # Calculate totals
        summary['total_added'] = len(summary['openai_added']) + len(summary['anthropic_added'])
        summary['total_updated'] = len(summary['openai_updated']) + len(summary['anthropic_updated'])
        
        # Save if changes were made
        if summary['total_added'] > 0 or summary['total_updated'] > 0:
            models_data['models'] = current_models
            self._save_models(models_data)
            self.logger.info(f"Model update complete: {summary['total_added']} added, {summary['total_updated']} updated")
        
        # Update last check timestamp in settings
        settings = self._load_settings()
        if 'model_updates' not in settings:
            settings['model_updates'] = {}
        settings['model_updates']['last_check'] = datetime.now().isoformat()
        self._save_settings(settings)
        
        return summary
    
    def display_update_summary(self, summary: dict):
        """Display a Rich panel with update summary"""
        lines = []
        
        # OpenAI results
        if summary['openai_added']:
            lines.append(f"[green]✓ Added {len(summary['openai_added'])} new OpenAI model(s):[/green]")
            for model in summary['openai_added'][:5]:  # Show max 5
                lines.append(f"  [dim]•[/dim] {model.get('name', model['id'])}")
            if len(summary['openai_added']) > 5:
                lines.append(f"  [dim]... and {len(summary['openai_added']) - 5} more[/dim]")
        
        if summary['openai_updated']:
            lines.append(f"[blue]↻ Updated {len(summary['openai_updated'])} OpenAI model(s)[/blue]")
        
        if summary['openai_error']:
            lines.append(f"[yellow]⚠ OpenAI: {summary['openai_error']}[/yellow]")
        
        # Anthropic results
        if summary['anthropic_added']:
            lines.append(f"[green]✓ Added {len(summary['anthropic_added'])} new Anthropic model(s):[/green]")
            for model in summary['anthropic_added'][:5]:
                lines.append(f"  [dim]•[/dim] {model.get('name', model['id'])}")
            if len(summary['anthropic_added']) > 5:
                lines.append(f"  [dim]... and {len(summary['anthropic_added']) - 5} more[/dim]")
        
        if summary['anthropic_updated']:
            lines.append(f"[blue]↻ Updated {len(summary['anthropic_updated'])} Anthropic model(s)[/blue]")
        
        if summary['anthropic_error']:
            lines.append(f"[yellow]⚠ Anthropic: {summary['anthropic_error']}[/yellow]")
        
        # No changes
        if not lines:
            lines.append("[dim]No new models found[/dim]")
        elif summary['total_added'] == 0 and summary['total_updated'] == 0:
            if not summary['openai_error'] and not summary['anthropic_error']:
                lines.append("[dim]All models are up to date[/dim]")
        
        # Create and display panel
        content = "\n".join(lines)
        panel = Panel(
            content,
            title="[bold]Model Updates[/bold]",
            border_style="cyan",
            padding=(0, 1)
        )
        self.console.print(panel)
    
    def get_update_settings(self) -> dict:
        """Get current model update settings"""
        settings = self._load_settings()
        return settings.get('model_updates', {
            'check_on_startup': False,
            'last_check': None
        })
    
    def set_check_on_startup(self, enabled: bool):
        """Enable or disable checking for updates on startup"""
        settings = self._load_settings()
        if 'model_updates' not in settings:
            settings['model_updates'] = {}
        settings['model_updates']['check_on_startup'] = enabled
        self._save_settings(settings)
    
    def manage_model_updates(self):
        """Interactive menu to manage model update settings"""
        while True:
            update_settings = self.get_update_settings()
            check_on_startup = update_settings.get('check_on_startup', False)
            last_check = update_settings.get('last_check')
            
            # Format last check time
            if last_check:
                try:
                    last_dt = datetime.fromisoformat(last_check)
                    last_check_str = last_dt.strftime("%Y-%m-%d %H:%M")
                except:
                    last_check_str = "Unknown"
            else:
                last_check_str = "Never"
            
            choices = [
                ("═══ Model Updates ═══", None),
                (f"Check on Startup: {'✓ Enabled' if check_on_startup else '✗ Disabled'}", "toggle_startup"),
                (f"Last Check: {last_check_str}", None),
                ("─────────────────────", None),
                ("🔄 Check for New Models Now", "check_now"),
                ("🔍 Test API Connections", "test_apis"),
                ("═══ Navigation ═══", None),
                ("Back", "back")
            ]
            
            questions = [
                inquirer.List('action',
                    message="Model Update Settings",
                    choices=choices,
                    carousel=True
                ),
            ]
            
            answer = inquirer.prompt(questions)
            if not answer or answer['action'] == "back":
                break
            
            if answer['action'] == "toggle_startup":
                self.set_check_on_startup(not check_on_startup)
                status = "enabled" if not check_on_startup else "disabled"
                self.console.print(f"[green]Check on startup {status}[/green]")
            
            elif answer['action'] == "check_now":
                summary = self.check_for_updates(show_progress=True)
                self.display_update_summary(summary)
                self.console.input("\nPress Enter to continue...")
            
            elif answer['action'] == "test_apis":
                self._test_api_connections()
    
    def _test_api_connections(self):
        """Test API connections and show what models are being fetched"""
        self.console.print("\n[bold cyan]Testing API Connections...[/bold cyan]\n")
        
        # Test OpenAI
        self.console.print("[bold]OpenAI API:[/bold]")
        if self.openai_api_key:
            self.console.print(f"  API Key: [green]✓ Configured[/green] ({self.openai_api_key[:8]}...)")
            openai_models, openai_error = self.fetch_openai_models()
            if openai_error:
                self.console.print(f"  Status: [red]✗ Error: {openai_error}[/red]")
            else:
                self.console.print(f"  Status: [green]✓ Connected[/green]")
                self.console.print(f"  Models Found: [cyan]{len(openai_models)}[/cyan]")
                if openai_models:
                    self.console.print("  [dim]Sample models:[/dim]")
                    for model in openai_models[:5]:
                        self.console.print(f"    • {model['id']} → {model['name']}")
                    if len(openai_models) > 5:
                        self.console.print(f"    [dim]... and {len(openai_models) - 5} more[/dim]")
        else:
            self.console.print("  API Key: [yellow]✗ Not configured[/yellow]")
        
        self.console.print()
        
        # Test Anthropic
        self.console.print("[bold]Anthropic API:[/bold]")
        if self.anthropic_api_key:
            self.console.print(f"  API Key: [green]✓ Configured[/green] ({self.anthropic_api_key[:8]}...)")
            anthropic_models, anthropic_error = self.fetch_anthropic_models()
            if anthropic_error:
                self.console.print(f"  Status: [red]✗ Error: {anthropic_error}[/red]")
            else:
                self.console.print(f"  Status: [green]✓ Connected[/green]")
                self.console.print(f"  Models Found: [cyan]{len(anthropic_models)}[/cyan]")
                if anthropic_models:
                    self.console.print("  [dim]Sample models:[/dim]")
                    for model in anthropic_models[:5]:
                        self.console.print(f"    • {model['id']} → {model['name']}")
                    if len(anthropic_models) > 5:
                        self.console.print(f"    [dim]... and {len(anthropic_models) - 5} more[/dim]")
        else:
            self.console.print("  API Key: [yellow]✗ Not configured[/yellow]")
        
        self.console.print()
        self.console.input("Press Enter to continue...")

