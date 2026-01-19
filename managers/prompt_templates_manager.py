from imports import *


class PromptTemplatesManager:
    """Class to manage favorite prompt templates"""
    def __init__(self, logger, console):
        self.logger = logger
        self.console = console
        self.templates_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'prompt_templates.json')
        self.templates = self._load_templates()

    def _load_templates(self):
        """Load prompt templates from file"""
        try:
            if os.path.exists(self.templates_file):
                with open(self.templates_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'templates' not in data:
                        data['templates'] = []
                    return data
            return {'templates': []}
        except Exception as e:
            self.logger.error(f"Error loading prompt templates: {e}")
            return {'templates': []}

    def _save_templates(self):
        """Save prompt templates to file"""
        try:
            with open(self.templates_file, 'w', encoding='utf-8') as f:
                json.dump(self.templates, f, indent=4, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Error saving prompt templates: {e}")

    def list_templates(self):
        """List all prompt templates"""
        return self.templates.get('templates', [])

    def get_template(self, name):
        """Get a prompt template by name"""
        for template in self.templates.get('templates', []):
            if template['name'] == name:
                return template
        return None

    def add_template(self, name, content):
        """Add a new prompt template"""
        if any(t['name'] == name for t in self.templates.get('templates', [])):
            return False, "Template with this name already exists"

        self.templates.setdefault('templates', []).append({
            'name': name,
            'content': content,
            'created_at': datetime.now().isoformat()
        })
        self._save_templates()
        return True, "Prompt template added successfully"

    def update_template(self, name, content):
        """Update an existing prompt template"""
        templates = self.templates.get('templates', [])
        for template in templates:
            if template['name'] == name:
                template['content'] = content
                template['updated_at'] = datetime.now().isoformat()
                self._save_templates()
                return True, "Prompt template updated successfully"
        return False, "Template not found"

    def remove_template(self, name):
        """Remove a prompt template"""
        templates = self.templates.get('templates', [])
        self.templates['templates'] = [t for t in templates if t['name'] != name]
        self._save_templates()
        return True, "Prompt template removed successfully"