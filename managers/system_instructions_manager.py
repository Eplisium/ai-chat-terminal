from imports import *

class SystemInstructionsManager:
    """Class to manage system instructions for AI models"""
    def __init__(self, logger, console):
        self.logger = logger
        self.console = console
        self.instructions_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'system_instructions.json')
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