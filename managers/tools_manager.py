import json
import os
import importlib.util
from typing import Dict, List, Optional, Any
import logging
import inspect

class ToolsManager:
    """Class to manage AI tools and their execution"""
    
    def __init__(self, logger=None, settings_manager=None):
        self.logger = logger or logging.getLogger(__name__)
        self.settings_manager = settings_manager
        self.tools_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tools')
        self.tools_config_file = os.path.join(self.tools_dir, 'tools_config.json')
        self.available_tools = {}
        self.loaded_tools = {}
        
        # Create tools directory if it doesn't exist
        os.makedirs(self.tools_dir, exist_ok=True)
        
        # Initialize or load tools configuration
        self._init_tools_config()
        
        # Load tool implementations
        self._load_tools()
    
    def _discover_tools(self) -> Dict[str, Dict]:
        """Discover tools from the tools directory"""
        discovered_tools = {}
        
        # Skip these files when discovering tools
        skip_files = ['__init__.py', 'tools_config.json']
        
        for filename in os.listdir(self.tools_dir):
            if filename in skip_files or not filename.endswith('_tool.py'):
                continue
                
            try:
                tool_name = filename[:-3]  # Remove .py extension
                if tool_name.endswith('_tool'):
                    tool_name = tool_name[:-5]  # Remove _tool suffix
                
                module_path = os.path.join(self.tools_dir, filename)
                spec = importlib.util.spec_from_file_location(tool_name, module_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                if hasattr(module, 'execute'):
                    # Get tool description from docstring
                    description = inspect.getdoc(module.execute) or f"Execute {tool_name} tool"
                    
                    # Get parameters from function signature
                    sig = inspect.signature(module.execute)
                    parameters = {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                    
                    # Only process the first parameter (arguments dict)
                    for param_name, param in list(sig.parameters.items())[:1]:
                        if param.annotation == Dict:
                            # Try to get parameter info from docstring
                            param_doc = inspect.getdoc(module.execute)
                            if param_doc:
                                # Parse docstring for parameter descriptions
                                lines = param_doc.split('\n')
                                current_param = None
                                for line in lines:
                                    if ':param' in line:
                                        param_parts = line.split(':param')[-1].split(':')
                                        if len(param_parts) >= 2:
                                            p_name = param_parts[0].strip()
                                            p_desc = param_parts[1].strip()
                                            parameters["properties"][p_name] = {
                                                "type": "string",
                                                "description": p_desc
                                            }
                                            parameters["required"].append(p_name)
                    
                    discovered_tools[tool_name] = {
                        "enabled": True,
                        "description": description,
                        "implementation": filename,
                        "parameters": parameters
                    }
                
            except Exception as e:
                self.logger.error(f"Error discovering tool {filename}: {e}")
                continue
        
        return discovered_tools
    
    def _init_tools_config(self):
        """Initialize or load tools configuration"""
        try:
            # Discover available tools
            discovered_tools = self._discover_tools()
            
            if os.path.exists(self.tools_config_file):
                # Load existing config
                with open(self.tools_config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    existing_tools = config.get('tools', {})
                
                # Update existing tools with any new ones
                for tool_name, tool_info in discovered_tools.items():
                    if tool_name not in existing_tools:
                        existing_tools[tool_name] = tool_info
                    else:
                        # Update implementation and parameters but keep enabled status
                        existing_tools[tool_name].update({
                            k: v for k, v in tool_info.items() 
                            if k not in ['enabled']
                        })
                
                # Remove tools that no longer exist
                for tool_name in list(existing_tools.keys()):
                    if tool_name not in discovered_tools:
                        del existing_tools[tool_name]
                
                self.available_tools = existing_tools
            else:
                self.available_tools = discovered_tools
            
            # Save updated configuration
            with open(self.tools_config_file, 'w', encoding='utf-8') as f:
                json.dump({"tools": self.available_tools}, f, indent=4)
            
        except Exception as e:
            self.logger.error(f"Error initializing tools configuration: {e}")
            self.available_tools = {}
    
    def _load_tools(self):
        """Load tool implementations from the tools directory"""
        self.loaded_tools = {}
        
        for tool_name, tool_info in self.available_tools.items():
            if not tool_info.get('enabled', True):
                continue
            
            implementation_file = tool_info.get('implementation')
            if not implementation_file:
                continue
            
            try:
                module_path = os.path.join(self.tools_dir, implementation_file)
                if not os.path.exists(module_path):
                    self.logger.warning(f"Tool implementation not found: {module_path}")
                    continue
                
                spec = importlib.util.spec_from_file_location(tool_name, module_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                if hasattr(module, 'execute'):
                    self.loaded_tools[tool_name] = module.execute
                else:
                    self.logger.warning(f"Tool {tool_name} missing execute function")
            except Exception as e:
                self.logger.error(f"Error loading tool {tool_name}: {e}")
    
    def get_enabled_tools(self) -> List[Dict[str, Any]]:
        """Get list of enabled tools in OpenAI function format"""
        tools = []
        for tool_name, tool_info in self.available_tools.items():
            if not tool_info.get('enabled', True):
                continue
            
            tools.append({
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": tool_info.get('description', ''),
                    "parameters": tool_info.get('parameters', {})
                }
            })
        return tools
    
    def execute_tool(self, tool_call: Dict) -> str:
        """Execute a tool call and return the result"""
        try:
            function_name = tool_call['function']['name']
            arguments = json.loads(tool_call['function']['arguments'])
            
            if function_name not in self.loaded_tools:
                return f"Tool not found or not enabled: {function_name}"
            
            return self.loaded_tools[function_name](arguments)
            
        except Exception as e:
            self.logger.error(f"Error executing tool {function_name}: {e}")
            return f"Error executing tool: {str(e)}"
    
    def register_tool(self, name: str, description: str, implementation: str, parameters: Dict, enabled: bool = True) -> bool:
        """Register a new tool"""
        try:
            # Validate implementation file exists
            implementation_path = os.path.join(self.tools_dir, implementation)
            if not os.path.exists(implementation_path):
                self.logger.error(f"Tool implementation file not found: {implementation}")
                return False
            
            # Add tool to configuration
            self.available_tools[name] = {
                "enabled": enabled,
                "description": description,
                "implementation": implementation,
                "parameters": parameters
            }
            
            # Save configuration
            with open(self.tools_config_file, 'w', encoding='utf-8') as f:
                json.dump({"tools": self.available_tools}, f, indent=4)
            
            # Reload tools
            self._load_tools()
            
            return True
        except Exception as e:
            self.logger.error(f"Error registering tool {name}: {e}")
            return False
    
    def unregister_tool(self, name: str) -> bool:
        """Unregister a tool"""
        try:
            if name in self.available_tools:
                del self.available_tools[name]
                
                # Save configuration
                with open(self.tools_config_file, 'w', encoding='utf-8') as f:
                    json.dump({"tools": self.available_tools}, f, indent=4)
                
                # Reload tools
                self._load_tools()
                
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error unregistering tool {name}: {e}")
            return False
    
    def enable_tool(self, name: str, enabled: bool = True) -> bool:
        """Enable or disable a tool"""
        try:
            if name in self.available_tools:
                self.available_tools[name]['enabled'] = enabled
                
                # Save configuration
                with open(self.tools_config_file, 'w', encoding='utf-8') as f:
                    json.dump({"tools": self.available_tools}, f, indent=4)
                
                # Reload tools if enabling
                if enabled:
                    self._load_tools()
                elif name in self.loaded_tools:
                    del self.loaded_tools[name]
                
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error changing tool state {name}: {e}")
            return False 