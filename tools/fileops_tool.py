import os
import json
from typing import Dict, List, Optional

def get_file_info(path: str) -> Dict:
    """Get information about a file"""
    try:
        stats = os.stat(path)
        return {
            'exists': True,
            'size': stats.st_size,
            'last_modified': stats.st_mtime,
            'is_file': os.path.isfile(path),
            'is_dir': os.path.isdir(path)
        }
    except:
        return {
            'exists': False,
            'error': f"Unable to access {path}"
        }

def execute(arguments: Dict) -> str:
    """Execute file operations tool"""
    operation = arguments.get('operation', '').lower()
    path = arguments.get('path', '')
    
    if not path:
        return "Error: No path provided"
    
    # Normalize path
    path = os.path.normpath(path)
    
    # Basic security check - don't allow access outside workspace
    workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    abs_path = os.path.abspath(path)
    if not abs_path.startswith(workspace_root):
        return "Error: Access denied - path outside workspace"
    
    try:
        if operation == 'info':
            info = get_file_info(path)
            if info['exists']:
                if info['is_file']:
                    size_kb = info['size'] / 1024
                    return f"File: {path}\nSize: {size_kb:.2f} KB\nLast Modified: {info['last_modified']}"
                else:
                    return f"Directory: {path}"
            else:
                return f"Path does not exist: {path}"
        
        elif operation == 'list':
            if not os.path.isdir(path):
                return f"Error: {path} is not a directory"
            
            files = []
            dirs = []
            try:
                for item in os.listdir(path):
                    full_path = os.path.join(path, item)
                    if os.path.isdir(full_path):
                        dirs.append(f"📁 {item}/")
                    else:
                        files.append(f"📄 {item}")
            except Exception as e:
                return f"Error listing directory: {str(e)}"
            
            # Sort and combine
            dirs.sort()
            files.sort()
            items = dirs + files
            
            if items:
                return f"Contents of {path}:\n" + "\n".join(items)
            else:
                return f"Directory {path} is empty"
        
        elif operation == 'exists':
            exists = os.path.exists(path)
            return f"Path {'exists' if exists else 'does not exist'}: {path}"
        
        else:
            return f"Unknown operation: {operation}"
            
    except Exception as e:
        return f"Error performing {operation} on {path}: {str(e)}" 