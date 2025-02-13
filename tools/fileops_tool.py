import os
import json
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import mimetypes
import pygments
from pygments import lexers, formatters, util

def format_size(size_bytes: int) -> str:
    """Convert bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"

def format_timestamp(timestamp: float) -> str:
    """Convert timestamp to human readable format"""
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

def get_file_info(path: str, detailed: bool = False) -> Dict:
    """Get detailed information about a file or directory
    
    Args:
        path: Path to file or directory
        detailed: Whether to include detailed information
    """
    try:
        stats = os.stat(path)
        info = {
            'exists': True,
            'name': os.path.basename(path),
            'path': path,
            'type': 'directory' if os.path.isdir(path) else 'file',
            'size': stats.st_size,
            'size_formatted': format_size(stats.st_size),
            'last_modified': format_timestamp(stats.st_mtime),
            'created': format_timestamp(stats.st_ctime)
        }
        
        if detailed:
            info.update({
                'is_symlink': os.path.islink(path),
                'parent': os.path.dirname(path),
                'permissions': oct(stats.st_mode)[-3:],
                'owner': stats.st_uid,
                'group': stats.st_gid,
                'last_accessed': format_timestamp(stats.st_atime)
            })
            
            if os.path.isfile(path):
                info.update({
                    'extension': os.path.splitext(path)[1],
                    'is_hidden': path.startswith('.'),
                    'is_readonly': not os.access(path, os.W_OK)
                })
            elif os.path.isdir(path):
                try:
                    contents = os.listdir(path)
                    info.update({
                        'files_count': len([x for x in contents if os.path.isfile(os.path.join(path, x))]),
                        'dirs_count': len([x for x in contents if os.path.isdir(os.path.join(path, x))]),
                        'is_empty': len(contents) == 0
                    })
                except:
                    pass
                    
        return info
    except Exception as e:
        return {
            'exists': False,
            'path': path,
            'error': str(e)
        }

def list_directory(path: str, recursive: bool = False, pattern: str = None) -> List[Dict]:
    """List contents of a directory with optional filtering
    
    Args:
        path: Directory path
        recursive: Whether to list subdirectories recursively
        pattern: Optional glob pattern to filter results
    """
    if not os.path.isdir(path):
        raise ValueError(f"Not a directory: {path}")
        
    results = []
    try:
        if recursive:
            for root, dirs, files in os.walk(path):
                for d in dirs:
                    full_path = os.path.join(root, d)
                    if pattern and not Path(full_path).match(pattern):
                        continue
                    results.append(get_file_info(full_path))
                for f in files:
                    full_path = os.path.join(root, f)
                    if pattern and not Path(full_path).match(pattern):
                        continue
                    results.append(get_file_info(full_path))
        else:
            for item in os.listdir(path):
                full_path = os.path.join(path, item)
                if pattern and not Path(full_path).match(pattern):
                    continue
                results.append(get_file_info(full_path))
                
        return sorted(results, key=lambda x: (x['type'] == 'file', x['name'].lower()))
    except Exception as e:
        raise RuntimeError(f"Error listing directory: {str(e)}")

def format_list_output(items: List[Dict]) -> str:
    """Format directory listing for display"""
    if not items:
        return "Directory is empty"
        
    output = []
    for item in items:
        icon = '📁' if item['type'] == 'directory' else '📄'
        name = f"{item['name']}/" if item['type'] == 'directory' else item['name']
        size = item['size_formatted']
        modified = item['last_modified']
        output.append(f"{icon} {name:<30} {size:>10}  {modified}")
        
    return "\n".join(output)

def ensure_parent_dir(path: str) -> None:
    """Ensure parent directory exists"""
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent)

def get_file_type(path: str) -> str:
    """Detect file type based on extension and content"""
    # Get mime type
    mime_type, _ = mimetypes.guess_type(path)
    
    if mime_type:
        if mime_type.startswith('text/'):
            return 'text'
        elif mime_type.startswith('image/'):
            return 'image'
        elif mime_type.startswith('video/'):
            return 'video'
        elif mime_type.startswith('audio/'):
            return 'audio'
    
    # Check extension for code files
    ext = os.path.splitext(path)[1].lower()
    code_extensions = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.html': 'html',
        '.css': 'css',
        '.json': 'json',
        '.md': 'markdown',
        '.xml': 'xml',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.sh': 'bash',
        '.bat': 'batch',
        '.ps1': 'powershell',
        '.sql': 'sql',
        '.cpp': 'cpp',
        '.c': 'c',
        '.h': 'c',
        '.cs': 'csharp',
        '.java': 'java',
        '.rb': 'ruby',
        '.php': 'php',
        '.go': 'go',
        '.rs': 'rust'
    }
    
    if ext in code_extensions:
        return f"code/{code_extensions[ext]}"
    
    return 'binary'

def format_code_with_highlighting(content: str, language: str) -> str:
    """Format code with syntax highlighting"""
    try:
        # Get lexer for language
        try:
            lexer = lexers.get_lexer_by_name(language)
        except util.ClassNotFound:
            lexer = lexers.guess_lexer(content)
            
        # Format with terminal colors
        formatter = formatters.TerminalFormatter()
        highlighted = pygments.highlight(content, lexer, formatter)
        
        return highlighted
    except:
        return content

def read_file_content(path: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> str:
    """Read file content with optional line range
    
    Args:
        path: Path to file
        start_line: Start line number (1-based, inclusive)
        end_line: End line number (1-based, inclusive)
    """
    try:
        # Check if file exists
        if not os.path.exists(path):
            return f"File not found: {path}"
        
        if not os.path.isfile(path):
            return f"Not a file: {path}"
            
        # Read file content
        with open(path, 'r', encoding='utf-8') as f:
            if start_line is None and end_line is None:
                content = f.read()
                return f"Contents of file {os.path.basename(path)}:\n```\n{content}\n```"
                
            lines = f.readlines()
            total_lines = len(lines)
            
            # Adjust line numbers
            if start_line is None:
                start_line = 1
            if end_line is None:
                end_line = total_lines
                
            # Convert to 0-based indexing
            start_idx = max(0, start_line - 1)
            end_idx = min(total_lines, end_line)
            
            # Get the requested lines
            content_lines = lines[start_idx:end_idx]
            content = ''.join(content_lines)
            
            # Format output with line numbers and context
            output = [f"Contents of file {os.path.basename(path)} (lines {start_line}-{end_line}):"]
            output.append("```")
            
            if start_line > 1:
                output.append(f"... {start_line-1} lines before ...")
            
            for i, line in enumerate(content_lines, start=start_line):
                output.append(f"{i:4d} | {line.rstrip()}")
                
            if end_line < total_lines:
                output.append(f"... {total_lines-end_line} lines after ...")
                
            output.append("```")
            return "\n".join(output)
            
    except Exception as e:
        return f"Error reading file: {str(e)}"

def execute(arguments: Dict) -> str:
    """Execute file operations tool
    
    Args:
        operation: Operation to perform (info, list, exists, find, size, create, delete, copy, move, write, read)
        path: Target path
        detailed: Get detailed information (for info operation)
        recursive: List subdirectories recursively (for list/find operations)
        pattern: Glob pattern for filtering (for list/find operations)
        content: Content to write (for write operation)
        destination: Destination path (for copy/move operations)
        mkdir: Create directory instead of file (for create operation)
        start_line: Start line number for read operation (1-based, inclusive)
        end_line: End line number for read operation (1-based, inclusive)
    """
    operation = arguments.get('operation', '').lower()
    path = arguments.get('path', '')
    detailed = bool(arguments.get('detailed', False))
    recursive = bool(arguments.get('recursive', False))
    pattern = arguments.get('pattern')
    content = arguments.get('content', '')
    destination = arguments.get('destination', '')
    mkdir = bool(arguments.get('mkdir', False))
    start_line = arguments.get('start_line')
    end_line = arguments.get('end_line')
    
    if not path:
        return "Error: No path provided"
    
    # Normalize paths
    path = os.path.normpath(path)
    if destination:
        destination = os.path.normpath(destination)
    
    try:
        if operation == 'info':
            info = get_file_info(path, detailed)
            if info['exists']:
                return json.dumps(info, indent=2)
            else:
                return f"Error: {info['error']}"
        
        elif operation == 'list':
            items = list_directory(path, recursive, pattern)
            return format_list_output(items)
        
        elif operation == 'exists':
            exists = os.path.exists(path)
            return f"Path {'exists' if exists else 'does not exist'}: {path}"
            
        elif operation == 'find':
            if not pattern:
                return "Error: Pattern required for find operation"
            items = list_directory(path, recursive=True, pattern=pattern)
            return format_list_output(items)
            
        elif operation == 'size':
            if not os.path.exists(path):
                return f"Error: Path does not exist: {path}"
                
            if os.path.isfile(path):
                size = os.path.getsize(path)
                return f"File size: {format_size(size)}"
            
            total_size = 0
            for root, dirs, files in os.walk(path):
                for f in files:
                    try:
                        total_size += os.path.getsize(os.path.join(root, f))
                    except:
                        continue
            return f"Directory size: {format_size(total_size)}"
            
        elif operation == 'create':
            if os.path.exists(path):
                return f"Error: Path already exists: {path}"
                
            try:
                if mkdir:
                    os.makedirs(path)
                    return f"Created directory: {path}"
                else:
                    ensure_parent_dir(path)
                    with open(path, 'w') as f:
                        f.write(content or '')
                    return f"Created file: {path}"
            except Exception as e:
                return f"Error creating {path}: {str(e)}"
                
        elif operation == 'delete':
            if not os.path.exists(path):
                return f"Error: Path does not exist: {path}"
                
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                    return f"Deleted directory: {path}"
                else:
                    os.remove(path)
                    return f"Deleted file: {path}"
            except Exception as e:
                return f"Error deleting {path}: {str(e)}"
                
        elif operation == 'copy':
            if not destination:
                return "Error: Destination path required for copy operation"
                
            if not os.path.exists(path):
                return f"Error: Source path does not exist: {path}"
                
            try:
                ensure_parent_dir(destination)
                if os.path.isdir(path):
                    shutil.copytree(path, destination)
                    return f"Copied directory from {path} to {destination}"
                else:
                    shutil.copy2(path, destination)
                    return f"Copied file from {path} to {destination}"
            except Exception as e:
                return f"Error copying {path}: {str(e)}"
                
        elif operation == 'move':
            if not destination:
                return "Error: Destination path required for move operation"
                
            if not os.path.exists(path):
                return f"Error: Source path does not exist: {path}"
                
            try:
                ensure_parent_dir(destination)
                shutil.move(path, destination)
                return f"Moved {path} to {destination}"
            except Exception as e:
                return f"Error moving {path}: {str(e)}"
                
        elif operation == 'write':
            if not content:
                return "Error: Content required for write operation"
                
            try:
                ensure_parent_dir(path)
                with open(path, 'w') as f:
                    f.write(content)
                return f"Written content to: {path}"
            except Exception as e:
                return f"Error writing to {path}: {str(e)}"
                
        elif operation == 'read':
            try:
                # Convert line numbers to integers if provided
                start = int(start_line) if start_line is not None else None
                end = int(end_line) if end_line is not None else None
                
                content = read_file_content(path, start, end)
                return content
            except Exception as e:
                return f"Error reading {path}: {str(e)}"
                
        else:
            return f"Error: Unknown operation '{operation}'. Valid operations: info, list, exists, find, size, create, delete, copy, move, write, read"
            
    except Exception as e:
        return f"Error performing {operation} on {path}: {str(e)}" 