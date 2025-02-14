import os
import glob
import json
import logging
import mimetypes
import chardet
import getpass
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime

# Set up logging
logger = logging.getLogger(__name__)

# Constants
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB max file size for safety
MAX_SEARCH_RESULTS = 100
BINARY_EXTENSIONS = {'.exe', '.dll', '.bin', '.dat', '.db', '.mdb', '.accdb', '.iso'}
TEXT_EXTENSIONS = {'.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.xml', '.csv', '.log', '.ini', '.cfg', '.conf'}

class FileSystemError(Exception):
    """Custom exception for filesystem operations"""
    pass

def get_user_info() -> Dict:
    """Get current user information and common paths
    
    :return: Dictionary containing user information and paths
    """
    try:
        username = getpass.getuser()
        home = str(Path.home())
        desktop = os.path.join(home, 'Desktop')
        documents = os.path.join(home, 'Documents')
        downloads = os.path.join(home, 'Downloads')
        
        return {
            'username': username,
            'home': home,
            'desktop': desktop,
            'documents': documents,
            'downloads': downloads,
            'exists': {
                'desktop': os.path.exists(desktop),
                'documents': os.path.exists(documents),
                'downloads': os.path.exists(downloads)
            }
        }
    except Exception as e:
        raise FileSystemError(f"Error getting user info: {str(e)}")

def is_safe_path(path: str) -> bool:
    """Check if the path is safe to access
    
    :param path: Path to check
    :return: True if path is safe, False otherwise
    """
    try:
        # Convert to absolute path
        abs_path = os.path.abspath(path)
        
        # Check for system directories that should be protected
        system_paths = [
            os.environ.get('WINDIR', 'C:\\Windows'),
            os.environ.get('SYSTEMROOT', 'C:\\Windows'),
            os.environ.get('PROGRAMFILES', 'C:\\Program Files'),
            os.environ.get('PROGRAMFILES(X86)', 'C:\\Program Files (x86)'),
        ]
        
        return not any(abs_path.lower().startswith(p.lower()) for p in system_paths if p)
    except Exception:
        return False

def get_file_info(path: str, include_preview: bool = False) -> Dict:
    """Get detailed information about a file
    
    :param path: Path to the file
    :param include_preview: Whether to include a content preview
    :return: Dictionary containing file information
    """
    try:
        stat = os.stat(path)
        info = {
            'name': os.path.basename(path),
            'path': path,
            'size': stat.st_size,
            'size_human': f"{stat.st_size / 1024:.2f} KB" if stat.st_size < 1024 * 1024 else f"{stat.st_size / (1024 * 1024):.2f} MB",
            'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'accessed': datetime.fromtimestamp(stat.st_atime).isoformat(),
            'type': mimetypes.guess_type(path)[0] or 'application/octet-stream',
            'extension': os.path.splitext(path)[1].lower(),
            'is_binary': os.path.splitext(path)[1].lower() in BINARY_EXTENSIONS,
            'is_hidden': os.path.basename(path).startswith('.'),
            'is_readonly': not os.access(path, os.W_OK),
            'parent_dir': os.path.dirname(path)
        }
        
        if include_preview and not info['is_binary'] and stat.st_size < MAX_FILE_SIZE:
            try:
                encoding = detect_encoding(path)
                with open(path, 'r', encoding=encoding) as f:
                    preview = f.read(200)
                    info['preview'] = preview + '...' if len(preview) >= 200 else preview
            except Exception:
                info['preview'] = '[Preview not available]'
                
        return info
    except Exception as e:
        raise FileSystemError(f"Error getting file info: {str(e)}")

def detect_encoding(file_path: str) -> str:
    """Detect the encoding of a text file
    
    :param file_path: Path to the file
    :return: Detected encoding
    """
    try:
        with open(file_path, 'rb') as f:
            raw = f.read(min(MAX_FILE_SIZE, os.path.getsize(file_path)))
            result = chardet.detect(raw)
            return result['encoding'] or 'utf-8'
    except Exception:
        return 'utf-8'

def read_file_content(path: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> str:
    """Read content from a file with optional line range
    
    :param path: Path to the file
    :param start_line: Starting line number (1-based, inclusive)
    :param end_line: Ending line number (1-based, inclusive)
    :return: File content as string
    """
    try:
        if os.path.splitext(path)[1].lower() in BINARY_EXTENSIONS:
            return "[Binary file content not shown]"
            
        file_size = os.path.getsize(path)
        if file_size > MAX_FILE_SIZE:
            return f"[File too large to read directly. Size: {file_size / (1024 * 1024):.2f} MB. Max size: {MAX_FILE_SIZE / (1024 * 1024)} MB]"
            
        encoding = detect_encoding(path)
        with open(path, 'r', encoding=encoding) as f:
            if start_line is None or end_line is None:
                return f.read()  # Read entire file
            else:
                lines = f.readlines()
                start_idx = max(0, start_line - 1)
                end_idx = min(len(lines), end_line)
                return ''.join(lines[start_idx:end_idx])
    except Exception as e:
        raise FileSystemError(f"Error reading file: {str(e)}")

def list_directory(path: str, pattern: str = "*", include_hidden: bool = False) -> List[Dict]:
    """List contents of a directory with detailed information
    
    :param path: Directory path to list
    :param pattern: File pattern to match
    :param include_hidden: Whether to include hidden files
    :return: List of file/directory information
    """
    try:
        results = []
        entries = os.scandir(path)
        
        for entry in entries:
            try:
                if not include_hidden and entry.name.startswith('.'):
                    continue
                    
                if pattern != "*" and not glob.fnmatch.fnmatch(entry.name, pattern):
                    continue
                    
                info = {
                    'name': entry.name,
                    'path': entry.path,
                    'is_file': entry.is_file(),
                    'is_dir': entry.is_dir(),
                    'is_hidden': entry.name.startswith('.'),
                }
                
                if entry.is_file():
                    file_info = get_file_info(entry.path, include_preview=True)
                    info.update(file_info)
                else:
                    try:
                        stat = entry.stat()
                        info.update({
                            'size': sum(f.stat().st_size for f in os.scandir(entry.path) if f.is_file()),
                            'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                            'accessed': datetime.fromtimestamp(stat.st_atime).isoformat(),
                            'item_count': sum(1 for _ in os.scandir(entry.path))
                        })
                    except Exception:
                        # If can't access directory contents
                        info.update({
                            'size': 0,
                            'item_count': 0,
                            'error': 'Access denied'
                        })
                        
                results.append(info)
                
            except Exception as e:
                logger.error(f"Error processing entry {entry.path}: {str(e)}")
                continue
                
        return sorted(results, key=lambda x: (not x['is_dir'], x['name'].lower()))
    except Exception as e:
        raise FileSystemError(f"Error listing directory: {str(e)}")

def search_files(
    root_path: str,
    pattern: str = "*",
    content_query: Optional[str] = None,
    max_depth: int = 10,
    exclude_dirs: Optional[List[str]] = None,
    include_hidden: bool = False
) -> List[Dict]:
    """Search for files matching pattern and optionally containing text
    
    :param root_path: Root directory to start search from
    :param pattern: File pattern to match (e.g., "*.txt", "*.py")
    :param content_query: Optional text to search for within files
    :param max_depth: Maximum directory depth to search
    :param exclude_dirs: List of directory names to exclude
    :param include_hidden: Whether to include hidden files
    :return: List of matching file information
    """
    if not exclude_dirs:
        exclude_dirs = ['.git', 'node_modules', '__pycache__', 'venv', '.env']
        
    results = []
    try:
        for root, dirs, files in os.walk(root_path):
            # Check depth
            depth = root[len(root_path):].count(os.sep)
            if depth > max_depth:
                continue
                
            # Skip excluded and hidden directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs and (include_hidden or not d.startswith('.'))]
            
            for file in files:
                if not include_hidden and file.startswith('.'):
                    continue
                    
                if not glob.fnmatch.fnmatch(file, pattern):
                    continue
                    
                file_path = os.path.join(root, file)
                
                # Skip if path is not safe
                if not is_safe_path(file_path):
                    continue
                    
                try:
                    file_info = get_file_info(file_path, include_preview=True)
                    
                    # If content search is requested
                    if content_query and file_info['extension'] in TEXT_EXTENSIONS:
                        try:
                            encoding = detect_encoding(file_path)
                            with open(file_path, 'r', encoding=encoding) as f:
                                content = f.read(MAX_FILE_SIZE)
                                if content_query.lower() in content.lower():
                                    results.append(file_info)
                        except Exception:
                            continue
                    else:
                        results.append(file_info)
                        
                    if len(results) >= MAX_SEARCH_RESULTS:
                        return results
                except Exception:
                    continue
                    
        return results
    except Exception as e:
        raise FileSystemError(f"Error searching files: {str(e)}")

def execute(arguments: Dict) -> str:
    """Execute filesystem operations for searching and reading files
    
    :param operation: Operation to perform (search, read, info, list, user)
    :param path: Target path for the operation
    :param pattern: File pattern for search (e.g., "*.txt", "*.py")
    :param content: Text to search for within files
    :param max_depth: Maximum directory depth for search
    :param exclude_dirs: List of directory names to exclude from search
    :param include_hidden: Whether to include hidden files
    :param start_line: Starting line number for read operation (1-based, inclusive)
    :param end_line: Ending line number for read operation (1-based, inclusive)
    :return: Operation result as JSON string
    
    Example:
        >>> # Get user info and common paths
        >>> result = execute({
        ...     "operation": "user"
        ... })
        >>> 
        >>> # List desktop contents
        >>> result = execute({
        ...     "operation": "list",
        ...     "path": "C:\\Users\\username\\Desktop"
        ... })
        >>> 
        >>> # Search for Python files containing "import"
        >>> result = execute({
        ...     "operation": "search",
        ...     "path": "C:\\Projects",
        ...     "pattern": "*.py",
        ...     "content": "import"
        ... })
        >>> 
        >>> # Read entire file
        >>> result = execute({
        ...     "operation": "read",
        ...     "path": "C:\\Projects\\example.py"
        ... })
    """
    try:
        operation = arguments.get('operation', '').lower()
        
        # Special case for user info operation
        if operation == 'user':
            user_info = get_user_info()
            return json.dumps({
                "operation": "user",
                "user_info": user_info
            }, indent=2)
            
        # All other operations require a path
        path = arguments.get('path', '')
        if not path:
            return json.dumps({"error": "Path is required"}, indent=2)
            
        if not is_safe_path(path):
            return json.dumps({"error": "Access to this path is not allowed"}, indent=2)
            
        if operation == 'search':
            pattern = arguments.get('pattern', '*')
            content = arguments.get('content')
            max_depth = int(arguments.get('max_depth', 10))
            exclude_dirs = arguments.get('exclude_dirs', [])
            include_hidden = bool(arguments.get('include_hidden', False))
            
            results = search_files(
                path,
                pattern=pattern,
                content_query=content,
                max_depth=max_depth,
                exclude_dirs=exclude_dirs,
                include_hidden=include_hidden
            )
            
            return json.dumps({
                "operation": "search",
                "path": path,
                "pattern": pattern,
                "content_query": content,
                "results": results,
                "total_results": len(results),
                "max_results_reached": len(results) >= MAX_SEARCH_RESULTS
            }, indent=2)
            
        elif operation == 'read':
            start_line = arguments.get('start_line')
            end_line = arguments.get('end_line')
            
            if start_line:
                start_line = int(start_line)
            if end_line:
                end_line = int(end_line)
                
            content = read_file_content(path, start_line, end_line)
            file_info = get_file_info(path)
            
            return json.dumps({
                "operation": "read",
                "file_info": file_info,
                "content": content,
                "start_line": start_line,
                "end_line": end_line
            }, indent=2)
            
        elif operation == 'list':
            pattern = arguments.get('pattern', '*')
            include_hidden = bool(arguments.get('include_hidden', False))
            
            contents = list_directory(path, pattern, include_hidden)
            return json.dumps({
                "operation": "list",
                "path": path,
                "pattern": pattern,
                "contents": contents,
                "total_items": len(contents)
            }, indent=2)
            
        elif operation == 'info':
            include_preview = bool(arguments.get('include_preview', False))
            file_info = get_file_info(path, include_preview)
            return json.dumps({
                "operation": "info",
                "file_info": file_info
            }, indent=2)
            
        else:
            return json.dumps({
                "error": f"Invalid operation: {operation}",
                "valid_operations": ["search", "read", "info", "list", "user"]
            }, indent=2)
            
    except Exception as e:
        return json.dumps({
            "error": str(e),
            "type": type(e).__name__
        }, indent=2) 