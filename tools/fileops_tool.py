import os
import json
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple, Iterator
from pathlib import Path
import mimetypes
import pygments
from pygments.lexers import get_lexer_for_filename
from pygments.formatters import HtmlFormatter
import mmap
from io import StringIO
import itertools
import hashlib
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import magic  # python-magic for better file type detection
import chardet  # for encoding detection
import logging
import stat
from functools import wraps

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='fileops.log'
)
logger = logging.getLogger(__name__)

def log_operation(func):
    """Decorator to log file operations"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            logger.info(f"Successfully executed {func.__name__} with args: {args}, kwargs: {kwargs}")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise
    return wrapper

def check_path_access(path: str, check_type: str = 'read') -> bool:
    """Check if path is accessible and allowed
    
    Args:
        path: Path to check
        check_type: Type of access to check ('read', 'write', 'execute')
    """
    try:
        # Convert to absolute path
        abs_path = os.path.abspath(path)
        
        # Basic security checks
        if not os.path.exists(abs_path):
            return True  # Allow creating new files
            
        # Check if path is accessible
        if check_type == 'read':
            return os.access(abs_path, os.R_OK)
        elif check_type == 'write':
            return os.access(abs_path, os.W_OK)
        elif check_type == 'execute':
            return os.access(abs_path, os.X_OK)
        return False
        
    except Exception as e:
        logger.error(f"Error checking path access: {str(e)}")
        return False

@dataclass
class FileInsight:
    """Class for storing enhanced file metadata and insights"""
    path: str
    name: str
    size: int
    created: datetime
    modified: datetime
    file_type: str
    mime_type: str
    checksum: str
    encoding: Optional[str] = None
    is_binary: bool = False
    line_count: Optional[int] = None
    preview: Optional[str] = None
    metadata: Dict = None

@log_operation
def get_file_checksum(file_path: str, chunk_size: int = 8192) -> str:
    """Calculate file checksum using SHA-256"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()

@log_operation
def get_file_insight(path: str) -> FileInsight:
    """Get comprehensive file insights including metadata and preview"""
    try:
        stat = os.stat(path)
        mime = magic.Magic(mime=True)
        mime_type = mime.from_file(path)
        
        insight = FileInsight(
            path=path,
            name=os.path.basename(path),
            size=stat.st_size,
            created=datetime.fromtimestamp(stat.st_ctime),
            modified=datetime.fromtimestamp(stat.st_mtime),
            file_type=os.path.splitext(path)[1].lower(),
            mime_type=mime_type,
            checksum=get_file_checksum(path),
            metadata={}
        )
        
        # Determine if file is binary
        insight.is_binary = mime_type.startswith('application/') and 'text' not in mime_type.lower()
        
        # Handle text files
        if not insight.is_binary:
            try:
                # Try to detect encoding
                with open(path, 'rb') as f:
                    raw = f.read(4096)
                    result = chardet.detect(raw)
                    insight.encoding = result['encoding']
                
                # Get line count and preview for text files
                with open(path, 'r', encoding=insight.encoding) as f:
                    lines = f.readlines()
                    insight.line_count = len(lines)
                    insight.preview = ''.join(lines[:10])  # First 10 lines as preview
                    
            except Exception:
                insight.encoding = None
                insight.preview = None
        
        # Extract additional metadata based on file type
        if insight.file_type in {'.jpg', '.jpeg', '.png', '.gif'}:
            from PIL import Image
            with Image.open(path) as img:
                insight.metadata['dimensions'] = img.size
                insight.metadata['format'] = img.format
                insight.metadata['mode'] = img.mode
                
        elif insight.file_type in {'.pdf'}:
            import PyPDF2
            with open(path, 'rb') as f:
                pdf = PyPDF2.PdfReader(f)
                insight.metadata['pages'] = len(pdf.pages)
                insight.metadata['title'] = pdf.metadata.get('/Title', '')
                insight.metadata['author'] = pdf.metadata.get('/Author', '')
                
        elif insight.file_type in {'.docx', '.xlsx', '.pptx'}:
            from docx import Document
            doc = Document(path)
            insight.metadata['paragraphs'] = len(doc.paragraphs)
            insight.metadata['sections'] = len(doc.sections)
            
        return insight
        
    except Exception as e:
        return FileInsight(
            path=path,
            name=os.path.basename(path),
            size=0,
            created=datetime.now(),
            modified=datetime.now(),
            file_type='unknown',
            mime_type='unknown',
            checksum='',
            metadata={'error': str(e)}
        )

@log_operation
def analyze_directory(directory: str, max_depth: int = None, exclude_patterns: List[str] = None) -> List[FileInsight]:
    """Analyze all files in a directory recursively"""
    insights = []
    exclude_patterns = exclude_patterns or []
    
    def should_exclude(path: str) -> bool:
        return any(pattern in path for pattern in exclude_patterns)
    
    def process_file(file_path: str, current_depth: int) -> Optional[FileInsight]:
        if should_exclude(file_path):
            return None
        try:
            return get_file_insight(file_path)
        except Exception:
            return None
            
    def walk_directory(path: str, current_depth: int = 0):
        if max_depth is not None and current_depth > max_depth:
            return
            
        try:
            with ThreadPoolExecutor() as executor:
                for root, dirs, files in os.walk(path):
                    if should_exclude(root):
                        continue
                        
                    file_paths = [os.path.join(root, f) for f in files]
                    futures = [executor.submit(process_file, fp, current_depth) for fp in file_paths]
                    
                    for future in futures:
                        insight = future.result()
                        if insight:
                            insights.append(insight)
                            
                    for dir_name in dirs:
                        dir_path = os.path.join(root, dir_name)
                        walk_directory(dir_path, current_depth + 1)
        except Exception:
            pass
            
    walk_directory(directory)
    return insights

@log_operation
def format_size(size_bytes: int) -> str:
    """Convert bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"

@log_operation
def format_timestamp(timestamp: float) -> str:
    """Convert timestamp to human readable format"""
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

@log_operation
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

@log_operation
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

@log_operation
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

@log_operation
def ensure_parent_dir(path: str) -> None:
    """Ensure parent directory exists"""
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent)

@log_operation
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

@log_operation
def format_code_with_highlighting(content: str, language: str) -> str:
    """Format code with syntax highlighting"""
    try:
        # Get lexer for language
        try:
            lexer = get_lexer_for_filename(language)
        except Exception:
            lexer = get_lexer_for_filename('text')
            
        # Format with terminal colors
        formatter = HtmlFormatter()
        highlighted = pygments.highlight(content, lexer, formatter)
        
        return highlighted
    except:
        return content

@log_operation
def read_file_chunks(file_obj, chunk_size: int = 1024 * 1024) -> Iterator[str]:
    """Read a file in chunks to handle large files efficiently.
    
    Args:
        file_obj: File object to read from
        chunk_size: Size of each chunk in bytes (default 1MB)
        
    Yields:
        str: Chunks of the file content
    """
    while True:
        chunk = file_obj.read(chunk_size)
        if not chunk:
            break
        yield chunk

@log_operation
def read_file_content(path: str, start_line: Optional[int] = None, end_line: Optional[int] = None, 
                     chunk_size: int = 1024 * 1024, include_insights: bool = False) -> Union[str, Tuple[str, FileInsight]]:
    """Enhanced read file content with optional insights"""
    content = None
    insight = None
    
    try:
        if include_insights:
            insight = get_file_insight(path)
            
            # Use detected encoding if available
            if insight.encoding:
                encoding = insight.encoding
            else:
                encoding = 'utf-8'
                
            # Handle binary files appropriately
            if insight.is_binary:
                return (f"[Binary file detected: {insight.file_type}] - Use appropriate binary file handler", insight)
        else:
            encoding = 'utf-8'
            
        # Get file size
        file_size = os.path.getsize(path)
        
        # For large files (>50MB), use memory mapping or chunked reading
        if file_size > 50 * 1024 * 1024:  # 50MB threshold
            if start_line is not None or end_line is not None:
                content = read_large_file_lines(path, start_line, end_line, chunk_size)
            else:
                try:
                    with open(path, 'rb') as f:
                        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                            content = mm.read().decode(encoding, errors='replace')
                except Exception:
                    content = ''.join(chunked_read_generator(path, chunk_size))
        else:
            # Handle regular text-based files
            encodings = [encoding, 'utf-16', 'ascii', 'latin1']
            
            for enc in encodings:
                try:
                    with open(path, 'r', encoding=enc) as f:
                        if start_line is not None and end_line is not None:
                            lines = f.readlines()
                            start_idx = max(0, start_line)
                            end_idx = min(len(lines), end_line + 1)
                            content = ''.join(lines[start_idx:end_idx])
                        else:
                            content = f.read()
                        break
                except UnicodeDecodeError:
                    continue
                    
            # If no encoding worked, try binary mode as last resort
            if content is None:
                with open(path, 'rb') as f:
                    content = f.read().decode('utf-8', errors='replace')
                    
        if include_insights:
            return (content, insight)
        return content
                
    except Exception as e:
        error_msg = f"Error reading file: {str(e)}"
        if include_insights:
            return (error_msg, insight)
        return error_msg

@log_operation
def read_large_file_lines(path: str, start_line: Optional[int], end_line: Optional[int], 
                         chunk_size: int) -> str:
    """Read specific lines from a large file efficiently.
    
    Args:
        path: Path to the file
        start_line: Starting line number
        end_line: Ending line number
        chunk_size: Size of chunks to read
        
    Returns:
        str: Requested lines from the file
    """
    if start_line is None:
        start_line = 0
    if end_line is None:
        end_line = float('inf')
        
    buffer = StringIO()
    current_line = 0
    
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for chunk in read_file_chunks(f, chunk_size):
            lines = chunk.splitlines(True)
            for line in lines:
                if current_line >= start_line and current_line <= end_line:
                    buffer.write(line)
                current_line += 1
                if current_line > end_line:
                    break
            if current_line > end_line:
                break
                
    return buffer.getvalue()

@log_operation
def chunked_read_generator(path: str, chunk_size: int) -> Iterator[str]:
    """Generate chunks of file content for very large files.
    
    Args:
        path: Path to the file
        chunk_size: Size of chunks to read
        
    Yields:
        str: Chunks of the file content
    """
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for chunk in read_file_chunks(f, chunk_size):
            yield chunk

@log_operation
def execute(arguments: Dict) -> str:
    """Execute file operations tool
    
    Args:
        operation: Operation to perform (info, list, exists, find, size, create, delete, copy, move, write, read, search)
        path: Target path
        detailed: Get detailed information (for info operation)
        recursive: List subdirectories recursively (for list/find operations)
        pattern: Glob pattern for filtering (for list/find operations)
        content: Content to write (for write operation)
        destination: Destination path (for copy/move operations)
        mkdir: Create directory instead of file (for create operation)
        start_line: Start line number for read operation (1-based, inclusive)
        end_line: End line number for read operation (1-based, inclusive)
        query: Search query for search operation
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
    query = arguments.get('query', '')
    
    if not path:
        return "Error: No path provided"
    
    # Normalize and check paths
    try:
        path = os.path.normpath(path)
        if destination:
            destination = os.path.normpath(destination)
            
        # Check path access based on operation
        access_type = 'write' if operation in ['create', 'delete', 'copy', 'move', 'write'] else 'read'
        if not check_path_access(path, access_type):
            return f"Error: Access denied to path: {path}"
            
        if destination and not check_path_access(destination, 'write'):
            return f"Error: Access denied to destination: {destination}"
    except Exception as e:
        return f"Error with path validation: {str(e)}"
    
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
            
        elif operation == 'search':
            if not query:
                return "Error: Query required for search operation"
            results = []
            for root, dirs, files in os.walk(path):
                for file in files:
                    try:
                        file_path = os.path.join(root, file)
                        if not os.path.exists(file_path) or not os.access(file_path, os.R_OK):
                            continue
                            
                        # Get file type
                        mime = magic.Magic(mime=True)
                        mime_type = mime.from_file(file_path)
                        
                        # Only search text files
                        if mime_type and mime_type.startswith('text/'):
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                                if query.lower() in content.lower():
                                    results.append({
                                        'path': file_path,
                                        'type': 'file',
                                        'size': os.path.getsize(file_path)
                                    })
                    except Exception as e:
                        logger.error(f"Error searching file {file_path}: {str(e)}")
                        continue
                        
            return format_list_output(results)
            
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
            return f"Error: Unknown operation '{operation}'. Valid operations: info, list, exists, find, size, create, delete, copy, move, write, read, search"
            
    except Exception as e:
        logger.error(f"Error performing {operation} on {path}: {str(e)}")
        return f"Error performing {operation} on {path}: {str(e)}"