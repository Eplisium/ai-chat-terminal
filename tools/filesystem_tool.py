import os
import glob
import json
import logging
import mimetypes
import chardet
import getpass
import shutil
import hashlib
import stat
import re
import time
import platform
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Union, BinaryIO, TextIO, Tuple, Any, Set
from datetime import datetime
from dotenv import load_dotenv  # Added for .env file support

# Set up logging
logger = logging.getLogger(__name__)

# Constants
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB max file size for safety
MAX_SEARCH_RESULTS = 100
BINARY_EXTENSIONS = {'.exe', '.dll', '.bin', '.dat', '.db', '.mdb', '.accdb', '.iso', '.img', '.zip', '.tar', '.gz', '.7z', '.rar'}
TEXT_EXTENSIONS = {'.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.xml', '.csv', '.log', '.ini', '.cfg', '.conf', '.bat', '.markdown', '.yml', '.yaml', '.java', '.c', '.cpp', '.h', '.php', '.sql', '.ps1', '.sh', '.ts', '.tsx', '.jsx', '.rb'}
HASH_ALGORITHMS = {'md5', 'sha1', 'sha256', 'sha512'}
DEFAULT_ENCODING = 'utf-8'
DEFAULT_CHUNK_SIZE = 8192  # 8KB chunks for file operations

class FileSystemError(Exception):
    """Custom exception for filesystem operations"""
    pass

# ========================
# Helper Functions
# ========================

def normalize_path(path: str) -> str:
    """Normalize a path for the current platform
    
    :param path: Path to normalize
    :return: Normalized path
    """
    try:
        return os.path.normpath(os.path.expanduser(path))
    except Exception as e:
        raise FileSystemError(f"Error normalizing path: {str(e)}")

def is_valid_filename(filename: str) -> bool:
    """Check if a filename is valid
    
    :param filename: Filename to check
    :return: True if valid, False otherwise
    """
    if not filename or len(filename) > 255:
        return False
        
    # Check for invalid characters based on OS
    if platform.system() == 'Windows':
        invalid_chars = r'[<>:"/\\|?*\x00-\x1f]'
    else:  # Unix-based systems
        invalid_chars = r'[/\x00]'
        
    return not bool(re.search(invalid_chars, filename))

def get_drive_info() -> Dict:
    """Get information about available drives
    
    :return: Dictionary of drive information
    """
    try:
        import psutil
        drives = {}
        
        for part in psutil.disk_partitions(all=True):
            if os.path.exists(part.mountpoint):
                try:
                    usage = psutil.disk_usage(part.mountpoint)
                    drives[part.mountpoint] = {
                        'device': part.device,
                        'fstype': part.fstype,
                        'opts': part.opts,
                        'total': usage.total,
                        'used': usage.used,
                        'free': usage.free,
                        'percent': usage.percent,
                        'total_human': f"{usage.total / (1024**3):.2f} GB",
                        'used_human': f"{usage.used / (1024**3):.2f} GB",
                        'free_human': f"{usage.free / (1024**3):.2f} GB"
                    }
                except (PermissionError, OSError):
                    continue
                    
        return drives
    except ImportError:
        # Fallback method if psutil is not available
        drives = {}
        
        if platform.system() == 'Windows':
            import win32api
            drives_list = win32api.GetLogicalDriveStrings().split('\000')[:-1]
            for drive in drives_list:
                try:
                    info = shutil.disk_usage(drive)
                    drives[drive] = {
                        'total': info.total,
                        'used': info.used,
                        'free': info.free,
                        'percent': (info.used / info.total) * 100 if info.total > 0 else 0,
                        'total_human': f"{info.total / (1024**3):.2f} GB",
                        'used_human': f"{info.used / (1024**3):.2f} GB",
                        'free_human': f"{info.free / (1024**3):.2f} GB"
                    }
                except (PermissionError, OSError):
                    continue
        else:
            # For Unix-based systems, check standard mount points
            mount_points = ['/']
            if platform.system() == 'Darwin':  # macOS
                mount_points.append('/Volumes')
            else:  # Linux
                mount_points.extend(['/home', '/mnt', '/media'])
                
            for mount in mount_points:
                if os.path.exists(mount):
                    try:
                        info = shutil.disk_usage(mount)
                        drives[mount] = {
                            'total': info.total,
                            'used': info.used,
                            'free': info.free,
                            'percent': (info.used / info.total) * 100 if info.total > 0 else 0,
                            'total_human': f"{info.total / (1024**3):.2f} GB",
                            'used_human': f"{info.used / (1024**3):.2f} GB",
                            'free_human': f"{info.free / (1024**3):.2f} GB"
                        }
                    except (PermissionError, OSError):
                        continue
                        
        return drives
    except Exception as e:
        raise FileSystemError(f"Error getting drive information: {str(e)}")

def parse_path(path: str) -> Dict:
    """Parse a path into components
    
    :param path: Path to parse
    :return: Dictionary with path components
    """
    try:
        path_obj = Path(path)
        return {
            'original': path,
            'normalized': str(path_obj),
            'absolute': str(path_obj.absolute()),
            'parent': str(path_obj.parent),
            'name': path_obj.name,
            'stem': path_obj.stem,
            'suffix': path_obj.suffix,
            'is_absolute': path_obj.is_absolute(),
            'exists': path_obj.exists(),
            'is_file': path_obj.is_file() if path_obj.exists() else None,
            'is_dir': path_obj.is_dir() if path_obj.exists() else None
        }
    except Exception as e:
        raise FileSystemError(f"Error parsing path: {str(e)}")

def calculate_checksum(path: str, algorithm: str = 'sha256', chunk_size: int = DEFAULT_CHUNK_SIZE) -> str:
    """Calculate checksum for a file
    
    :param path: Path to the file
    :param algorithm: Hash algorithm to use ('md5', 'sha1', 'sha256', 'sha512')
    :param chunk_size: Size of chunks to read for large files
    :return: Hexadecimal digest of the hash
    """
    if algorithm not in HASH_ALGORITHMS:
        raise FileSystemError(f"Unsupported hash algorithm: {algorithm}")
        
    try:
        hasher = getattr(hashlib, algorithm)()
        
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(chunk_size), b''):
                hasher.update(chunk)
                
        return hasher.hexdigest()
    except Exception as e:
        raise FileSystemError(f"Error calculating checksum: {str(e)}")

def calculate_multiple_checksums(path: str, algorithms: List[str] = None) -> Dict[str, str]:
    """Calculate multiple checksums for a file
    
    :param path: Path to the file
    :param algorithms: List of hash algorithms to use (default: sha256)
    :return: Dictionary of algorithm->hash pairs
    """
    if not algorithms:
        algorithms = ['sha256']
    
    results = {}
    for algorithm in algorithms:
        if algorithm in HASH_ALGORITHMS:
            results[algorithm] = calculate_checksum(path, algorithm)
            
    return results

def get_permissions_string(path: str) -> str:
    """Get Unix-style permission string for a file
    
    :param path: Path to the file
    :return: Permission string (e.g., 'rwxr-xr--')
    """
    try:
        mode = os.stat(path).st_mode
        perm_string = ''
        
        # Owner permissions
        perm_string += 'r' if mode & stat.S_IRUSR else '-'
        perm_string += 'w' if mode & stat.S_IWUSR else '-'
        perm_string += 'x' if mode & stat.S_IXUSR else '-'
        
        # Group permissions
        perm_string += 'r' if mode & stat.S_IRGRP else '-'
        perm_string += 'w' if mode & stat.S_IWGRP else '-'
        perm_string += 'x' if mode & stat.S_IXGRP else '-'
        
        # Other permissions
        perm_string += 'r' if mode & stat.S_IROTH else '-'
        perm_string += 'w' if mode & stat.S_IWOTH else '-'
        perm_string += 'x' if mode & stat.S_IXOTH else '-'
        
        return perm_string
    except Exception:
        return '?????????'  # Return a question mark string if permissions can't be read

def is_path_accessible(path: str, check_write: bool = False) -> bool:
    """Check if a path is accessible
    
    :param path: Path to check
    :param check_write: Whether to check for write access
    :return: True if accessible, False otherwise
    """
    try:
        if os.path.exists(path):
            if check_write:
                return os.access(path, os.R_OK | os.W_OK)
            else:
                return os.access(path, os.R_OK)
        elif check_write:
            # If path doesn't exist, check if parent directory is writable
            parent = os.path.dirname(path)
            return os.path.exists(parent) and os.access(parent, os.W_OK)
        return False
    except Exception:
        return False

def create_secure_temp_file(content: str = None, suffix: str = None) -> str:
    """Create a secure temporary file with optional content
    
    :param content: Optional content to write to file
    :param suffix: Optional suffix for the temporary file
    :return: Path to the created temporary file
    """
    try:
        fd, temp_path = tempfile.mkstemp(suffix=suffix)
        
        if content:
            with os.fdopen(fd, 'w') as f:
                f.write(content)
        else:
            os.close(fd)
            
        return temp_path
    except Exception as e:
        raise FileSystemError(f"Error creating temporary file: {str(e)}")

def get_user_info() -> Dict:
    """Get current user information and common paths
    
    :return: Dictionary containing user information and paths
    """
    try:
        # Load environment variables from .env file
        load_dotenv()
        
        username = getpass.getuser()
        home = str(Path.home())
        
        # Use environment variables if available, otherwise use default paths
        desktop = os.environ.get('DESKTOP') or os.path.join(home, 'Desktop')
        documents = os.environ.get('DOCUMENTS') or os.path.join(home, 'Documents')
        downloads = os.environ.get('DOWNLOADS') or os.path.join(home, 'Downloads')
        
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

def detect_encoding(file_path: str) -> str:
    """Detect the encoding of a text file
    
    :param file_path: Path to the file
    :return: Detected encoding
    """
    try:
        with open(file_path, 'rb') as f:
            raw = f.read(min(MAX_FILE_SIZE, os.path.getsize(file_path)))
            result = chardet.detect(raw)
            return result['encoding'] or DEFAULT_ENCODING
    except Exception:
        return DEFAULT_ENCODING

# ========================
# File Information Functions
# ========================

def get_file_info(path: str, include_preview: bool = False, include_metadata: bool = False) -> Dict:
    """Get detailed information about a file
    
    :param path: Path to the file
    :param include_preview: Whether to include a content preview
    :param include_metadata: Whether to include extended metadata
    :return: Dictionary containing file information
    """
    try:
        stat_info = os.stat(path)
        info = {
            'name': os.path.basename(path),
            'path': path,
            'size': stat_info.st_size,
            'size_human': f"{stat_info.st_size / 1024:.2f} KB" if stat_info.st_size < 1024 * 1024 else f"{stat_info.st_size / (1024 * 1024):.2f} MB",
            'created': datetime.fromtimestamp(stat_info.st_ctime).isoformat(),
            'modified': datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
            'accessed': datetime.fromtimestamp(stat_info.st_atime).isoformat(),
            'type': mimetypes.guess_type(path)[0] or 'application/octet-stream',
            'extension': os.path.splitext(path)[1].lower(),
            'is_binary': os.path.splitext(path)[1].lower() in BINARY_EXTENSIONS,
            'is_hidden': os.path.basename(path).startswith('.'),
            'is_readonly': not os.access(path, os.W_OK),
            'is_system': bool(stat_info.st_file_attributes & stat.FILE_ATTRIBUTE_SYSTEM) if hasattr(stat_info, 'st_file_attributes') else False,
            'parent_dir': os.path.dirname(path),
            'permissions': get_permissions_string(path)
        }
        
        if include_preview and not info['is_binary'] and stat_info.st_size < MAX_FILE_SIZE:
            try:
                encoding = detect_encoding(path)
                with open(path, 'r', encoding=encoding) as f:
                    preview = f.read(200)
                    info['preview'] = preview + '...' if len(preview) >= 200 else preview
            except Exception:
                info['preview'] = '[Preview not available]'
        
        if include_metadata:
            # Add additional metadata
            try:
                checksums = calculate_multiple_checksums(path, ['md5', 'sha256'])
                info['checksums'] = checksums
                
                # File-specific metadata based on type
                mime_type = info['type']
                if mime_type:
                    if mime_type.startswith('image/'):
                        try:
                            from PIL import Image, ExifTags
                            image = Image.open(path)
                            info['metadata'] = {
                                'format': image.format,
                                'mode': image.mode,
                                'width': image.width,
                                'height': image.height
                            }
                            
                            # Get EXIF data if available
                            if hasattr(image, '_getexif') and image._getexif():
                                exif = {
                                    ExifTags.TAGS[k]: v
                                    for k, v in image._getexif().items()
                                    if k in ExifTags.TAGS and isinstance(v, (str, int, float, bytes))
                                }
                                info['metadata']['exif'] = exif
                        except Exception:
                            pass
                    elif mime_type.startswith('video/') or mime_type.startswith('audio/'):
                        # Basic audio/video metadata
                        info['metadata'] = {
                            'size_mb': f"{stat_info.st_size / (1024 * 1024):.2f} MB"
                        }
                    elif mime_type == 'application/pdf':
                        try:
                            import PyPDF2
                            with open(path, 'rb') as pdf_file:
                                pdf_reader = PyPDF2.PdfFileReader(pdf_file)
                                info['metadata'] = {
                                    'pages': pdf_reader.getNumPages(),
                                    'encrypted': pdf_reader.isEncrypted
                                }
                                
                                if not pdf_reader.isEncrypted and pdf_reader.documentInfo:
                                    info['metadata']['document_info'] = {k: str(v) for k, v in pdf_reader.documentInfo.items()}
                        except Exception:
                            pass
            except Exception:
                pass
                
        return info
    except Exception as e:
        raise FileSystemError(f"Error getting file info: {str(e)}")

def get_metadata(path: str) -> Dict:
    """Get extended metadata for a file
    
    :param path: Path to the file
    :return: Dictionary containing metadata
    """
    file_info = get_file_info(path, include_preview=False, include_metadata=True)
    return file_info.get('metadata', {})

# ========================
# File Reading Functions
# ========================

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

def read_file_chunk(path: str, offset: int = 0, size: int = 1024) -> Dict:
    """Read a chunk of a file from a specific offset
    
    :param path: Path to the file
    :param offset: Starting byte offset
    :param size: Number of bytes to read
    :return: Dictionary with content and position information
    """
    try:
        file_size = os.path.getsize(path)
        
        # Check if file is binary
        if os.path.splitext(path)[1].lower() in BINARY_EXTENSIONS:
            return {
                'file_size': file_size,
                'offset': offset,
                'size': 0,
                'content': "[Binary file content not shown]",
                'is_binary': True
            }
            
        # Validate parameters
        if offset < 0:
            offset = 0
        if offset >= file_size:
            offset = max(0, file_size - 1)
        if size <= 0:
            size = 1024
        if size > MAX_FILE_SIZE:
            size = MAX_FILE_SIZE
            
        # Read the chunk
        encoding = detect_encoding(path)
        with open(path, 'r', encoding=encoding) as f:
            f.seek(offset)
            content = f.read(size)
            
        return {
            'file_size': file_size,
            'offset': offset,
            'size': len(content),
            'content': content,
            'is_binary': False,
            'encoding': encoding
        }
    except Exception as e:
        raise FileSystemError(f"Error reading file chunk: {str(e)}")

# ========================
# Directory Functions
# ========================

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

def list_directory_tree(path: str, max_depth: int = 3, include_hidden: bool = False) -> Dict:
    """List directory tree structure up to a specified depth
    
    :param path: Directory path to list
    :param max_depth: Maximum depth to traverse
    :param include_hidden: Whether to include hidden files
    :return: Dictionary representing tree structure
    """
    def _build_tree(current_path, depth=0):
        if depth > max_depth:
            return {'truncated': True}
            
        result = {'name': os.path.basename(current_path) or current_path, 'path': current_path, 'children': []}
        
        try:
            entries = sorted(os.scandir(current_path), key=lambda e: (not e.is_dir(), e.name.lower()))
            
            for entry in entries:
                if not include_hidden and entry.name.startswith('.'):
                    continue
                    
                if entry.is_dir():
                    child = _build_tree(entry.path, depth + 1)
                    result['children'].append(child)
                else:
                    result['children'].append({
                        'name': entry.name,
                        'path': entry.path,
                        'is_file': True,
                        'size': entry.stat().st_size,
                        'size_human': f"{entry.stat().st_size / 1024:.2f} KB" if entry.stat().st_size < 1024 * 1024 else f"{entry.stat().st_size / (1024 * 1024):.2f} MB"
                    })
        except PermissionError:
            result['error'] = 'Permission denied'
        except Exception as e:
            result['error'] = str(e)
            
        return result
    
    try:
        return _build_tree(path)
    except Exception as e:
        raise FileSystemError(f"Error listing directory tree: {str(e)}")

# ========================
# Search Functions
# ========================

def search_files(
    root_path: str,
    pattern: str = "*",
    content_query: Optional[str] = None,
    max_depth: int = 10,
    exclude_dirs: Optional[List[str]] = None,
    include_hidden: bool = False,
    min_size: Optional[int] = None,
    max_size: Optional[int] = None,
    modified_after: Optional[str] = None,
    modified_before: Optional[str] = None,
    case_sensitive: bool = False,
    file_extensions: Optional[List[str]] = None,
    regex_pattern: bool = False
) -> List[Dict]:
    """Search for files matching pattern and optionally containing text
    
    :param root_path: Root directory to start search from
    :param pattern: File pattern to match (e.g., "*.txt", "*.py")
    :param content_query: Optional text to search for within files
    :param max_depth: Maximum directory depth to search
    :param exclude_dirs: List of directory names to exclude
    :param include_hidden: Whether to include hidden files
    :param min_size: Minimum file size in bytes
    :param max_size: Maximum file size in bytes
    :param modified_after: Include files modified after this date (ISO format)
    :param modified_before: Include files modified before this date (ISO format)
    :param case_sensitive: Whether to perform case-sensitive content search
    :param file_extensions: List of file extensions to include (e.g., ['.txt', '.md'])
    :param regex_pattern: Whether the content_query is a regex pattern
    :return: List of matching file information
    """
    if not exclude_dirs:
        exclude_dirs = ['.git', 'node_modules', '__pycache__', 'venv', '.env']
        
    # Convert date strings to timestamps if provided
    mod_after_ts = None
    mod_before_ts = None
    
    if modified_after:
        try:
            mod_after_ts = datetime.fromisoformat(modified_after).timestamp()
        except (ValueError, TypeError):
            logger.warning(f"Invalid modified_after date format: {modified_after}")
            
    if modified_before:
        try:
            mod_before_ts = datetime.fromisoformat(modified_before).timestamp()
        except (ValueError, TypeError):
            logger.warning(f"Invalid modified_before date format: {modified_before}")
    
    # Compile regex if needed
    regex = None
    if content_query and regex_pattern:
        try:
            regex_flags = 0 if case_sensitive else re.IGNORECASE
            regex = re.compile(content_query, regex_flags)
        except re.error:
            logger.warning(f"Invalid regex pattern: {content_query}")
            regex_pattern = False
            
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
                    
                # Check file pattern
                if pattern != "*" and not glob.fnmatch.fnmatch(file, pattern):
                    continue
                    
                file_path = os.path.join(root, file)
                
                # Skip if path is not safe
                if not is_safe_path(file_path):
                    continue
                    
                try:
                    # Get file information
                    stat_info = os.stat(file_path)
                    file_size = stat_info.st_size
                    file_mtime = stat_info.st_mtime
                    file_ext = os.path.splitext(file)[1].lower()
                    
                    # Apply filters
                    if min_size is not None and file_size < min_size:
                        continue
                    if max_size is not None and file_size > max_size:
                        continue
                    if mod_after_ts is not None and file_mtime < mod_after_ts:
                        continue
                    if mod_before_ts is not None and file_mtime > mod_before_ts:
                        continue
                    if file_extensions and file_ext not in file_extensions:
                        continue
                        
                    # Check content if needed
                    if content_query and file_ext in TEXT_EXTENSIONS and file_size < MAX_FILE_SIZE:
                        try:
                            encoding = detect_encoding(file_path)
                            with open(file_path, 'r', encoding=encoding) as f:
                                content = f.read()
                                
                                if regex_pattern and regex:
                                    if not regex.search(content):
                                        continue
                                else:
                                    if case_sensitive:
                                        if content_query not in content:
                                            continue
                                    else:
                                        if content_query.lower() not in content.lower():
                                            continue
                        except Exception:
                            continue
                            
                    # If we made it here, add file to results
                    file_info = get_file_info(file_path, include_preview=True)
                    results.append(file_info)
                    
                    if len(results) >= MAX_SEARCH_RESULTS:
                        return results
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {str(e)}")
                    continue
                    
        return results
    except Exception as e:
        raise FileSystemError(f"Error searching files: {str(e)}")

def find_duplicate_files(
    root_path: str,
    hash_algorithm: str = 'sha256',
    exclude_dirs: Optional[List[str]] = None,
    min_size: int = 1024,  # 1KB minimum to improve performance
    include_hidden: bool = False
) -> Dict[str, List[str]]:
    """Find duplicate files by content hash
    
    :param root_path: Root directory to start search from
    :param hash_algorithm: Hash algorithm to use for comparison
    :param exclude_dirs: List of directory names to exclude
    :param min_size: Minimum file size to consider
    :param include_hidden: Whether to include hidden files
    :return: Dictionary of hash -> file paths
    """
    if hash_algorithm not in HASH_ALGORITHMS:
        raise FileSystemError(f"Unsupported hash algorithm: {hash_algorithm}")
        
    if not exclude_dirs:
        exclude_dirs = ['.git', 'node_modules', '__pycache__', 'venv', '.env']
        
    # Dictionary to store file hashes -> list of file paths
    files_by_size = {}
    duplicates = {}
    
    try:
        # First, group files by size to reduce hash calculations
        for root, dirs, files in os.walk(root_path):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs and (include_hidden or not d.startswith('.'))]
            
            for filename in files:
                if not include_hidden and filename.startswith('.'):
                    continue
                    
                file_path = os.path.join(root, filename)
                
                if not is_safe_path(file_path):
                    continue
                    
                try:
                    file_size = os.path.getsize(file_path)
                    
                    # Skip small files if minimum size is set
                    if file_size < min_size:
                        continue
                        
                    # Group files by size
                    if file_size not in files_by_size:
                        files_by_size[file_size] = []
                    files_by_size[file_size].append(file_path)
                except Exception:
                    continue
                    
        # For each group of same-sized files, calculate hashes
        for size, files in files_by_size.items():
            if len(files) < 2:
                continue  # Skip if only one file with this size
                
            for file_path in files:
                try:
                    file_hash = calculate_checksum(file_path, hash_algorithm)
                    
                    if file_hash not in duplicates:
                        duplicates[file_hash] = []
                    duplicates[file_hash].append(file_path)
                except Exception:
                    continue
                    
        # Filter out unique files
        return {hash_val: paths for hash_val, paths in duplicates.items() if len(paths) > 1}
    except Exception as e:
        raise FileSystemError(f"Error finding duplicate files: {str(e)}")

# ========================
# File Operations Functions
# ========================

def create_directory(path: str, mode: int = 0o755, exist_ok: bool = False) -> Dict:
    """Create a new directory
    
    :param path: Path to the directory to create
    :param mode: Directory permissions (octal)
    :param exist_ok: Don't error if directory already exists
    :return: Dictionary with operation result
    """
    try:
        if not is_safe_path(path):
            raise FileSystemError("Path is not safe for writing")
            
        os.makedirs(path, mode=mode, exist_ok=exist_ok)
        
        return {
            'success': True,
            'operation': 'create_directory',
            'path': path,
            'created': datetime.now().isoformat()
        }
    except FileExistsError as e:
        if exist_ok:
            return {
                'success': True,
                'operation': 'create_directory',
                'path': path,
                'warning': 'Directory already exists'
            }
        else:
            raise FileSystemError(f"Directory already exists: {str(e)}")
    except Exception as e:
        raise FileSystemError(f"Error creating directory: {str(e)}")

def create_file(path: str, content: str = '', encoding: str = DEFAULT_ENCODING) -> Dict:
    """Create a new file with optional content
    
    :param path: Path to the file to create
    :param content: Optional content to write to the file
    :param encoding: File encoding
    :return: Dictionary with operation result
    """
    try:
        if not is_safe_path(path):
            raise FileSystemError("Path is not safe for writing")
            
        # Create parent directory if it doesn't exist
        parent_dir = os.path.dirname(path)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
            
        with open(path, 'w', encoding=encoding) as f:
            f.write(content)
            
        return {
            'success': True,
            'operation': 'create_file',
            'path': path,
            'size': len(content),
            'created': datetime.now().isoformat()
        }
    except Exception as e:
        raise FileSystemError(f"Error creating file: {str(e)}")

def write_file(path: str, content: str, encoding: str = DEFAULT_ENCODING) -> Dict:
    """Write content to a file (overwriting existing content)
    
    :param path: Path to the file
    :param content: Content to write
    :param encoding: File encoding
    :return: Dictionary with operation result
    """
    try:
        if not is_safe_path(path):
            raise FileSystemError("Path is not safe for writing")
            
        with open(path, 'w', encoding=encoding) as f:
            f.write(content)
            
        return {
            'success': True,
            'operation': 'write_file',
            'path': path,
            'size': len(content),
            'modified': datetime.now().isoformat()
        }
    except Exception as e:
        raise FileSystemError(f"Error writing to file: {str(e)}")

def append_file(path: str, content: str, encoding: str = DEFAULT_ENCODING) -> Dict:
    """Append content to a file
    
    :param path: Path to the file
    :param content: Content to append
    :param encoding: File encoding
    :return: Dictionary with operation result
    """
    try:
        if not is_safe_path(path):
            raise FileSystemError("Path is not safe for writing")
            
        # Create file if it doesn't exist
        if not os.path.exists(path):
            return create_file(path, content, encoding)
            
        with open(path, 'a', encoding=encoding) as f:
            f.write(content)
            
        return {
            'success': True,
            'operation': 'append_file',
            'path': path,
            'appended_size': len(content),
            'modified': datetime.now().isoformat()
        }
    except Exception as e:
        raise FileSystemError(f"Error appending to file: {str(e)}")

def delete_file(path: str, secure_delete: bool = False) -> Dict:
    """Delete a file
    
    :param path: Path to the file to delete
    :param secure_delete: Perform secure deletion by overwriting with zeros first
    :return: Dictionary with operation result
    """
    try:
        if not is_safe_path(path):
            raise FileSystemError("Path is not safe for deletion")
            
        if not os.path.exists(path):
            raise FileSystemError(f"File not found: {path}")
            
        if not os.path.isfile(path):
            raise FileSystemError(f"Not a file: {path}")
            
        if secure_delete:
            # Overwrite file with zeros before deletion
            file_size = os.path.getsize(path)
            with open(path, 'wb') as f:
                f.write(b'\x00' * min(file_size, 1024 * 1024))  # Max 1MB at a time to avoid memory issues
                f.flush()
                os.fsync(f.fileno())
                
        os.remove(path)
        
        return {
            'success': True,
            'operation': 'delete_file',
            'path': path,
            'deleted_time': datetime.now().isoformat(),
            'secure_delete': secure_delete
        }
    except Exception as e:
        raise FileSystemError(f"Error deleting file: {str(e)}")

def delete_directory(path: str, recursive: bool = False) -> Dict:
    """Delete a directory
    
    :param path: Path to the directory to delete
    :param recursive: Whether to delete all contents recursively
    :return: Dictionary with operation result
    """
    try:
        if not is_safe_path(path):
            raise FileSystemError("Path is not safe for deletion")
            
        if not os.path.exists(path):
            raise FileSystemError(f"Directory not found: {path}")
            
        if not os.path.isdir(path):
            raise FileSystemError(f"Not a directory: {path}")
            
        if recursive:
            shutil.rmtree(path)
        else:
            os.rmdir(path)  # Will only work if directory is empty
            
        return {
            'success': True,
            'operation': 'delete_directory',
            'path': path,
            'recursive': recursive,
            'deleted_time': datetime.now().isoformat()
        }
    except OSError as e:
        if e.errno == 39:  # Directory not empty
            raise FileSystemError(f"Directory not empty. Use recursive=True to delete all contents: {path}")
        else:
            raise FileSystemError(f"Error deleting directory: {str(e)}")
    except Exception as e:
        raise FileSystemError(f"Error deleting directory: {str(e)}")

def copy_file(source: str, destination: str, overwrite: bool = False) -> Dict:
    """Copy a file to a new location
    
    :param source: Source file path
    :param destination: Destination file path
    :param overwrite: Whether to overwrite existing destination
    :return: Dictionary with operation result
    """
    try:
        if not is_safe_path(source) or not is_safe_path(destination):
            raise FileSystemError("Path is not safe for copy operation")
            
        if not os.path.exists(source):
            raise FileSystemError(f"Source file not found: {source}")
            
        if not os.path.isfile(source):
            raise FileSystemError(f"Source is not a file: {source}")
            
        if os.path.exists(destination) and not overwrite:
            raise FileSystemError(f"Destination already exists: {destination}")
            
        # Create destination directory if needed
        dest_dir = os.path.dirname(destination)
        if dest_dir and not os.path.exists(dest_dir):
            os.makedirs(dest_dir, exist_ok=True)
            
        shutil.copy2(source, destination)  # copy2 preserves metadata
        
        return {
            'success': True,
            'operation': 'copy_file',
            'source': source,
            'destination': destination,
            'size': os.path.getsize(destination),
            'copied_time': datetime.now().isoformat()
        }
    except Exception as e:
        raise FileSystemError(f"Error copying file: {str(e)}")

def move_file(source: str, destination: str, overwrite: bool = False) -> Dict:
    """Move a file to a new location
    
    :param source: Source file path
    :param destination: Destination file path
    :param overwrite: Whether to overwrite existing destination
    :return: Dictionary with operation result
    """
    try:
        if not is_safe_path(source) or not is_safe_path(destination):
            raise FileSystemError("Path is not safe for move operation")
            
        if not os.path.exists(source):
            raise FileSystemError(f"Source file not found: {source}")
            
        if not os.path.isfile(source):
            raise FileSystemError(f"Source is not a file: {source}")
            
        if os.path.exists(destination) and not overwrite:
            raise FileSystemError(f"Destination already exists: {destination}")
            
        # Create destination directory if needed
        dest_dir = os.path.dirname(destination)
        if dest_dir and not os.path.exists(dest_dir):
            os.makedirs(dest_dir, exist_ok=True)
            
        # If destination exists and overwrite is True, remove it first
        if os.path.exists(destination) and overwrite:
            os.remove(destination)
            
        shutil.move(source, destination)
        
        return {
            'success': True,
            'operation': 'move_file',
            'source': source,
            'destination': destination,
            'size': os.path.getsize(destination),
            'moved_time': datetime.now().isoformat()
        }
    except Exception as e:
        raise FileSystemError(f"Error moving file: {str(e)}")

def copy_directory(source: str, destination: str, overwrite: bool = False) -> Dict:
    """Copy a directory to a new location
    
    :param source: Source directory path
    :param destination: Destination directory path
    :param overwrite: Whether to overwrite existing destination
    :return: Dictionary with operation result
    """
    try:
        if not is_safe_path(source) or not is_safe_path(destination):
            raise FileSystemError("Path is not safe for copy operation")
            
        if not os.path.exists(source):
            raise FileSystemError(f"Source directory not found: {source}")
            
        if not os.path.isdir(source):
            raise FileSystemError(f"Source is not a directory: {source}")
            
        if os.path.exists(destination):
            if not overwrite:
                raise FileSystemError(f"Destination already exists: {destination}")
            if os.path.isdir(destination):
                shutil.rmtree(destination)
            else:
                os.remove(destination)
                
        shutil.copytree(source, destination)
        
        # Count copied files and total size
        file_count = 0
        dir_count = 0
        total_size = 0
        
        for root, dirs, files in os.walk(destination):
            dir_count += len(dirs)
            file_count += len(files)
            for file in files:
                file_path = os.path.join(root, file)
                total_size += os.path.getsize(file_path)
                
        return {
            'success': True,
            'operation': 'copy_directory',
            'source': source,
            'destination': destination,
            'file_count': file_count,
            'directory_count': dir_count,
            'total_size': total_size,
            'total_size_human': f"{total_size / 1024:.2f} KB" if total_size < 1024 * 1024 else f"{total_size / (1024 * 1024):.2f} MB",
            'copied_time': datetime.now().isoformat()
        }
    except Exception as e:
        raise FileSystemError(f"Error copying directory: {str(e)}")

def move_directory(source: str, destination: str, overwrite: bool = False) -> Dict:
    """Move a directory to a new location
    
    :param source: Source directory path
    :param destination: Destination directory path
    :param overwrite: Whether to overwrite existing destination
    :return: Dictionary with operation result
    """
    try:
        if not is_safe_path(source) or not is_safe_path(destination):
            raise FileSystemError("Path is not safe for move operation")
            
        if not os.path.exists(source):
            raise FileSystemError(f"Source directory not found: {source}")
            
        if not os.path.isdir(source):
            raise FileSystemError(f"Source is not a directory: {source}")
            
        if os.path.exists(destination):
            if not overwrite:
                raise FileSystemError(f"Destination already exists: {destination}")
            if os.path.isdir(destination):
                shutil.rmtree(destination)
            else:
                os.remove(destination)
                
        # Get stats before moving for reporting
        file_count = 0
        dir_count = 0
        total_size = 0
        
        for root, dirs, files in os.walk(source):
            dir_count += len(dirs)
            file_count += len(files)
            for file in files:
                file_path = os.path.join(root, file)
                total_size += os.path.getsize(file_path)
                
        # Move the directory
        shutil.move(source, destination)
        
        return {
            'success': True,
            'operation': 'move_directory',
            'source': source,
            'destination': destination,
            'file_count': file_count,
            'directory_count': dir_count,
            'total_size': total_size,
            'total_size_human': f"{total_size / 1024:.2f} KB" if total_size < 1024 * 1024 else f"{total_size / (1024 * 1024):.2f} MB",
            'moved_time': datetime.now().isoformat()
        }
    except Exception as e:
        raise FileSystemError(f"Error moving directory: {str(e)}")

def set_file_permissions(path: str, mode: int) -> Dict:
    """Set file permissions
    
    :param path: Path to the file or directory
    :param mode: Permission mode in octal (e.g., 0o755)
    :return: Dictionary with operation result
    """
    try:
        if not is_safe_path(path):
            raise FileSystemError("Path is not safe for permission modification")
            
        if not os.path.exists(path):
            raise FileSystemError(f"Path not found: {path}")
            
        os.chmod(path, mode)
        
        new_permissions = get_permissions_string(path)
        
        return {
            'success': True,
            'operation': 'set_permissions',
            'path': path,
            'mode': mode,
            'permissions': new_permissions,
            'modified': datetime.now().isoformat()
        }
    except Exception as e:
        raise FileSystemError(f"Error setting file permissions: {str(e)}")

def set_file_modification_time(path: str, mtime: float = None, atime: float = None) -> Dict:
    """Set file modification and access times
    
    :param path: Path to the file
    :param mtime: Modification time (seconds since epoch)
    :param atime: Access time (seconds since epoch)
    :return: Dictionary with operation result
    """
    try:
        if not is_safe_path(path):
            raise FileSystemError("Path is not safe for timestamp modification")
            
        if not os.path.exists(path):
            raise FileSystemError(f"Path not found: {path}")
            
        # Use current time for any unspecified timestamp
        if mtime is None:
            mtime = time.time()
        if atime is None:
            atime = mtime
            
        os.utime(path, (atime, mtime))
        
        return {
            'success': True,
            'operation': 'set_file_times',
            'path': path,
            'atime': datetime.fromtimestamp(atime).isoformat(),
            'mtime': datetime.fromtimestamp(mtime).isoformat(),
            'modified': datetime.now().isoformat()
        }
    except Exception as e:
        raise FileSystemError(f"Error setting file times: {str(e)}")

def compress_files(source_paths: List[str], destination: str, archive_format: str = 'zip') -> Dict:
    """Compress files into an archive
    
    :param source_paths: List of paths to compress
    :param destination: Destination archive path
    :param archive_format: Archive format ('zip', 'tar', 'gztar', 'bztar', 'xztar')
    :return: Dictionary with operation result
    """
    try:
        valid_formats = ['zip', 'tar', 'gztar', 'bztar', 'xztar']
        if archive_format not in valid_formats:
            raise FileSystemError(f"Invalid archive format. Use one of: {', '.join(valid_formats)}")
            
        for path in source_paths:
            if not is_safe_path(path):
                raise FileSystemError(f"Path is not safe for archiving: {path}")
                
        # Create base directory for output
        base_dir = os.path.dirname(destination)
        if base_dir and not os.path.exists(base_dir):
            os.makedirs(base_dir, exist_ok=True)
            
        # Handle single directory case
        if len(source_paths) == 1 and os.path.isdir(source_paths[0]):
            # Archive directory contents
            base_name = os.path.splitext(destination)[0]
            root_dir = source_paths[0]
            
            base_path = os.path.normpath(os.path.join(root_dir, '..'))
            base_dir_name = os.path.basename(root_dir)
            
            archive_path = shutil.make_archive(
                base_name=base_name,
                format=archive_format,
                root_dir=base_path,
                base_dir=base_dir_name
            )
        else:
            # Create a temporary directory to gather all files
            with tempfile.TemporaryDirectory() as temp_dir:
                for src_path in source_paths:
                    dst_name = os.path.basename(src_path)
                    dst_path = os.path.join(temp_dir, dst_name)
                    
                    if os.path.isdir(src_path):
                        shutil.copytree(src_path, dst_path)
                    else:
                        shutil.copy2(src_path, dst_path)
                        
                # Archive the temporary directory
                base_name = os.path.splitext(destination)[0]
                archive_path = shutil.make_archive(
                    base_name=base_name,
                    format=archive_format,
                    root_dir=os.path.dirname(temp_dir),
                    base_dir=os.path.basename(temp_dir)
                )
                
        return {
            'success': True,
            'operation': 'compress_files',
            'source_paths': source_paths,
            'destination': archive_path,
            'format': archive_format,
            'size': os.path.getsize(archive_path),
            'size_human': f"{os.path.getsize(archive_path) / (1024 * 1024):.2f} MB",
            'created': datetime.now().isoformat()
        }
    except Exception as e:
        raise FileSystemError(f"Error compressing files: {str(e)}")

def extract_archive(archive_path: str, destination: str, password: str = None) -> Dict:
    """Extract an archive
    
    :param archive_path: Path to the archive
    :param destination: Destination directory
    :param password: Optional password for encrypted archives
    :return: Dictionary with operation result
    """
    try:
        if not is_safe_path(archive_path) or not is_safe_path(destination):
            raise FileSystemError("Path is not safe for extraction")
            
        if not os.path.exists(archive_path):
            raise FileSystemError(f"Archive not found: {archive_path}")
            
        # Create destination directory if it doesn't exist
        if not os.path.exists(destination):
            os.makedirs(destination, exist_ok=True)
            
        # Check archive format
        archive_ext = os.path.splitext(archive_path)[1].lower()
        
        # Directly use shutil for standard formats
        if archive_ext in ['.zip', '.tar', '.gz', '.bz2', '.xz']:
            shutil.unpack_archive(archive_path, destination)
        # Use specialized libraries for other formats
        elif archive_ext == '.rar':
            try:
                import rarfile
                with rarfile.RarFile(archive_path) as rf:
                    if password:
                        rf.setpassword(password)
                    rf.extractall(destination)
            except ImportError:
                raise FileSystemError("RAR extraction requires the 'rarfile' library.")
        elif archive_ext == '.7z':
            try:
                import py7zr
                with py7zr.SevenZipFile(archive_path, mode='r', password=password) as z:
                    z.extractall(destination)
            except ImportError:
                raise FileSystemError("7z extraction requires the 'py7zr' library.")
        else:
            raise FileSystemError(f"Unsupported archive format: {archive_ext}")
            
        # Count extracted files and total size
        file_count = 0
        dir_count = 0
        total_size = 0
        
        for root, dirs, files in os.walk(destination):
            dir_count += len(dirs)
            file_count += len(files)
            total_size += sum(os.path.getsize(os.path.join(root, file)) for file in files)
            
        return {
            'success': True,
            'operation': 'extract_archive',
            'archive_path': archive_path,
            'destination': destination,
            'file_count': file_count,
            'directory_count': dir_count,
            'total_size': total_size,
            'total_size_human': f"{total_size / (1024 * 1024):.2f} MB",
            'extracted': datetime.now().isoformat()
        }
    except Exception as e:
        raise FileSystemError(f"Error extracting archive: {str(e)}")

def open_file_or_application(path: str, arguments: Optional[List[str]] = None, timeout: int = 30) -> Dict:
    """Open a file or application with the system's default program
    
    :param path: Path to the file or application to open
    :param arguments: Optional list of arguments to pass to the application
    :param timeout: Timeout in seconds for command execution (only used for error checking)
    :return: Dictionary with structured operation result
    """
    try:
        normalized_path = os.path.normpath(path)
        
        if not is_safe_path(normalized_path):
            raise FileSystemError("Path is not safe for opening")
            
        if not os.path.exists(normalized_path):
            raise FileSystemError(f"Path not found: {normalized_path}")
            
        result = {
            'success': False,
            'operation': 'open',
            'path': normalized_path,
            'original_path': path,
            'time': datetime.now().isoformat(),
            'platform': platform.system(),
            'file_type': os.path.splitext(normalized_path)[1].lower(),
        }
        
        # Handle based on platform
        if platform.system() == 'Windows':
            # Handle Windows shortcut files (.lnk)
            if normalized_path.lower().endswith('.lnk'):
                try:
                    import win32com.client
                    shell = win32com.client.Dispatch("WScript.Shell")
                    shortcut = shell.CreateShortCut(normalized_path)
                    target_path = shortcut.Targetpath
                    working_dir = shortcut.WorkingDirectory
                    
                    result['shortcut_target'] = target_path
                    result['shortcut_working_dir'] = working_dir
                    
                    # If target exists, use it instead
                    if os.path.exists(target_path):
                        normalized_path = target_path
                        result['resolved_path'] = normalized_path
                    else:
                        result['warning'] = f"Shortcut target not found: {target_path}"
                except ImportError:
                    result['warning'] = "win32com module not available, cannot resolve shortcut target"
                except Exception as e:
                    result['warning'] = f"Error resolving shortcut: {str(e)}"
            
            # On Windows, use PowerShell to handle path properly
            try:
                # Properly quote the path to handle spaces and special characters
                quoted_path = f'"{normalized_path}"'
                
                if arguments:
                    # If there are arguments, construct proper PowerShell command
                    args_str = ','.join([f'"{arg}"' for arg in arguments])
                    cmd = ['powershell', '-Command', f'Start-Process -FilePath {quoted_path} -ArgumentList {args_str}']
                else:
                    # Simple case with no arguments
                    cmd = ['powershell', '-Command', f'Start-Process {quoted_path}']
                
                proc = subprocess.Popen(
                    cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE
                )
                
                # Wait for a short time to check for immediate errors
                try:
                    stdout, stderr = proc.communicate(timeout=timeout)
                    result['return_code'] = proc.returncode
                    
                    if stderr:
                        result['stderr'] = stderr.decode('utf-8', errors='replace').strip()
                    
                    result['success'] = proc.returncode == 0
                except subprocess.TimeoutExpired:
                    # This is expected for applications that stay open
                    proc.kill()
                    result['success'] = True
                    result['warning'] = "Process started but didn't complete within timeout (likely running as expected)"
            
            except Exception as e:
                result['error'] = str(e)
                result['error_type'] = type(e).__name__
                return result
                
        elif platform.system() == 'Darwin':  # macOS
            try:
                cmd = ['open']
                if arguments:
                    cmd.extend(['-a', normalized_path, '--args'])
                    cmd.extend(arguments)
                else:
                    cmd.append(normalized_path)
                
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE
                )
                
                # Wait for a short time to check for immediate errors
                try:
                    stdout, stderr = proc.communicate(timeout=timeout)
                    result['return_code'] = proc.returncode
                    
                    if stderr:
                        result['stderr'] = stderr.decode('utf-8', errors='replace').strip()
                    
                    result['success'] = proc.returncode == 0
                except subprocess.TimeoutExpired:
                    # This is expected for applications that stay open
                    proc.kill()
                    result['success'] = True
                    result['warning'] = "Process started but didn't complete within timeout (likely running as expected)"
            
            except Exception as e:
                result['error'] = str(e)
                result['error_type'] = type(e).__name__
                return result
                
        else:  # Linux and others
            try:
                cmd = ['xdg-open', normalized_path]
                
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE
                )
                
                # Wait for a short time to check for immediate errors
                try:
                    stdout, stderr = proc.communicate(timeout=timeout)
                    result['return_code'] = proc.returncode
                    
                    if stderr:
                        result['stderr'] = stderr.decode('utf-8', errors='replace').strip()
                    
                    result['success'] = proc.returncode == 0
                except subprocess.TimeoutExpired:
                    # This is expected for applications that stay open
                    proc.kill()
                    result['success'] = True
                    result['warning'] = "Process started but didn't complete within timeout (likely running as expected)"
            
            except Exception as e:
                result['error'] = str(e)
                result['error_type'] = type(e).__name__
                return result
                
        # Add file information if successful
        if result['success'] and os.path.isfile(normalized_path):
            result['file_info'] = {
                'size': os.path.getsize(normalized_path),
                'size_human': f"{os.path.getsize(normalized_path) / 1024:.2f} KB" if os.path.getsize(normalized_path) < 1024 * 1024 else f"{os.path.getsize(normalized_path) / (1024 * 1024):.2f} MB",
                'type': mimetypes.guess_type(normalized_path)[0] or 'application/octet-stream',
                'extension': os.path.splitext(normalized_path)[1].lower(),
                'modified': datetime.fromtimestamp(os.path.getmtime(normalized_path)).isoformat()
            }
            
        return result
    except FileSystemError as e:
        # Return structured error for FileSystemError
        return {
            'success': False,
            'operation': 'open',
            'path': path,
            'error': str(e),
            'error_type': 'FileSystemError',
            'time': datetime.now().isoformat()
        }
    except Exception as e:
        # Return structured error for other exceptions
        return {
            'success': False,
            'operation': 'open',
            'path': path,
            'error': str(e),
            'error_type': type(e).__name__,
            'time': datetime.now().isoformat()
        }

def run_command(command: str, timeout: int = 60, shell: bool = True, cwd: Optional[str] = None, 
                env: Optional[Dict[str, str]] = None, strict_security: bool = True) -> Dict:
    """Run a command in the system shell with improved error handling and security
    
    :param command: Command to run
    :param timeout: Timeout in seconds for command execution
    :param shell: Whether to use shell execution
    :param cwd: Current working directory for the command
    :param env: Environment variables for the command
    :param strict_security: Whether to apply strict security checks
    :return: Dictionary with structured operation result
    """
    start_time = time.time()
    result = {
        'success': False,
        'operation': 'run_command',
        'command': command,
        'time_start': datetime.now().isoformat(),
        'platform': platform.system(),
        'pid': None,
        'timeout_specified': timeout,
        'shell': shell,
    }
    
    if cwd:
        result['cwd'] = cwd
    
    try:
        # Security checks
        if strict_security:
            # Define dangerous patterns
            dangerous_commands = [
                # Destructive file operations
                r'rm\s+-rf\s+[/~]', r'deltree\s+/[a-z]', r'format\s+[a-z]:',
                # Fork bombs and resource exhaustion
                r':\(\){:', r'while\s*true', r'for\s*\(\(\s*;;\s*\)\)',
                # Disk operations
                r'>\s*/dev/[sh]d[a-z]', r'dd\s+if=/dev/zero\s+of=',
                # Network operations that could download malicious content
                r'wget\s+http', r'curl\s+http', r'powershell\s+-\w+\s+iex',
                # Shell redirections that hide output
                r'>\s*/dev/null\s+2>&1', r'>\s+NUL\s+2>&1',
                # System modification commands
                r'chmod\s+-R\s+777\s+/', r'chown\s+-R\s+\w+\s+/',
                # Shutdown/reboot commands
                r'shutdown', r'reboot', r'halt', r'init\s+[06]'
            ]
            
            # Check command against dangerous patterns
            for pattern in dangerous_commands:
                if re.search(pattern, command, re.IGNORECASE):
                    result['error'] = f"Potentially dangerous command pattern detected: {pattern}"
                    result['error_type'] = 'SecurityError'
                    return result
                    
        # Choose execution method based on platform
        if platform.system() == 'Windows':
            # Windows-specific execution wrapper
            if not shell:
                # Direct execution
                process = subprocess.Popen(
                    command if shell else command.split(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    shell=shell,
                    cwd=cwd,
                    env=env,
                    text=True,
                    errors='replace'
                )
            else:
                # Use PowerShell for better command execution on Windows
                process = subprocess.Popen(
                    ['powershell', '-Command', command],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    shell=False,  # Don't use shell for the powershell command itself
                    cwd=cwd,
                    env=env,
                    text=True,
                    errors='replace'
                )
        else:
            # Unix-based execution
            process = subprocess.Popen(
                command if shell else command.split(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=shell,
                cwd=cwd,
                env=env,
                text=True,
                errors='replace'
            )
            
        # Record process ID
        result['pid'] = process.pid
        
        try:
            # Wait for command completion with timeout
            stdout, stderr = process.communicate(timeout=timeout)
            
            # Format output for clean parsing
            result.update({
                'return_code': process.returncode,
                'success': process.returncode == 0,
                'stdout': stdout.strip() if stdout else '',
                'stderr': stderr.strip() if stderr else '',
                'stdout_lines': stdout.strip().split('\n') if stdout else [],
                'stderr_lines': stderr.strip().split('\n') if stderr else [],
                'execution_time_seconds': round(time.time() - start_time, 3),
                'time_end': datetime.now().isoformat(),
                'timed_out': False
            })
            
            # Add error details if command failed
            if process.returncode != 0:
                result['error'] = stderr.strip() if stderr else "Command failed with no error output"
                result['error_type'] = 'CommandExecutionError'
                
        except subprocess.TimeoutExpired:
            # Handle timeout
            process.kill()
            
            result.update({
                'success': False,
                'timed_out': True,
                'error': f"Command execution timed out after {timeout} seconds",
                'error_type': 'TimeoutError',
                'execution_time_seconds': round(time.time() - start_time, 3),
                'time_end': datetime.now().isoformat()
            })
            
            # Try to collect any output that was generated before timeout
            try:
                stdout, stderr = process.communicate(timeout=0.5)
                result['stdout'] = stdout.strip() if stdout else ''
                result['stderr'] = stderr.strip() if stderr else ''
                result['stdout_lines'] = stdout.strip().split('\n') if stdout else []
                result['stderr_lines'] = stderr.strip().split('\n') if stderr else []
                result['partial_output'] = True
            except:
                result['partial_output'] = False
                
    except FileNotFoundError:
        result.update({
            'success': False,
            'error': f"Command not found: {command.split()[0] if not shell else command}",
            'error_type': 'FileNotFoundError',
            'execution_time_seconds': round(time.time() - start_time, 3),
            'time_end': datetime.now().isoformat()
        })
        
    except PermissionError:
        result.update({
            'success': False,
            'error': f"Permission denied when executing command",
            'error_type': 'PermissionError',
            'execution_time_seconds': round(time.time() - start_time, 3),
            'time_end': datetime.now().isoformat()
        })
        
    except Exception as e:
        result.update({
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__,
            'execution_time_seconds': round(time.time() - start_time, 3),
            'time_end': datetime.now().isoformat()
        })
        
    # AI parsability improvements - detect and convert output formats
    try:
        if result.get('success', False) and result.get('stdout', ''):
            stdout = result.get('stdout', '')
            
            # Try to detect JSON output
            if stdout.strip().startswith('{') and stdout.strip().endswith('}'):
                try:
                    json_data = json.loads(stdout)
                    result['parsed_json'] = json_data
                    result['detected_format'] = 'json'
                except:
                    pass
                    
            # Try to detect table/csv output
            elif '\n' in stdout and ',' in stdout and stdout.count(',') > stdout.count('\n'):
                lines = stdout.strip().split('\n')
                if all(line.count(',') == lines[0].count(',') for line in lines):
                    try:
                        import csv
                        from io import StringIO
                        
                        # Parse as CSV
                        csv_data = []
                        csv_reader = csv.reader(StringIO(stdout))
                        for row in csv_reader:
                            csv_data.append(row)
                            
                        if len(csv_data) > 1:  # Must have header and at least one data row
                            result['parsed_csv'] = {
                                'headers': csv_data[0],
                                'rows': csv_data[1:]
                            }
                            result['detected_format'] = 'csv'
                    except:
                        pass
    except:
        # Don't fail if parsing enhancement fails
        pass
        
    return result

def execute(arguments: Dict) -> str:
    """Execute filesystem operations for searching, reading, and manipulating files
    
    :param operation: Operation to perform (search, read, read_chunk, info, list, list_tree, tree, user, drives,
                      path_info, create_dir, create_file, write, append, delete_file, delete_dir, 
                      copy_file, move_file, copy_dir, move_dir, permissions, find_duplicates,
                      checksum, checksums, compress, extract, set_times, metadata, open, run)
    :param path: Target path for the operation
    :param pattern: File pattern for search/list operations (e.g., "*.txt", "*.py")
    :param content: Text to search for within files or content to write
    :param content_query: Text to search for within files (alternative to content)
    :param max_depth: Maximum directory depth for search
    :param exclude_dirs: List of directory names to exclude from search
    :param include_hidden: Whether to include hidden files
    :param start_line: Starting line number for read operation (1-based, inclusive)
    :param end_line: Ending line number for read operation (1-based, inclusive)
    :param offset: Byte offset for read_chunk operation
    :param size: Chunk size for read_chunk operation
    :param destination: Destination path for copy/move/compress operations
    :param overwrite: Whether to overwrite existing files/directories
    :param recursive: Whether to delete directories recursively
    :param algorithm: Hash algorithm for checksum calculation
    :param algorithms: List of hash algorithms for multiple checksums
    :param min_size: Minimum file size for search/duplicate operations
    :param max_size: Maximum file size for search operations
    :param modified_after: Include files modified after this date (ISO format)
    :param modified_before: Include files modified before this date (ISO format)
    :param case_sensitive: Whether to perform case-sensitive content search
    :param file_extensions: List of file extensions to include in search
    :param regex_pattern: Whether the content_query is a regex pattern
    :param mode: Permission mode for set_permissions operation (octal int)
    :param mtime: Modification time for set_times operation (seconds since epoch)
    :param atime: Access time for set_times operation (seconds since epoch)
    :param archive_format: Archive format for compress operation
    :param source_paths: List of paths for compress operation
    :param password: Password for extract operation
    :param include_metadata: Whether to include extended metadata
    :param encoding: File encoding for write/append operations
    :param secure_delete: Whether to perform secure deletion
    :param arguments: Optional list of arguments to pass to the application when opening
    :param command: Command to run for run operation
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
        >>>
        >>> # Create and write to a file
        >>> result = execute({
        ...     "operation": "write",
        ...     "path": "C:\\Projects\\new_file.txt",
        ...     "content": "Hello, world!"
        ... })
        >>>
        >>> # Copy a file
        >>> result = execute({
        ...     "operation": "copy_file",
        ...     "source": "C:\\Projects\\example.py",
        ...     "destination": "C:\\Backup\\example.py"
        ... })
        >>>
        >>> # Open a file or application
        >>> result = execute({
        ...     "operation": "open",
        ...     "path": "C:\\Program Files\\Example\\app.exe"
        ... })
        >>>
        >>> # Run a command
        >>> result = execute({
        ...     "operation": "run",
        ...     "command": "dir"
        ... })
    """
    try:
        operation = arguments.get('operation', '').lower()
        
        # Operations that don't require a path
        if operation == 'user':
            user_info = get_user_info()
            return json.dumps({
                "operation": "user",
                "user_info": user_info
            }, indent=2)
            
        if operation == 'drives':
            drives_info = get_drive_info()
            return json.dumps({
                "operation": "drives",
                "drives": drives_info
            }, indent=2)
            
        # All other operations require a path
        path = arguments.get('path', '')
        
        # Path parsing doesn't modify files, so we can skip the safety check
        if operation == 'path_info':
            path_info = parse_path(path)
            return json.dumps({
                "operation": "path_info",
                "path_info": path_info
            }, indent=2)
            
        # Check path safety for all path-based operations
        if operation != 'path_info' and path and not is_safe_path(path):
            return json.dumps({"error": "Access to this path is not allowed for security reasons"}, indent=2)
            
        # Operations that read/search files
        if operation == 'search':
            pattern = arguments.get('pattern', '*')
            content = arguments.get('content')
            content_query = arguments.get('content_query', content)
            max_depth = int(arguments.get('max_depth', 10))
            exclude_dirs = arguments.get('exclude_dirs', [])
            include_hidden = bool(arguments.get('include_hidden', False))
            min_size = arguments.get('min_size')
            max_size = arguments.get('max_size')
            modified_after = arguments.get('modified_after')
            modified_before = arguments.get('modified_before')
            case_sensitive = bool(arguments.get('case_sensitive', False))
            file_extensions = arguments.get('file_extensions')
            regex_pattern = bool(arguments.get('regex_pattern', False))
            
            if min_size is not None:
                min_size = int(min_size)
            if max_size is not None:
                max_size = int(max_size)
                
            results = search_files(
                path,
                pattern=pattern,
                content_query=content_query,
                max_depth=max_depth,
                exclude_dirs=exclude_dirs,
                include_hidden=include_hidden,
                min_size=min_size,
                max_size=max_size,
                modified_after=modified_after,
                modified_before=modified_before,
                case_sensitive=case_sensitive,
                file_extensions=file_extensions,
                regex_pattern=regex_pattern
            )
            
            return json.dumps({
                "operation": "search",
                "path": path,
                "pattern": pattern,
                "content_query": content_query,
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
            
        elif operation == 'read_chunk':
            offset = int(arguments.get('offset', 0))
            size = int(arguments.get('size', 1024))
            
            chunk_result = read_file_chunk(path, offset, size)
            
            return json.dumps({
                "operation": "read_chunk",
                "path": path,
                "chunk": chunk_result
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
            
        elif operation == 'list_tree' or operation == 'tree':
            max_depth = int(arguments.get('max_depth', 3))
            include_hidden = bool(arguments.get('include_hidden', False))
            
            tree = list_directory_tree(path, max_depth, include_hidden)
            return json.dumps({
                "operation": "tree",
                "path": path,
                "max_depth": max_depth,
                "tree": tree
            }, indent=2)
            
        elif operation == 'info':
            include_preview = bool(arguments.get('include_preview', False))
            include_metadata = bool(arguments.get('include_metadata', False))
            
            file_info = get_file_info(path, include_preview, include_metadata)
            return json.dumps({
                "operation": "info",
                "file_info": file_info
            }, indent=2)
            
        elif operation == 'metadata':
            metadata = get_metadata(path)
            return json.dumps({
                "operation": "metadata",
                "path": path,
                "metadata": metadata
            }, indent=2)
            
        elif operation == 'checksum':
            algorithm = arguments.get('algorithm', 'sha256')
            
            if algorithm not in HASH_ALGORITHMS:
                return json.dumps({
                    "error": f"Unsupported hash algorithm: {algorithm}. Use one of: {', '.join(HASH_ALGORITHMS)}"
                }, indent=2)
                
            checksum = calculate_checksum(path, algorithm)
            return json.dumps({
                "operation": "checksum",
                "path": path,
                "algorithm": algorithm,
                "checksum": checksum
            }, indent=2)
            
        elif operation == 'checksums':
            algorithms = arguments.get('algorithms', ['md5', 'sha1', 'sha256'])
            
            checksums = calculate_multiple_checksums(path, algorithms)
            return json.dumps({
                "operation": "checksums",
                "path": path,
                "checksums": checksums
            }, indent=2)
            
        elif operation == 'find_duplicates':
            hash_algorithm = arguments.get('algorithm', 'sha256')
            exclude_dirs = arguments.get('exclude_dirs', [])
            min_size = int(arguments.get('min_size', 1024))
            include_hidden = bool(arguments.get('include_hidden', False))
            
            duplicates = find_duplicate_files(
                path,
                hash_algorithm=hash_algorithm,
                exclude_dirs=exclude_dirs,
                min_size=min_size,
                include_hidden=include_hidden
            )
            
            return json.dumps({
                "operation": "find_duplicates",
                "path": path,
                "algorithm": hash_algorithm,
                "duplicates": duplicates,
                "duplicate_groups": len(duplicates),
                "duplicate_files": sum(len(files) for files in duplicates.values())
            }, indent=2)
            
        # Operations that create/modify files
        elif operation == 'create_dir' or operation == 'mkdir':
            mode = int(arguments.get('mode', 0o755))
            exist_ok = bool(arguments.get('exist_ok', False))
            
            result = create_directory(path, mode, exist_ok)
            return json.dumps({
                "operation": "create_directory",
                "result": result
            }, indent=2)
            
        elif operation == 'create_file':
            content = arguments.get('content', '')
            encoding = arguments.get('encoding', DEFAULT_ENCODING)
            
            result = create_file(path, content, encoding)
            return json.dumps({
                "operation": "create_file",
                "result": result
            }, indent=2)
            
        elif operation == 'write':
            content = arguments.get('content', '')
            encoding = arguments.get('encoding', DEFAULT_ENCODING)
            
            result = write_file(path, content, encoding)
            return json.dumps({
                "operation": "write",
                "result": result
            }, indent=2)
            
        elif operation == 'append':
            content = arguments.get('content', '')
            encoding = arguments.get('encoding', DEFAULT_ENCODING)
            
            result = append_file(path, content, encoding)
            return json.dumps({
                "operation": "append",
                "result": result
            }, indent=2)
            
        elif operation == 'delete_file' or operation == 'remove':
            secure_delete = bool(arguments.get('secure_delete', False))
            
            result = delete_file(path, secure_delete)
            return json.dumps({
                "operation": "delete_file",
                "result": result
            }, indent=2)
            
        elif operation == 'delete_dir' or operation == 'rmdir':
            recursive = bool(arguments.get('recursive', False))
            
            result = delete_directory(path, recursive)
            return json.dumps({
                "operation": "delete_directory",
                "result": result
            }, indent=2)
            
        elif operation == 'copy_file' or operation == 'cp':
            destination = arguments.get('destination')
            if not destination:
                return json.dumps({"error": "Destination path is required"}, indent=2)
                
            overwrite = bool(arguments.get('overwrite', False))
            
            result = copy_file(path, destination, overwrite)
            return json.dumps({
                "operation": "copy_file",
                "result": result
            }, indent=2)
            
        elif operation == 'move_file' or operation == 'mv':
            destination = arguments.get('destination')
            if not destination:
                return json.dumps({"error": "Destination path is required"}, indent=2)
                
            overwrite = bool(arguments.get('overwrite', False))
            
            result = move_file(path, destination, overwrite)
            return json.dumps({
                "operation": "move_file",
                "result": result
            }, indent=2)
            
        elif operation == 'copy_dir' or operation == 'cp_dir':
            destination = arguments.get('destination')
            if not destination:
                return json.dumps({"error": "Destination path is required"}, indent=2)
                
            overwrite = bool(arguments.get('overwrite', False))
            
            result = copy_directory(path, destination, overwrite)
            return json.dumps({
                "operation": "copy_directory",
                "result": result
            }, indent=2)
            
        elif operation == 'move_dir' or operation == 'mv_dir':
            destination = arguments.get('destination')
            if not destination:
                return json.dumps({"error": "Destination path is required"}, indent=2)
                
            overwrite = bool(arguments.get('overwrite', False))
            
            result = move_directory(path, destination, overwrite)
            return json.dumps({
                "operation": "move_directory",
                "result": result
            }, indent=2)
            
        elif operation == 'permissions' or operation == 'chmod':
            mode = int(arguments.get('mode', 0))
            if mode == 0:
                return json.dumps({"error": "Permission mode is required"}, indent=2)
                
            result = set_file_permissions(path, mode)
            return json.dumps({
                "operation": "permissions",
                "result": result
            }, indent=2)
            
        elif operation == 'set_times' or operation == 'touch':
            mtime = arguments.get('mtime')
            atime = arguments.get('atime')
            
            if mtime is not None:
                mtime = float(mtime)
            if atime is not None:
                atime = float(atime)
                
            result = set_file_modification_time(path, mtime, atime)
            return json.dumps({
                "operation": "set_times",
                "result": result
            }, indent=2)
            
        elif operation == 'compress':
            source_paths = arguments.get('source_paths', [path])
            destination = arguments.get('destination')
            archive_format = arguments.get('archive_format', 'zip')
            
            if not destination:
                return json.dumps({"error": "Destination path is required"}, indent=2)
                
            result = compress_files(source_paths, destination, archive_format)
            return json.dumps({
                "operation": "compress",
                "result": result
            }, indent=2)
            
        elif operation == 'extract':
            destination = arguments.get('destination')
            password = arguments.get('password')
            
            if not destination:
                return json.dumps({"error": "Destination path is required"}, indent=2)
                
            result = extract_archive(path, destination, password)
            return json.dumps({
                "operation": "extract",
                "result": result
            }, indent=2)
            
        elif operation == 'run':
            command = arguments.get('command')
            if not command:
                return json.dumps({"error": "Command is required"}, indent=2)
                
            result = run_command(command)
            return json.dumps({
                "operation": "run_command",
                "result": result
            }, indent=2)
            
        elif operation == 'open':
            args = arguments.get('arguments', [])
            result = open_file_or_application(path, args)
            return json.dumps({
                "operation": "open",
                "result": result
            }, indent=2)
            
        else:
            valid_operations = [
                "search", "read", "read_chunk", "info", "list", "list_tree", "tree", "user", "drives",
                "path_info", "create_dir", "mkdir", "create_file", "write", "append", "delete_file",
                "remove", "delete_dir", "rmdir", "copy_file", "cp", "move_file", "mv", "copy_dir",
                "cp_dir", "move_dir", "mv_dir", "permissions", "chmod", "find_duplicates",
                "checksum", "checksums", "compress", "extract", "set_times", "touch", "metadata",
                "open", "run"
            ]
            return json.dumps({
                "error": f"Invalid operation: {operation}",
                "valid_operations": valid_operations
            }, indent=2)
            
    except Exception as e:
        return json.dumps({
            "error": str(e),
            "type": type(e).__name__
        }, indent=2) 