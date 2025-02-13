import platform
import psutil
import json
from typing import Dict, Union, List, Optional
from functools import lru_cache
import os
from datetime import datetime

# Cache duration for CPU intensive operations (in seconds)
CACHE_DURATION = 2

class SystemToolError(Exception):
    """Custom exception for system tool errors"""
    pass

@lru_cache(maxsize=1)
def get_cached_cpu_percent(percpu: bool = False) -> Union[float, List[float]]:
    """Cache CPU percentage calculations to avoid excessive polling"""
    return psutil.cpu_percent(interval=1, percpu=percpu)

def validate_info_type(info_type: str) -> None:
    """Validate the requested information type"""
    valid_types = {'os', 'memory', 'disk', 'cpu', 'all'}
    if info_type not in valid_types:
        raise SystemToolError(f"Invalid info_type. Must be one of: {', '.join(valid_types)}")

def get_disk_info(path: Optional[str] = None) -> Dict:
    """Get disk information for specified path or all mounted drives"""
    try:
        if path:
            if not os.path.exists(path):
                raise SystemToolError(f"Path does not exist: {path}")
            disk = psutil.disk_usage(path)
            return {
                'total': f"{disk.total / (1024**3):.2f} GB",
                'used': f"{disk.used / (1024**3):.2f} GB",
                'free': f"{disk.free / (1024**3):.2f} GB",
                'percent': f"{disk.percent}%"
            }
        else:
            # Get info for all mounted partitions
            partitions = {}
            for partition in psutil.disk_partitions():
                try:
                    disk = psutil.disk_usage(partition.mountpoint)
                    partitions[partition.mountpoint] = {
                        'device': partition.device,
                        'fstype': partition.fstype,
                        'total': f"{disk.total / (1024**3):.2f} GB",
                        'used': f"{disk.used / (1024**3):.2f} GB",
                        'free': f"{disk.free / (1024**3):.2f} GB",
                        'percent': f"{disk.percent}%"
                    }
                except PermissionError:
                    continue
            return partitions
    except Exception as e:
        raise SystemToolError(f"Error getting disk information: {str(e)}")

def execute(arguments: Dict[str, str]) -> str:
    """Execute system tool to get system information
    
    Args:
        arguments: Dictionary containing:
            type (str): Type of information to retrieve ('os', 'memory', 'disk', 'cpu', 'all')
            path (str, optional): Specific path for disk information
            
    Returns:
        str: JSON formatted system information
        
    Raises:
        SystemToolError: If there's an error getting system information
    """
    try:
        info_type = arguments.get('type', 'all').lower()
        path = arguments.get('path')
        
        # Validate input
        validate_info_type(info_type)
        
        if info_type == 'os':
            return json.dumps({
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'processor': platform.processor(),
                'architecture': platform.architecture(),
                'node': platform.node(),
                'python_version': platform.python_version(),
                'boot_time': datetime.fromtimestamp(psutil.boot_time()).strftime('%Y-%m-%d %H:%M:%S')
            }, indent=2)
            
        elif info_type == 'memory':
            mem = psutil.virtual_memory()
            swap = psutil.swap_memory()
            return json.dumps({
                'virtual_memory': {
                    'total': f"{mem.total / (1024**3):.2f} GB",
                    'available': f"{mem.available / (1024**3):.2f} GB",
                    'used': f"{mem.used / (1024**3):.2f} GB",
                    'free': f"{mem.free / (1024**3):.2f} GB",
                    'percent': f"{mem.percent}%",
                    'cached': f"{getattr(mem, 'cached', 0) / (1024**3):.2f} GB",
                    'buffers': f"{getattr(mem, 'buffers', 0) / (1024**3):.2f} GB"
                },
                'swap_memory': {
                    'total': f"{swap.total / (1024**3):.2f} GB",
                    'used': f"{swap.used / (1024**3):.2f} GB",
                    'free': f"{swap.free / (1024**3):.2f} GB",
                    'percent': f"{swap.percent}%"
                }
            }, indent=2)
            
        elif info_type == 'disk':
            return json.dumps(get_disk_info(path), indent=2)
            
        elif info_type == 'cpu':
            cpu_freq = psutil.cpu_freq()
            return json.dumps({
                'physical_cores': psutil.cpu_count(logical=False),
                'total_cores': psutil.cpu_count(logical=True),
                'cpu_freq': {
                    'current': f"{cpu_freq.current:.2f} MHz" if cpu_freq else "N/A",
                    'min': f"{cpu_freq.min:.2f} MHz" if cpu_freq and hasattr(cpu_freq, 'min') else "N/A",
                    'max': f"{cpu_freq.max:.2f} MHz" if cpu_freq and hasattr(cpu_freq, 'max') else "N/A"
                },
                'cpu_usage_per_core': [f"{percentage:.1f}%" for percentage in get_cached_cpu_percent(percpu=True)],
                'cpu_usage_total': f"{get_cached_cpu_percent():.1f}%",
                'cpu_stats': psutil.cpu_stats()._asdict(),
                'cpu_times_percent': psutil.cpu_times_percent()._asdict()
            }, indent=2)
            
        else:  # all
            return json.dumps({
                'os': {
                    'system': platform.system(),
                    'release': platform.release(),
                    'version': platform.version(),
                    'machine': platform.machine(),
                    'processor': platform.processor(),
                    'architecture': platform.architecture(),
                    'node': platform.node(),
                    'python_version': platform.python_version(),
                    'boot_time': datetime.fromtimestamp(psutil.boot_time()).strftime('%Y-%m-%d %H:%M:%S')
                },
                'memory': {
                    'virtual_memory': {
                        'total': f"{psutil.virtual_memory().total / (1024**3):.2f} GB",
                        'available': f"{psutil.virtual_memory().available / (1024**3):.2f} GB",
                        'used': f"{psutil.virtual_memory().used / (1024**3):.2f} GB",
                        'percent': f"{psutil.virtual_memory().percent}%"
                    },
                    'swap_memory': {
                        'total': f"{psutil.swap_memory().total / (1024**3):.2f} GB",
                        'used': f"{psutil.swap_memory().used / (1024**3):.2f} GB",
                        'free': f"{psutil.swap_memory().free / (1024**3):.2f} GB",
                        'percent': f"{psutil.swap_memory().percent}%"
                    }
                },
                'disk': get_disk_info(),
                'cpu': {
                    'physical_cores': psutil.cpu_count(logical=False),
                    'total_cores': psutil.cpu_count(logical=True),
                    'cpu_freq': {
                        'current': f"{psutil.cpu_freq().current:.2f} MHz" if psutil.cpu_freq() else "N/A",
                        'min': f"{psutil.cpu_freq().min:.2f} MHz" if psutil.cpu_freq() and hasattr(psutil.cpu_freq(), 'min') else "N/A",
                        'max': f"{psutil.cpu_freq().max:.2f} MHz" if psutil.cpu_freq() and hasattr(psutil.cpu_freq(), 'max') else "N/A"
                    },
                    'cpu_usage_per_core': [f"{percentage:.1f}%" for percentage in get_cached_cpu_percent(percpu=True)],
                    'cpu_usage_total': f"{get_cached_cpu_percent():.1f}%"
                }
            }, indent=2)
            
    except SystemToolError as e:
        return json.dumps({"error": str(e)}, indent=2)
    except Exception as e:
        return json.dumps({
            "error": "Unexpected error occurred",
            "details": str(e),
            "type": type(e).__name__
        }, indent=2) 