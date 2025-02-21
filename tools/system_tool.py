import platform
import psutil
import json
from typing import Dict, Union, List, Optional
from functools import lru_cache
import os
from datetime import datetime
import wmi
import GPUtil
from psutil._common import bytes2human
import socket
import urllib.request
import urllib.error
import ssl

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
    valid_types = {'os', 'memory', 'disk', 'cpu', 'gpu', 'network', 'processes', 'services', 'storage', 'all'}
    if info_type not in valid_types:
        raise SystemToolError(f"Invalid info_type. Must be one of: {', '.join(valid_types)}")

def format_datetime(timestamp: float) -> str:
    """Format timestamp into a readable 12-hour format with AM/PM"""
    return datetime.fromtimestamp(timestamp).strftime('%B %d, %Y %I:%M:%S %p')

def get_storage_details() -> Dict:
    """Get detailed storage information including physical drives"""
    try:
        w = wmi.WMI()
        storage_info = {}
        
        # Physical drive details
        physical_drives = []
        for disk in w.Win32_DiskDrive():
            drive_info = {
                'name': disk.Caption,
                'size': bytes2human(int(disk.Size)) if disk.Size else "Unknown",
                'interface_type': disk.InterfaceType,
                'media_type': disk.MediaType,
                'serial': disk.SerialNumber,
                'model': disk.Model,
                'status': disk.Status,
                'partitions': disk.Partitions
            }
            physical_drives.append(drive_info)
        
        # Logical drive details from psutil
        logical_drives = get_disk_info()
        
        storage_info['physical_drives'] = physical_drives
        storage_info['logical_drives'] = logical_drives
        
        return storage_info
    except Exception as e:
        raise SystemToolError(f"Error getting storage details: {str(e)}")

def get_gpu_info() -> Dict:
    """Get GPU information using GPUtil"""
    try:
        gpus = GPUtil.getGPUs()
        gpu_info = []
        for gpu in gpus:
            gpu_info.append({
                'id': gpu.id,
                'name': gpu.name,
                'load': f"{gpu.load*100:.1f}%",
                'memory': {
                    'total': f"{gpu.memoryTotal} MB",
                    'used': f"{gpu.memoryUsed} MB",
                    'free': f"{gpu.memoryFree} MB",
                    'utilization': f"{gpu.memoryUtil*100:.1f}%"
                },
                'temperature': f"{gpu.temperature} °C",
                'uuid': gpu.uuid
            })
        return {'gpus': gpu_info}
    except Exception as e:
        return {'error': f"Unable to get GPU information: {str(e)}"}

def get_external_ip() -> Dict:
    """Get external IP information"""
    try:
        # Try multiple IP services in case one fails
        ip_services = [
            'https://api.ipify.org?format=json',
            'https://api.myip.com',
            'https://ip.seeip.org/json'
        ]
        
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        
        for service in ip_services:
            try:
                with urllib.request.urlopen(service, context=context, timeout=5) as response:
                    data = json.loads(response.read())
                    # Different services return different JSON structures
                    if 'ip' in data:
                        return {'external_ip': data['ip']}
                    break
            except (urllib.error.URLError, json.JSONDecodeError, KeyError):
                continue
                
        # Fallback to socket if web services fail
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # Doesn't need to be reachable
            s.connect(('8.8.8.8', 1))
            return {'external_ip': s.getsockname()[0]}
        except Exception:
            return {'external_ip': 'Unable to determine'}
        finally:
            s.close()
    except Exception as e:
        return {'external_ip': f'Error: {str(e)}'}

def get_network_info() -> Dict:
    """Get detailed network information"""
    try:
        network_info = {
            'interfaces': {},
            'connections': [],
            'stats': psutil.net_io_counters()._asdict()
        }
        
        # Add external IP information
        network_info.update(get_external_ip())
        
        # Network interfaces
        for nic, addrs in psutil.net_if_addrs().items():
            nic_info = []
            for addr in addrs:
                nic_info.append({
                    'address': addr.address,
                    'netmask': addr.netmask,
                    'family': str(addr.family),
                    'broadcast': addr.broadcast if hasattr(addr, 'broadcast') else None,
                    'ptp': addr.ptp if hasattr(addr, 'ptp') else None
                })
            network_info['interfaces'][nic] = {
                'addresses': nic_info,
                'stats': psutil.net_if_stats()[nic]._asdict() if nic in psutil.net_if_stats() else {}
            }
        
        # Active connections
        for conn in psutil.net_connections(kind='inet'):
            connection_info = {
                'fd': conn.fd,
                'family': str(conn.family),
                'type': str(conn.type),
                'local_addr': f"{conn.laddr.ip}:{conn.laddr.port}" if conn.laddr else None,
                'remote_addr': f"{conn.raddr.ip}:{conn.raddr.port}" if conn.raddr else None,
                'status': conn.status,
                'pid': conn.pid
            }
            network_info['connections'].append(connection_info)
            
        return network_info
    except Exception as e:
        raise SystemToolError(f"Error getting network information: {str(e)}")

def get_process_info() -> Dict:
    """Get information about running processes"""
    try:
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'username', 'cpu_percent', 'memory_percent', 'create_time', 'status']):
            try:
                pinfo = proc.info
                pinfo['create_time'] = format_datetime(pinfo['create_time'])
                processes.append(pinfo)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        return {'processes': sorted(processes, key=lambda x: x.get('cpu_percent', 0), reverse=True)}
    except Exception as e:
        raise SystemToolError(f"Error getting process information: {str(e)}")

def get_services_info() -> Dict:
    """Get information about system services"""
    try:
        w = wmi.WMI()
        services = []
        for service in w.Win32_Service():
            services.append({
                'name': service.Name,
                'display_name': service.DisplayName,
                'status': service.State,
                'start_mode': service.StartMode,
                'path': service.PathName,
                'description': service.Description
            })
        return {'services': services}
    except Exception as e:
        raise SystemToolError(f"Error getting services information: {str(e)}")

def get_disk_info(path: Optional[str] = None) -> Dict:
    """Get disk information for specified path or all mounted drives"""
    try:
        if path:
            if not os.path.exists(path):
                raise SystemToolError(f"Path does not exist: {path}")
            disk = psutil.disk_usage(path)
            return {
                'total': bytes2human(disk.total),
                'used': bytes2human(disk.used),
                'free': bytes2human(disk.free),
                'percent': f"{disk.percent}%"
            }
        else:
            # Get info for all mounted partitions
            partitions = {}
            for partition in psutil.disk_partitions(all=True):
                try:
                    disk = psutil.disk_usage(partition.mountpoint)
                    partitions[partition.mountpoint] = {
                        'device': partition.device,
                        'fstype': partition.fstype,
                        'opts': partition.opts,
                        'total': bytes2human(disk.total),
                        'used': bytes2human(disk.used),
                        'free': bytes2human(disk.free),
                        'percent': f"{disk.percent}%"
                    }
                except (PermissionError, OSError):
                    continue
            return partitions
    except Exception as e:
        raise SystemToolError(f"Error getting disk information: {str(e)}")

def execute(arguments: Dict[str, str]) -> str:
    """Execute system tool to get system information
    
    Args:
        arguments: Dictionary containing:
            type (str): Type of information to retrieve 
                ('os', 'memory', 'disk', 'cpu', 'gpu', 'network', 'processes', 'services', 'storage', 'all')
            path (str, optional): Specific path for disk information
            
    Returns:
        str: JSON formatted system information
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
                'boot_time': format_datetime(psutil.boot_time()),
                'uptime': str(datetime.now() - datetime.fromtimestamp(psutil.boot_time()))
            }, indent=2)
            
        elif info_type == 'memory':
            mem = psutil.virtual_memory()
            swap = psutil.swap_memory()
            return json.dumps({
                'virtual_memory': {
                    'total': bytes2human(mem.total),
                    'available': bytes2human(mem.available),
                    'used': bytes2human(mem.used),
                    'free': bytes2human(mem.free),
                    'percent': f"{mem.percent}%",
                    'cached': bytes2human(getattr(mem, 'cached', 0)),
                    'buffers': bytes2human(getattr(mem, 'buffers', 0))
                },
                'swap_memory': {
                    'total': bytes2human(swap.total),
                    'used': bytes2human(swap.used),
                    'free': bytes2human(swap.free),
                    'percent': f"{swap.percent}%"
                }
            }, indent=2)
            
        elif info_type == 'disk':
            return json.dumps(get_disk_info(path), indent=2)
            
        elif info_type == 'storage':
            return json.dumps(get_storage_details(), indent=2)
            
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
                'cpu_times_percent': psutil.cpu_times_percent()._asdict(),
                'load_avg': psutil.getloadavg()
            }, indent=2)
            
        elif info_type == 'gpu':
            return json.dumps(get_gpu_info(), indent=2)
            
        elif info_type == 'network':
            return json.dumps(get_network_info(), indent=2)
            
        elif info_type == 'processes':
            return json.dumps(get_process_info(), indent=2)
            
        elif info_type == 'services':
            return json.dumps(get_services_info(), indent=2)
            
        else:  # all
            all_info = {
                'os': {
                    'system': platform.system(),
                    'release': platform.release(),
                    'version': platform.version(),
                    'machine': platform.machine(),
                    'processor': platform.processor(),
                    'architecture': platform.architecture(),
                    'node': platform.node(),
                    'python_version': platform.python_version(),
                    'boot_time': format_datetime(psutil.boot_time()),
                    'uptime': str(datetime.now() - datetime.fromtimestamp(psutil.boot_time()))
                },
                'memory': {
                    'virtual_memory': {
                        'total': bytes2human(psutil.virtual_memory().total),
                        'available': bytes2human(psutil.virtual_memory().available),
                        'used': bytes2human(psutil.virtual_memory().used),
                        'percent': f"{psutil.virtual_memory().percent}%"
                    },
                    'swap_memory': {
                        'total': bytes2human(psutil.swap_memory().total),
                        'used': bytes2human(psutil.swap_memory().used),
                        'free': bytes2human(psutil.swap_memory().free),
                        'percent': f"{psutil.swap_memory().percent}%"
                    }
                },
                'storage': get_storage_details(),
                'cpu': {
                    'physical_cores': psutil.cpu_count(logical=False),
                    'total_cores': psutil.cpu_count(logical=True),
                    'cpu_freq': {
                        'current': f"{psutil.cpu_freq().current:.2f} MHz" if psutil.cpu_freq() else "N/A",
                        'min': f"{psutil.cpu_freq().min:.2f} MHz" if psutil.cpu_freq() and hasattr(psutil.cpu_freq(), 'min') else "N/A",
                        'max': f"{psutil.cpu_freq().max:.2f} MHz" if psutil.cpu_freq() and hasattr(psutil.cpu_freq(), 'max') else "N/A"
                    },
                    'cpu_usage_per_core': [f"{percentage:.1f}%" for percentage in get_cached_cpu_percent(percpu=True)],
                    'cpu_usage_total': f"{get_cached_cpu_percent():.1f}%",
                    'load_avg': psutil.getloadavg()
                },
                'gpu': get_gpu_info(),
                'network': get_network_info()
            }
            return json.dumps(all_info, indent=2)
            
    except SystemToolError as e:
        return json.dumps({"error": str(e)}, indent=2)
    except Exception as e:
        return json.dumps({
            "error": "Unexpected error occurred",
            "details": str(e),
            "type": type(e).__name__
        }, indent=2) 