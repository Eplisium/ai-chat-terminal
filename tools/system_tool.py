import platform
import psutil
import json

def execute(arguments):
    """Execute system tool to get system information"""
    info_type = arguments.get('type', 'all')
    
    try:
        if info_type == 'os':
            return json.dumps({
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'processor': platform.processor()
            }, indent=2)
            
        elif info_type == 'memory':
            mem = psutil.virtual_memory()
            return json.dumps({
                'total': f"{mem.total / (1024**3):.2f} GB",
                'available': f"{mem.available / (1024**3):.2f} GB",
                'used': f"{mem.used / (1024**3):.2f} GB",
                'percent': f"{mem.percent}%"
            }, indent=2)
            
        elif info_type == 'disk':
            disk = psutil.disk_usage('/')
            return json.dumps({
                'total': f"{disk.total / (1024**3):.2f} GB",
                'used': f"{disk.used / (1024**3):.2f} GB",
                'free': f"{disk.free / (1024**3):.2f} GB",
                'percent': f"{disk.percent}%"
            }, indent=2)
            
        elif info_type == 'cpu':
            return json.dumps({
                'physical_cores': psutil.cpu_count(logical=False),
                'total_cores': psutil.cpu_count(logical=True),
                'cpu_freq': {
                    'current': f"{psutil.cpu_freq().current:.2f} MHz",
                    'min': f"{psutil.cpu_freq().min:.2f} MHz",
                    'max': f"{psutil.cpu_freq().max:.2f} MHz"
                },
                'cpu_usage_per_core': [f"{percentage:.1f}%" for percentage in psutil.cpu_percent(percpu=True)]
            }, indent=2)
            
        else:  # all
            return json.dumps({
                'os': {
                    'system': platform.system(),
                    'release': platform.release(),
                    'version': platform.version(),
                    'machine': platform.machine(),
                    'processor': platform.processor()
                },
                'memory': {
                    'total': f"{psutil.virtual_memory().total / (1024**3):.2f} GB",
                    'available': f"{psutil.virtual_memory().available / (1024**3):.2f} GB",
                    'used': f"{psutil.virtual_memory().used / (1024**3):.2f} GB",
                    'percent': f"{psutil.virtual_memory().percent}%"
                },
                'disk': {
                    'total': f"{psutil.disk_usage('/').total / (1024**3):.2f} GB",
                    'used': f"{psutil.disk_usage('/').used / (1024**3):.2f} GB",
                    'free': f"{psutil.disk_usage('/').free / (1024**3):.2f} GB",
                    'percent': f"{psutil.disk_usage('/').percent}%"
                },
                'cpu': {
                    'physical_cores': psutil.cpu_count(logical=False),
                    'total_cores': psutil.cpu_count(logical=True),
                    'cpu_freq': {
                        'current': f"{psutil.cpu_freq().current:.2f} MHz",
                        'min': f"{psutil.cpu_freq().min:.2f} MHz",
                        'max': f"{psutil.cpu_freq().max:.2f} MHz"
                    },
                    'cpu_usage_per_core': [f"{percentage:.1f}%" for percentage in psutil.cpu_percent(percpu=True)]
                }
            }, indent=2)
            
    except Exception as e:
        return f"Error getting system information: {str(e)}" 