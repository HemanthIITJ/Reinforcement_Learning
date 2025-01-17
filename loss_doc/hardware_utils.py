import torch
import platform
import psutil
import GPUtil
import time
from datetime import datetime

def monitor_system():
    device_info = {
        'os': platform.system(),
        'architecture': platform.machine(),
        'processor': platform.processor(),
        'cuda_available': torch.cuda.is_available(),
        'cuda_devices': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'cuda_details': [torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else None
    }
    
    while True:
        with open('system_monitor.log', 'a') as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            gpu_stats = GPUtil.getGPUs() if torch.cuda.is_available() else []
            
            f.write(f"\n{timestamp}\nCPU: {cpu_percent}%\nRAM: {memory.percent}%\n")
            for gpu in gpu_stats:
                f.write(f"GPU {gpu.id}: Memory {gpu.memoryUtil*100}%, Load {gpu.load*100}%\n")
        time.sleep(60)



def get_device_info():
    info = {}
    info['system'] = platform.system()
    info['architecture'] = platform.machine()
    if torch.cuda.is_available():
        info['device'] = 'cuda'
        info['cuda_devices'] = []
        for i in range(torch.cuda.device_count()):
            device = torch.cuda.get_device_properties(i)
            info['cuda_devices'].append({
                'name': device.name,
                'total_memory_gb': device.total_memory / 1024**3,
                'multiprocessor_count': device.multi_processor_count,
                'cuda_capability': f"{device.major}.{device.minor}"
            })
    else:
        info['device'] = 'cpu'
    return info

def log_memory_usage(filename="memory_log.txt"):
    with open(filename, "a") as f:
        mem = psutil.virtual_memory()
        f.write(f"{datetime.now()} - Total: {mem.total / 1024**2:.2f} MB, Available: {mem.available / 1024**2:.2f} MB, Used: {mem.used / 1024**2:.2f} MB, Percent: {mem.percent}%\n")

if __name__ == "__main__":
    device_info = get_device_info()
    print(device_info)
    log_memory_usage()