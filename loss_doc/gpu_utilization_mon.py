import platform, torch, psutil, pynvml, time, datetime
import torch
import platform
import psutil
import pynvml
import time
from datetime import datetime
def monitor():
    with open("gpu_details.log", "a") as f:
        f.write(f"Timestamp: {datetime.datetime.now()}\n")
        f.write(f"System: {platform.system()} {platform.machine()}\n")
        if torch.cuda.is_available():
            f.write("Device: CUDA\n")
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            f.write(f"Number of GPUs: {count}\n")
            for i in range(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                props = torch.cuda.get_device_properties(i)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                f.write(f"  GPU {i}:\n")
                name = pynvml.nvmlDeviceGetName(handle)
                f.write(f"    Name: {name.decode() if isinstance(name, bytes) else name}\n")
                f.write(f"    Architecture: {props.major}.{props.minor}\n")
                f.write(f"    Total RAM (GB): {props.total_memory / 1024**3:.2f}\n")
                f.write(f"    Cores: {props.multi_processor_count}\n")
                f.write(f"    Driver Version: {pynvml.nvmlSystemGetCudaDriverVersion() / 1000:.2f}\n")
                f.write(f"    Memory (MB) - Total: {mem_info.total / 1024**2:.2f}, Used: {mem_info.used / 1024**2:.2f}\n")
                # f.write(f"    Max Blocks: {props.max_blocks_per_multi_processor}\n")
                # f.write(f"    Clock Rate: {props.clock_rate/1000:.2f} GHz\n")
            pynvml.nvmlShutdown()
        else:
            f.write("Device: CPU\n")
            mem = psutil.virtual_memory()
            f.write(f"  CPU Memory (MB) - Total: {mem.total / 1024**2:.2f}, Used: {mem.used / 1024**2:.2f}\n")
        f.write("-" * 30 + "\n")
        f.write("Memory Monitoring:\n")
        while True:
            if torch.cuda.is_available():
                pynvml.nvmlInit()
                count = pynvml.nvmlDeviceGetCount()
                for i in range(count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    f.write(f"{datetime.datetime.now()} - GPU {i} Memory Used: {mem_info.used / 1024**2:.2f} MB\n")
                pynvml.nvmlShutdown()
            else:
                mem = psutil.virtual_memory()
                f.write(f"{datetime.datetime.now()} - CPU Memory Used: {mem.used / 1024**2:.2f} MB\n")
            f.flush()
            time.sleep(60)





def gpu_monitor():
    with open('gpu_details.log', 'w') as f:
        f.write(f"System: {platform.system()} {platform.machine()}\n")
        f.write(f"Processor: {platform.processor()}\n")
        f.write(f"CUDA Available: {torch.cuda.is_available()}\n")
        
        if torch.cuda.is_available():
            pynvml.nvmlInit()
            gpu_count = pynvml.nvmlDeviceGetCount()
            f.write(f"Total GPUs: {gpu_count}\n\n")
            
            for i in range(gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_name = pynvml.nvmlDeviceGetName(handle)
                compute_mode = pynvml.nvmlDeviceGetComputeMode(handle)
                cuda_ver = pynvml.nvmlSystemGetCudaDriverVersion()
                gpu_temp = pynvml.nvmlDeviceGetTemperature(handle, 0)
                gpu_power = pynvml.nvmlDeviceGetPowerUsage(handle)
                props = torch.cuda.get_device_properties(i)
                
                f.write(f"GPU {i} Details:\n")
                f.write(f"Name: {gpu_name if isinstance(gpu_name, str) else gpu_name.decode()}\n")
                f.write(f"Architecture: {props.major}.{props.minor}\n")
                f.write(f"Total Memory: {info.total / 1024**2:.2f} MB\n")
                f.write(f"CUDA Cores: {props.multi_processor_count * 64}\n")
                f.write(f"CUDA Version: {cuda_ver/1000:.2f}\n")
                f.write(f"Temperature: {gpu_temp}°C\n")
                f.write(f"Power Usage: {gpu_power/1000:.2f}W\n")
                f.write(f"Compute Mode: {compute_mode}\n")
                # f.write(f"Max Blocks: {props.max_blocks_per_multi_processor}\n")
                # f.write(f"Clock Rate: {props.clock_rate/1000:.2f} GHz\n\n")
                
            while True:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"\n{timestamp} - Memory Usage:\n")
                for i in range(gpu_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    f.write(f"GPU {i}: Used Memory: {info.used/1024**2:.2f}MB, Utilization: {utilization.gpu}%\n")
                f.flush()
                time.sleep(60)

# if __name__ == "__main__":
#     gpu_monitor()
# if __name__ == "__main__":
#     monitor()