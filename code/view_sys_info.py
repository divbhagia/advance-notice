import psutil

# CPU information
print(f"Physical cores: {psutil.cpu_count(logical=False)}")
print(f"Total cores: {psutil.cpu_count(logical=True)}")

# Memory Information
vm = psutil.virtual_memory()
print(f"Total Memory: {vm.total / (1024 ** 3):.2f} GB")
print(f"Available Memory: {vm.available / (1024 ** 3):.2f} GB")
print(f"Used Memory: {vm.used / (1024 ** 3):.2f} GB")
print(f"Memory Usage: {vm.percent}%")
