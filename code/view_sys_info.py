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

# Python version
import platform
print(f"Python version: {platform.python_version()}")

# Package versions
import numpy, pandas, matplotlib, sklearn
print(f"NumPy version: {numpy.__version__}")
print(f"Pandas version: {pandas.__version__}")
print(f"Matplotlib version: {matplotlib.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")
