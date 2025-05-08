import os
from pathlib import Path

# Define the main project directory
project_name = "/home/srivabhi22/Desktop/navigation"
base_dir = Path(project_name)

# Create main project directory
base_dir.mkdir(exist_ok=True)

# Define subdirectories
subdirs = [
    "src/planners",
    "src/sensors",
    "src/preprocessing",
    "src/utils",
    "config",
    "tests",
    "docs",
    "logs",
    "data/raw",
    "data/processed"
]

# Create subdirectories
for subdir in subdirs:
    (base_dir / subdir).mkdir(parents=True, exist_ok=True)

# Create main files
files = [
    "src/planners/__init__.py",
    "src/planners/dwa_planner.py",
    "src/sensors/image_capture.py",
    "src/sensors/pointcloud_capture.py",
    "src/preprocessing/data_processor.py",
    "src/utils/logger.py",
    "config/navigation_params.yaml",
    "config/sensor_config.yaml",
    "tests/test_planner.py",
    "tests/test_sensors.py",
    "tests/test_preprocessing.py",
    "docs/README.md",
    "docs/architecture.md",
    "main.py",
    "requirements.txt",
    ".gitignore"
]

# Create files
for file in files:
    (base_dir / file).touch(exist_ok=True)

# Add content to README.md
with open(base_dir / 'docs/README.md', 'w') as readme_file:
    readme_file.write("# Quadruped Navigation System\n\n")
    readme_file.write("This project implements a navigation system for a quadruped robot, including DWA planning, sensor data capture, and preprocessing.\n")

# Add content to .gitignore
with open(base_dir / '.gitignore', 'w') as gitignore_file:
    gitignore_file.write("*.pyc\n")
    gitignore_file.write("__pycache__/\n")
    gitignore_file.write("logs/\n")
    gitignore_file.write("data/\n")

print(f"Project structure '{project_name}' created successfully!")
