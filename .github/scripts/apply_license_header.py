import os

# Path to the license header
HEADER_FILE = os.path.join(os.path.dirname(__file__), "license-header.txt")

# read license header
with open(HEADER_FILE) as file:
    license_header = file.read()


# Add license header to a file
def add_license_header(file_path, header):
    with open(file_path) as f:
        content = f.read()

    # Check if the header is already there
    if content.startswith(header):
        return

    # If not, add it
    with open(file_path, "w") as f:
        f.write(header + "\n" + content)
    print(f"Header added to {file_path}")


# Check the header in every file in root_dir
def apply_license_header_to_all_py_files(root_dir):
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(subdir, file)
                add_license_header(file_path, license_header)


# Source directory
PROJECT_SRC = "xdem"

# Add header to every source files
apply_license_header_to_all_py_files(PROJECT_SRC)
