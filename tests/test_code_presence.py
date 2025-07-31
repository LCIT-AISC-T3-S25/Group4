import os

def test_python_files_exist():
    # Walk through the project (excluding common virtual env dirs)
    found_py = False
    for root, dirs, files in os.walk("."):
        # Optionally skip environments or irrelevant folders
        if any(skip in root for skip in ['__pycache__', 'venv', '.venv', '.git']):
            continue
        if any(f.endswith('.py') for f in files):
            found_py = True
            break
    assert found_py, "No Python (.py) files found in the project."
