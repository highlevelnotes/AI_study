import os

def find_large_files_and_update_gitignore(folder_path, size_limit_mb=100, gitignore_path='./.gitignore'):
    size_limit_bytes = size_limit_mb * 1024 * 1024
    large_files = []

    # Walk through the folder and find files larger than size_limit_bytes
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                if os.path.getsize(file_path) > size_limit_bytes:
                    large_files.append(file_path)
            except OSError:
                # Skip files that can't be accessed
                continue

    # Read existing .gitignore content
    try:
        with open(gitignore_path, 'r') as f:
            gitignore_content = f.read().splitlines()
    except FileNotFoundError:
        gitignore_content = []

    # Append new large file paths to .gitignore if not already present
    with open(gitignore_path, 'a') as f:
        for file_path in large_files:
            # Convert absolute path to relative path if needed
            rel_path = os.path.relpath(file_path, start=os.path.dirname(gitignore_path))
            if rel_path not in gitignore_content:
                f.write(rel_path + '\n')

    return large_files

# 사용 예시
folder_to_search = '.'
large_files_added = find_large_files_and_update_gitignore(folder_to_search)
print(f"Found {len(large_files_added)} large files")
