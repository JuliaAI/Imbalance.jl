import subprocess
import os
import shutil

def convert_to_md(name, copy=True):
    md_name = name + '.md'
    ipynb_name = name + '.ipynb'
    command = ["jupyter", "nbconvert", "--to", "markdown", ipynb_name]
    subprocess.run(command, check=True)

    file_path = md_name  # Replace with the path to your Markdown file

    import re
    with open(file_path, 'r+') as file:
        content = file.read()

        # Task 1: Replace "python" with "julia"
        content = content.replace('python', 'julia')

        # Task 2: Remove ANSI escape codes (e.g., "[1m")
        ansi_escape_regex = re.compile(r'\x1B\[\d+m')
        content = re.sub(ansi_escape_regex, '', content)

        # Task 3: Delete text after !jupyter
        content = re.sub(r'import sys.*', '', content, flags=re.DOTALL)

        # Task 4: Delete the last occurrence of ```julia
        last_julia_code = r'```julia.*?'
        last_julia_match = re.finditer(last_julia_code, content, flags=re.DOTALL | re.MULTILINE)
        last_julia_positions = [match.span() for match in last_julia_match]
        
        if last_julia_positions:
            last_julia_start, last_julia_end = last_julia_positions[-1]
            content = content[:last_julia_start] + content[last_julia_end:]

        file.seek(0)
        file.write(content)
        file.truncate()
    if copy:    copy_images_to_parent_directory()
    print("Conversion Complete!")


# Needed due to a bug in documenter. When "./smote-animation.gif" is put as the path
# it becomes "../smote-animation.gif" in HTML and renders nothing. When its "../smote-animation.gif"
# it becomes "../../smote-animation.gif" and renders nothing. Basic inefficient solution to deal with
# this madness (which also depends on where file exactly is!!!) is to maintain two copies of the file.
def copy_images_to_parent_directory():
    source_dir = "./assets"
    destination_dir = "../assets"
    
    # List of valid image file extensions
    valid_extensions = ['.png', '.jpg', '.jpeg', '.gif']

    try:
        for filename in os.listdir(source_dir):
            if os.path.isfile(os.path.join(source_dir, filename)):
                # Check if the file has a valid image extension
                ext = os.path.splitext(filename)[-1].lower()
                if ext in valid_extensions:
                    src_path = os.path.join(source_dir, filename)
                    dest_path = os.path.join(destination_dir, filename)
                    shutil.copy(src_path, dest_path)
                    print(f"Copied {filename} to {dest_path}")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")