import subprocess

def convert_to_md(name):
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

    print("Conversion Complete!")
