import re
import os

def camel_to_snake(name):
    # Convert camelCase to snake_case
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def convert_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    # Regex to match camelCase variables outside of string literals
    camel_case_pattern = re.compile(r'\b([a-z]+(?:[A-Z][a-z]*)+)\b(?=(?:[^\'"]*\'[^\'"]*\')*[^\'"]*$)')

    # Replace each camelCase variable with snake_case
    def replace(match):
        camel_case = match.group(1)
        return camel_to_snake(camel_case)

    content = camel_case_pattern.sub(replace, content)

    with open(file_path, 'w') as file:
        file.write(content)

def convert_directory(directory_path):
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.mojo'):
                convert_file(os.path.join(root, file))

if __name__ == "__main__":
    convert_directory(".")