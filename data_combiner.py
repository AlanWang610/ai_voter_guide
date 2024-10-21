import sys
import os

folder_name = sys.argv[1]
# import all txt files from folder_name and combine them into a single file
files = [os.path.join(folder_name, file) for file in os.listdir(folder_name) if file.endswith('.txt')]

with open(f'{folder_name}.txt', 'w', encoding='utf-8') as combined_file:
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            content = f.read()
            ascii_content = content.encode('ascii', 'ignore').decode('ascii')
            combined_file.write(ascii_content)