import sys
import re

file_name = sys.argv[1]
# read the file
with open(file_name, 'r', encoding='utf-8') as f:
    content = f.read()
# within content, scan each line for a period, if a period is found, 
# then cut the line at the period and continue the line on the next line
# there may be multiple periods in a line, in which case a single line could spread into multiple lines
# however, if a line ends without a period, combine it with the next line
lines = content.split('\n')
new_lines = []
buffer = ''

for line in lines:
    line = buffer + line
    buffer = ''
    while '.' in line:
        index = line.index('.')
        new_lines.append(line[:index + 1])
        line = line[index + 1:]
    if line:
        buffer = line

if buffer:
    new_lines.append(buffer)

# if a line is less than 50 characters, remove it
new_lines = [line for line in new_lines if len(line) >= 50]

# write the new lines to a new file
try:
    with open(file_name, 'w', encoding='utf-8') as f:
        for line in new_lines:
            f.write(line + '\n')
except PermissionError:
    print(f"Error: Permission denied to write to the file '{file_name}'.")
    sys.exit(1)
except Exception as e:
    print(f"Error: An unexpected error occurred while writing to the file '{file_name}': {e}")
    sys.exit(1)

with open(file_name, 'r', encoding='utf-8') as f:
    content = f.read()

# preprocess the text data
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'\S+@\S+\.\S+', '', text)  # Remove email addresses
    text = re.sub('\'', '', text)  # Remove apostrophes
    text = re.sub('[^a-zA-Z]', ' ', text)  # Remove non-alphabet characters
    text = text.lower()  # Convert to lowercase
    return text

# remove any lines with less than 50 characters
content = '\n'.join([line for line in content.split('\n') if len(line) >= 50])

# preprocess each line of the text data
content = '\n'.join([preprocess_text(line) for line in content.split('\n')])

# save file
output_file_name = f'{file_name[:-4]}_processed.txt'
try:
    with open(output_file_name, 'w', encoding='utf-8') as f:
        f.write(content)
except PermissionError:
    print(f"Error: Permission denied to write to the file '{output_file_name}'.")
    sys.exit(1)