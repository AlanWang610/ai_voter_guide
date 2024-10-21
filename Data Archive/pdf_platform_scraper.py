import requests
import sys
import re
import pdfplumber
import nltk
from nltk.tokenize import sent_tokenize

url = sys.argv[1]

def extract_text_from_pdf(pdf_content):
    all_text = []
    
    # Open the PDF content using pdfplumber
    with pdfplumber.open(pdf_content) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_text.append(text)
    
    return "\n".join(all_text)

# Download the PDF file content
response = requests.get(url)
pdf_content = response.content

# Save the PDF content to a temporary file
with open('temp.pdf', 'wb') as f:
    f.write(pdf_content)

# Extract text from the saved PDF
result = extract_text_from_pdf('temp.pdf')

# Remove any sentences (so last period to next period) that contain the string 'Trump' or 'Donald'
nltk.download('punkt')

sentences = sent_tokenize(result)
filtered_sentences = [sentence for sentence in sentences if 'Trump' not in sentence and 'Donald' not in sentence]
result = ' '.join(filtered_sentences)
# Remove any lines from the result that contain the string 'Trump' or 'Donald'
filtered_result = filter(lambda line: 'Trump' not in line and 'Donald' not in line, result.split('\n'))
result = '\n'.join(filtered_result)
# Remove any lines that have less than 10 characters and no period
filtered_result = filter(lambda line: len(line) > 10 and '.' in line, result.split('\n'))
result = '\n'.join(filtered_result)

# Save the result as a .txt file with UTF-8 encoding
filename = re.sub(r'[./]', '', url[8:]) + '.txt'

with open(filename, 'w', encoding='utf-8') as f:
    f.write(result)
