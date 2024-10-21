from bs4 import BeautifulSoup
from bs4.element import Comment
import requests
import sys
import re
import nltk
from nltk.tokenize import sent_tokenize
url = sys.argv[1]

def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]', 'header', 'footer']:
        return False
    if isinstance(element, Comment):
        return False
    return True


def text_from_html(body):
    soup = BeautifulSoup(body, 'html.parser')

    # Find the main content container. Adjust this based on the structure of the page you're scraping.
    main_content = soup.find('main')  # Try to get the main section first.
    if not main_content:
        main_content = soup.body  # If there's no <main>, default to the body.

    texts = main_content.findAll(string=True)
    visible_texts = filter(tag_visible, texts)

    # Adding line breaks before certain section-related tags
    formatted_text = []
    for element in main_content.descendants:
        if element.name in ['h1', 'h2', 'h3', 'h4', 'p']:
            formatted_text.append("\n")  # Add a line break before each subsection
        if isinstance(element, str) and tag_visible(element):
            formatted_text.append(element.strip())

    return u" ".join(formatted_text)

# with open('Pages.txt', 'r') as file:
#     urls = file.readlines()

# for url in urls:
    # url = url.strip()  # Remove any leading/trailing whitespace
response = requests.get(url)
response.encoding = 'utf-8'  # Ensure proper decoding
html = response.text  # Decode the response content as UTF-8

result = text_from_html(html)
# Remove any sentences (so last period to next period) that contain the string 'Trump' or 'Donald'
nltk.download('punkt')
nltk.download('punkt_tab')

sentences = sent_tokenize(result)
filtered_sentences = [sentence for sentence in sentences if 'Biden' not in sentence and 'Joe' not in sentence and 'Kamala' not in sentence and 'Harris' not in sentence and 'Donald' not in sentence and 'Trump' not in sentence]
result = ' '.join(filtered_sentences)
# remove any lines from result that contain the string 'Biden' or 'Joe'
# filtered_result = filter(lambda line: 'Biden' not in line and 'Joe' not in line, result.split('\n'))
# result = '\n'.join(filtered_result)
# Remove any lines from result that are shorter than 300 characters
# filtered_result = filter(lambda line: len(line) > 300, result.split('\n'))
# result = '\n'.join(filtered_result)
# Save result as txt file with UTF-8 encoding
# Remove the first 8 characters, any periods, or slashes from the URL to create a valid filename

filename = re.sub(r'[./]', '', url[8:]) + '.txt'

# filename = 'Agenda47/' + filename

with open(filename, 'w', encoding='utf-8') as f:
    f.write(result)