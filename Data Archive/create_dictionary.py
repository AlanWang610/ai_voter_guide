import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from gensim.utils import simple_preprocess
from gensim.models.ldamodel import LdaModel
from gensim.matutils import cossim

import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
# nltk.download('wordnet')
from nltk.corpus import wordnet

from gensim import corpora
from gensim.models import LdaModel
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

# Load user paragraph and candidate lines
with open('user_paragraph_processed.txt', 'r') as f:
    user_paragraph_text = f.read()

with open('Harris_processed.txt', 'r') as f:
    candidate_lines = f.read()

# Ensure that candidate lines are properly split
candidate_lines_subset = candidate_lines.split('\n')[:500]  # Limit to 1000 lines

# Example preprocessing function
def preprocess(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text.lower())  # Tokenize into words
    # Ensure that tokens are alphabetic and not stop words
    return [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]

# Preprocess the user paragraph and the candidate lines
user_paragraph = preprocess(user_paragraph_text)  # Preprocess user input

# Preprocess each candidate line (line by line)
candidate_docs = [preprocess(line) for line in candidate_lines_subset]  # Preprocess each candidate line

# Check if preprocessing resulted in meaningful tokens
if any(len(doc) == 0 for doc in candidate_docs):
    print("Warning: Some candidate lines might have no meaningful words after preprocessing.")

# Create dictionary and corpus
dictionary = corpora.Dictionary([user_paragraph] + candidate_docs)
corpus = [dictionary.doc2bow(doc) for doc in [user_paragraph] + candidate_docs]

# Train the LDA model
lda_model = LdaModel(corpus=corpus, num_topics=10, id2word=dictionary, passes=10)

# Get the topic distribution of the user paragraph
user_bow = dictionary.doc2bow(user_paragraph)  # Convert user input to BoW format
user_topics = lda_model.get_document_topics(user_bow)

# Function to get similarity between a candidate line and the user paragraph
def get_similarity(line_bow, user_topics):
    line_topics = lda_model.get_document_topics(line_bow)
    return cossim(user_topics, line_topics)

# Rank candidate lines based on similarity
similarities = [get_similarity(dictionary.doc2bow(doc), user_topics) for doc in candidate_docs]
ranked_lines = sorted(zip(candidate_lines_subset, similarities), key=lambda x: x[1], reverse=True)

# Output the ranked lines along with their similarity scores
for line, score in ranked_lines[:10]:  # Limiting output to the top 10 lines for brevity
    print(f"Line: {line} | Similarity: {score}")
