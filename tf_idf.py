from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
def tf_idf(user_lines, candidate_lines):
    # Ensure that candidate lines are properly split
    candidate_lines = candidate_lines.split('\n')

    # Use TfidfVectorizer to vectorize both the user paragraph and candidate lines
    vectorizer = TfidfVectorizer()

    # Combine user paragraph and candidate lines for fitting the vectorizer
    all_text = [user_lines] + candidate_lines

    # Fit and transform the text
    tfidf_matrix = vectorizer.fit_transform(all_text)

    # Compute cosine similarity between the user paragraph (first item) and candidate lines
    user_paragraph_tfidf = tfidf_matrix[0]
    candidate_lines_tfidf = tfidf_matrix[1:]

    # Compute cosine similarities
    cosine_similarities = cosine_similarity(user_paragraph_tfidf, candidate_lines_tfidf).flatten()

    # Find the top 50 most relevant lines
    top_50_indices = np.argsort(cosine_similarities)[-50:][::-1]
    top_50_lines = [candidate_lines[i] for i in top_50_indices]

    # Output or process the top 50 lines as needed
    with open('relevant_candidate_lines.txt', 'w') as f:
        for line in top_50_lines:
            f.write(f"{line}\n")