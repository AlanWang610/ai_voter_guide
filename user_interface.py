from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv
import os
import google.generativeai as genai
import typing_extensions as typing
import json
import re

load_dotenv()
api_key = os.getenv("API_KEY")

def user_lines_processor(user_lines):
    lines = user_lines.split('\n')
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
    # preprocess the text data
    def preprocess_text(text):
        text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
        text = re.sub(r'\S+@\S+\.\S+', '', text)  # Remove email addresses
        text = re.sub('\'', '', text)  # Remove apostrophes
        text = re.sub('[^a-zA-Z]', ' ', text)  # Remove non-alphabet characters
        text = text.lower()  # Convert to lowercase
        return text
    # preprocess each line of the text data
    new_lines = [preprocess_text(line) for line in new_lines]
    return new_lines

def tf_idf(user_lines, candidate_lines):
    # If user_lines is a list, join it into a single string
    if isinstance(user_lines, list):
        user_lines = ' '.join(user_lines)
    
    # If candidate_lines is a list, join each candidate's lines into a single string
    if isinstance(candidate_lines, list):
        candidate_lines = [' '.join(candidate_lines)]
    
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
    
    # Find the top 10 most relevant lines
    top_10_indices = np.argsort(cosine_similarities)[-10:][::-1]
    top_10_candidate_lines = [candidate_lines[i] for i in top_10_indices]
    
    # Output or process the top 10 lines as needed
    return top_10_candidate_lines

def tf_idf_multiple(user_lines, candidates_lines):
    candidates_lines_selected = [None] * len(candidates_lines)
    for i in range(len(candidates_lines)):
        candidates_lines_selected[i] = tf_idf(user_lines, candidates_lines[i])
    return candidates_lines_selected

def call_llm(user_lines, candidates_lines):
    n = len(candidates_lines)
    genai.configure(api_key=api_key)
    
    class Similarity(typing.TypedDict):
        agreement_score: float
        rationale: str

    model = genai.GenerativeModel("gemini-1.5-flash")

    # Function to run one query for each candidate
    def get_similarity_for_candidate(user_lines, candidate_lines):
        response = model.generate_content(
            f"""
            Act as a skilled political analyst and provide a nuanced opinion on how closely a user's position aligns with a given candidate's position.
            I will provide one input titled user_lines and one input titled candidate_lines. 
            Create an output that is a value between -1 to 1 that represents how closely you believe the user's position matches with the candidate's position. 
            You should not extrapolate to other candidates or other policy domains.
            Focus on the aspects of the policy domain that are mentioned by the user and the candidate, and don't worry if the user doesn't mention a key part of the candidate's position.
            Do not penalize the score for the user not mentioning a key part of the candidate's position.
            You are allowed and encouraged to output a decimal value. 
            -1 represents a user position that is perfectly opposed to the candidate's position, 1 represents a user position that is perfectly aligned with the candidate's position, 
            and 0 means that there is no conflict nor agreement.
            Please provide a value with up to two decimals of precision for agreement_score.
            Also describe why you chose the agreement_score you did and what factors influenced your decision in the rationale field.
            In writing the rationale field, instead of "this user", refer to the user with the pronoun "you". Also when referring to the candidate, use the string "the_candidate".

            Example 1:
            user_lines: "I believe in strong environmental regulations to reduce carbon emissions."
            candidate_lines: "We should prioritize the economy over environmental regulations, and focus less on reducing carbon emissions."
            agreement_score: -0.75
            rationale: You strongly disagree with the_candidate on environmental regulation, especially since you prioritize reducing carbon emissions while the_candidate does not. The positions are almost entirely opposed.

            Example 2:
            user_lines: "I believe in a strong military presence to maintain national security."
            candidate_lines: "We should be judicious in how we procure equipment and deploy troops."
            agreement_score: -0.25
            rationale: You and the_candidate have some agreement on the need for a strong military presence, but you prefer a more aggressive approach to maintaining national security while the_candidate prefers a more cautious approach. There is some alignment but some more disagreement.

            Example 3:
            user_lines: "I believe in single payer healthcare for all citizens."
            candidate_lines: "We should have a mixed healthcare system with both private and public options."
            agreement_score: 0.25
            rationale: You and the_candidate have some agreement on healthcare policy, but you prefer a single payer system while the_candidate prefers a mixed system. There is some disagreement but also some more alignment.

            Example 4:
            user_lines: "I believe in reducing taxes to stimulate economic growth."
            candidate_lines: "The productivity of America is being limited by excessive taxation."
            agreement_score: 0.75
            rationale: You and the_candidate are in strong agreement on the need to reduce taxes to stimulate economic growth. Both of you believe that excessive taxation is limiting America's productivity.
            
            These examples use agreement_scores that round to a quarter. You are allowed to use any decimal value between -1 and 1.

            Now, please assess the following:
            user_lines: {user_lines}
            candidate_lines: {candidate_lines}
            """,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json", response_schema=Similarity, temperature=0.05
            ),
        )
        return response


    # List to collect all results
    agreement_scores = []

    # Loop through candidates and query one at a time
    for candidate_lines in candidates_lines:
        result = get_similarity_for_candidate(user_lines, candidate_lines)
        # Convert the response to a string (assuming the result has a `.text` attribute or similar)
        result_text = result.text  # Replace `.text` with the actual method to get the string from the response object

        # Parse the response as JSON (assuming it returns a JSON formatted string)
        try:
            agreement = json.loads(result_text)['agreement_score']
            rationale = json.loads(result_text)['rationale']
            print(agreement)
            print(rationale)
            # Ensure agreement scores are within [-1, 1]
            agreement = max(min(agreement, 1), -1)
            agreement_scores.append(agreement)  # Collect agreement scores
        except json.JSONDecodeError:
            print("Error parsing response from LLM:", result_text)

    print("Agreement scores:", agreement_scores)
    return agreement_scores

candidates = ['Harris', 'Trump', 'Oliver', 'Stein', 'West']
n = len(candidates)
total_agreement_scores = np.zeros(n)
policy_domains = ['Healthcare Policy', 'Domestic Economic Policy', 'Foreign Policy']
# policy_domains = ['Healthcare Policy', 'Domestic Economic Policy', 'Foreign Policy', 'Environmental Policy', 'Immigration Policy', 'Abortion Policy', 'Gun Policy', 'Trade Policy', 'Infrastructure Policy', 'Education Policy', 'Electoral Policy', 'Criminal Policy', 'Social Policy', 'Labor Policy', 'Housing Policy', 'Science Policy', 'Energy Policy']
print("Assign a weight to each policy domain that is an integer, where all the weights sum to 100.")
weights = {}
total_weight = 0
for domain in policy_domains:
    while True:
        try:
            weight = int(input(f"Enter weight for {domain}: "))
            if weight < 0:
                print("Weight must be a non-negative integer.")
                continue
            if total_weight + weight > 100:
                print(f"Total weight cannot exceed 100. Current total: {total_weight}")
                continue
            weights[domain] = weight
            total_weight += weight
            break
        except ValueError:
            print("Please enter a valid integer.")
if total_weight != 100:
    print("The total weight does not sum to 100. Please try again.")
else:
    print("Weights assigned successfully:", weights)

# Create ordered list of policy domains based on weights
ordered_policy_domains = [domain for domain, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True)]
cumulative_weights = 0
# Starting from the policy domain with the highest weight going down, prompt the user to input a paragraph of 50-100 words.
candidate_files = ['Harris_processed.txt', 'Trump_processed.txt', 'Oliver_processed.txt', 'Stein_processed.txt', 'West_processed.txt']
candidates_lines = [None] * len(candidate_files)
for i, candidate_file in enumerate(candidate_files):
    with open(candidate_file, 'r') as f:
        candidates_lines[i] = f.readlines()
for i, policy_domain in enumerate(ordered_policy_domains):
    cumulative_weights += weights[policy_domain]
    print('Current policy domain: ', policy_domain)
    user_lines = input("Please enter a paragraph of less than 100 words related to this policy domain: ")
    # Check if user_lines is under 100 words
    if len(user_lines.split()) >= 100:
        print("The paragraph must be < 100 words. Please try again.")
        continue
    user_lines = user_lines_processor(user_lines)
    print('User generated policy tokenized')
    candidates_lines_selected = tf_idf_multiple(user_lines, candidates_lines)
    print('Relevant candidate lines found')
    print('Relevant candidate lines tokenized')
    print('Requesting agreement scores from LLM')
    agreement = call_llm(user_lines, candidates_lines_selected)
    weighted_agreement = np.array(agreement) * weights[policy_domain] / 100
    total_agreement_scores += weighted_agreement
    print(f'Agreement score for candidates: {agreement}')
    print(f'Cumulative agreement scores: {total_agreement_scores}')
    # Let stop_criteria be the absolute difference between the highest and second highest cumulative agreement scores
    stop_criteria = np.max(total_agreement_scores) - np.sort(total_agreement_scores)[-2]
    # Let sum_remaining_weights be the sum of the weights of the remaining policy domains divided by 100
    sum_remaining_weights = (100 - cumulative_weights) / 100
    # If stop_critera is more than twice as large as sum_remaining_weights, stop the loop
    if stop_criteria > 2 * sum_remaining_weights:
        break
# Print the final agreement scores
print("Final agreement scores:", total_agreement_scores)
# Print the candidate with the highest agreement score
best_candidate_index = np.argmax(total_agreement_scores)
best_candidate = candidates[best_candidate_index]
print(f"The best candidate for you is: {best_candidate}")
    
"""
I support lifelong free public education at all levels, including trade schools and graduate programs, 
and the abolition of student debt. I want to increase and equalize public school funding, end school privatization
, and guarantee free childcare. I’d also reduce taxes for those earning under $75,000 
and strengthen Social Security by removing caps for the wealthy. 
I would close tax loopholes for the rich, implement progressive taxation, 
and ensure affordable utilities through public ownership. I believe in breaking up monopolies, 
banning corporate stock buybacks, and taxing the ultra-wealthy and corporations fairly to foster economic justice.
"""