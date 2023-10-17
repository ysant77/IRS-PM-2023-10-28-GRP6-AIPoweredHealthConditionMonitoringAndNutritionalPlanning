import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
from django.apps import apps

symptoms = apps.get_app_config("diagnosis").symptoms
mlb = apps.get_app_config("diagnosis").mlb
classifier = apps.get_app_config("diagnosis").classifier
nlp = apps.get_app_config("diagnosis").nlp
df_train = apps.get_app_config("diagnosis").dataframe
precautions_df = apps.get_app_config("diagnosis").precautions_df

def get_relevant_symptoms(user_symptoms, top_n=3):
    """
    The following function takes in the user_symptoms as input and gives the relevant symptoms as output.
    Here relevant refers to those symptoms that are most likely to occur with the given set of symptoms
    
    Input: user_symptoms (list of symptoms)
    Output: relevant_symptoms (list of relevant symptoms)

    Example:
    Itching fever cold vomiting headache
    1         1     0     0       1
    0         1     0     1       1

    If user is facing fever and headache then we first compute average of all the above:

    Itching fever cold vomiting headache
    0.5     1     0     0.5     1

    Here after dropping fever and headache the output would be Itching (sorted in ascending order, could have been vomiting either)
    
    """
    #Fetch all the rows from the dataframe where the user_symtpoms are present
    disease_given_symptoms = df_train[df_train[user_symptoms].all(axis=1)]
    
    # Compute the mean proportion for all the symtoms only (only symptoms are numeric 0/1)
    prob_symptoms_given_user_symptoms = disease_given_symptoms.mean(numeric_only=True)

    # Drop the already existing user_symptoms
    prob_symptoms_given_user_symptoms = prob_symptoms_given_user_symptoms.drop(
        user_symptoms, errors="ignore"
    )

    # Sort symptoms based on their average value (which denotes probability in essence) as per descending order and return topn
    relevant_symptoms = (
        prob_symptoms_given_user_symptoms.sort_values(ascending=False)
        .index[:top_n]
        .tolist()
    )

    return relevant_symptoms

def tokenize_and_filter(input_text):
    """
    This function cleans and pre-processes the text entered by user and returns a list of tokens.
    It will first remove stop words and punctuation symbols followed by combining multi-word (two word) symtptoms.

    It resembles the N-gram (bi-gram based approach)

    Input:
    input_text: Text entered by user

    Output:
    refined_tokens: list of tokens (mostly symptoms)

    Ex:
    I have high fever and vomiting

    Output:
    ["high_fever", "vomiting"]
    
    """
    #Load the en_core_web_md model on the lowercased text
    doc = nlp(input_text.lower())

    # Remove the stop words and the punctuation
    tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]

    refined_tokens = []
    i = 0
    while i < len(tokens):
        #this if statement helps to take two word symptoms as a single term, for example mild fever as mild_fever and then skip by 2 positions
        if i < len(tokens) - 1 and "_".join(tokens[i : i + 2]) in symptoms:
            refined_tokens.append("_".join(tokens[i : i + 2]))
            i += 2
        #handles the part of one word symptoms
        else:
            refined_tokens.append(tokens[i])
            i += 1

    return refined_tokens


def calculate_semantic_similarity(token_list1, token_list2):
    """
    This function takes two list of tokens as input and gives the semantic (cosine) similarity between them using the spacy model.

    Input:
    token_list1: list of user symptom tokens
    token_list2: list of possible symptom tokens

    Output:
    semantic_similarity: cosine similarity between the two lists

    The reason for having this function is to capture the possible semantic (synonym) between the user symptom and the possible symptom.
    """
    text1 = " ".join(token_list1)
    text2 = " ".join(token_list2)
    return nlp(text1).similarity(nlp(text2))

def find_best_match(user_symptom, symptom_columns):
    """
    This function takes the individual symptom, and finds the best possible symptom from list of available symptoms.
    It makes use of approximate string matching (fuzzy string matching) and the cosine similarity between the two symptoms.

    Input: 
    user_symptom: individual symptom
    symptom_columns: list of available symptoms

    Output:
    best matching symptom
    """
    best_match = None
    best_fuzzy_score = 0
    best_semantic_score = 0

    #iterate over all symptoms
    for possible_symptom in symptom_columns:
        fuzzy_score = fuzz.ratio(user_symptom, possible_symptom)

        # Only proceed with approximate matches having this threshold. The threshold here is to be adjusted as per feedback from user
        if fuzzy_score > 70: 
            #remove the underscore for extracting multiple words
            semantic_similarity = calculate_semantic_similarity(
                user_symptom.split("_"), possible_symptom.split("_")
            )

            # A weighted sum of fuzzy and semantic similarity
            combined_score = 0.5 * fuzzy_score + 0.5 * semantic_similarity

            #get the best match
            if combined_score > best_semantic_score + best_fuzzy_score:
                best_fuzzy_score = fuzzy_score
                best_semantic_score = semantic_similarity
                best_match = possible_symptom

    return best_match


def extract_symptoms(text, available_symptoms=symptoms):
    """
    This function acts as the coordinator between the tokenize_and_filter and the find_best_match functions.
    It will first get list of tokens from the tokenize_and_filter and then for each token get the best match of symptom from 
    the available symptoms.

    Input:
    text: user input

    Output:
    detected_symptoms: list of symptoms detected from the user input
    """
    #get the set of tokens
    tokens = tokenize_and_filter(text)

    detected_symptoms = []

    for token in tokens:
        #for each token check if there is available symptom or not
        best_match = find_best_match(token, available_symptoms)
        if best_match:
            detected_symptoms.append(best_match)

    return detected_symptoms

def create_multilabel_data(user_symptoms):
    """
    This function just takes the user symptoms and gives a dataframe in the way the pre-trained classifier really needs.
    Input:
    user_symptoms: list of symptoms entered by user (after calling extracted_symptoms and relevant_symptoms)

    Output:
    user_data: A dataframe to be used as input for pre-trained classifier having 0/1
    """
    user_data = pd.DataFrame(columns=symptoms)

    # Set the corresponding columns to 1 based on the user's closest matches
    for closest_match in user_symptoms:
        user_data.loc[0, closest_match] = 1  # Assuming 0 is the row index

    return user_data

def get_top_diseases(probabilities, disease_labels, top_n=3, threshold=0.01):
    """
    Given the model's output probabilities, return the top N diseases which exceed a certain threshold.
    Input:

    probabilities: List or ndarray of predicted probabilities.
    disease_labels: List of disease labels corresponding to indices in probabilities.
    top_n: Number of top diseases to return.
    threshold: Minimum probability threshold to consider a disease.
    Output:

    A list of tuples containing the top N diseases and their probabilities.
    """

    # Ensure probabilities is a numpy array
    probabilities = np.array(probabilities).flatten()

    # Sort diseases by probability
    sorted_indices = np.argsort(probabilities)[::-1]

    top_diseases = []
    count = 0

    for idx in sorted_indices:
        if probabilities[idx] >= threshold:
            top_diseases.append((disease_labels[idx], probabilities[idx]))
            count += 1

        if count == top_n:
            break

    return top_diseases

def perform_diagnosis(symptoms_list):
    """
    This function takes the symptoms_list (returned from extract_symptoms and relevant_symptoms if any)
    Then we use a pre-trained classifier for getting the probabilities of the diseases and we return the final
    top three diseases exceeding a thresold of 1% (0.01)
    The general guidelines/recommendations are also returned along with diseases here
    Input:
    symptoms_list: list of symptoms

    Output:
    top_diseases_with_precautions: Returns upto three 3 diseases, their associated probability and their recommendations
    """
    # Create a dataframe with the user's symptoms
    user_symptoms = create_multilabel_data(symptoms_list)
    #fill Nan values with 0
    user_symptoms = user_symptoms.fillna(0)
    # Predict probabilities for all diseases
    disease_probabilities = classifier.predict_proba(user_symptoms.values)

    # Get the top 3 predicted diseases
    top_diseases = get_top_diseases(
        disease_probabilities, mlb.classes_, top_n=3, threshold=0.01
    )
    top_diseases_with_precautions = []
    for disease, probability in top_diseases:
        disease = disease.strip()
        precautions = precautions_df[precautions_df["Disease"] == disease]
        precautions.fillna(value='',inplace=True)
        probability = round(probability*100, 2)
        top_diseases_with_precautions.append([disease, probability, precautions.iloc[0]["Recommendation 1"], 
                                              precautions.iloc[0]["Recommendation 2"], precautions.iloc[0]["Recommendation 3"], 
                                              precautions.iloc[0]["Recommendation 4"]])
    
    return top_diseases_with_precautions
