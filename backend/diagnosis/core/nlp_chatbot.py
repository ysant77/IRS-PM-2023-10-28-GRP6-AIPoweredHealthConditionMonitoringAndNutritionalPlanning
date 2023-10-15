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
    # Compute conditional probabilities for all symptoms given the user's symptoms.
    disease_given_symptoms = df_train[df_train[user_symptoms].all(axis=1)]
    
    #print(disease_given_symptoms)
    # Compute the conditional probability of each symptom given the user's symptoms
    prob_symptoms_given_user_symptoms = disease_given_symptoms.mean(numeric_only=True)

    # Remove symptoms that the user already mentioned
    prob_symptoms_given_user_symptoms = prob_symptoms_given_user_symptoms.drop(
        user_symptoms, errors="ignore"
    )

    # Sort symptoms based on their conditional probabilities
    relevant_symptoms = (
        prob_symptoms_given_user_symptoms.sort_values(ascending=False)
        .index[:top_n]
        .tolist()
    )

    return relevant_symptoms

def tokenize_and_filter(input_text):
    doc = nlp(input_text.lower())

    # Exclude stopwords and punctuation
    tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]

    refined_tokens = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and "_".join(tokens[i : i + 2]) in symptoms:
            refined_tokens.append("_".join(tokens[i : i + 2]))
            i += 2
        else:
            refined_tokens.append(tokens[i])
            i += 1

    return refined_tokens


def calculate_semantic_similarity(token_list1, token_list2):
    text1 = " ".join(token_list1)
    text2 = " ".join(token_list2)
    return nlp(text1).similarity(nlp(text2))

def find_best_match(user_symptom, symptom_columns):
    best_match = None
    best_fuzzy_score = 0
    best_semantic_score = 0

    for possible_symptom in symptom_columns:
        fuzzy_score = fuzz.ratio(user_symptom, possible_symptom)

        # Only proceed with relatively good fuzzy matches
        if fuzzy_score > 70:  # Threshold can be adjusted
            semantic_similarity = calculate_semantic_similarity(
                user_symptom.split("_"), possible_symptom.split("_")
            )

            # A weighted sum of fuzzy and semantic similarity
            combined_score = 0.5 * fuzzy_score + 0.5 * semantic_similarity

            if combined_score > best_semantic_score + best_fuzzy_score:
                best_fuzzy_score = fuzzy_score
                best_semantic_score = semantic_similarity
                best_match = possible_symptom

    return best_match


def extract_symptoms(text, available_symptoms=symptoms):
    tokens = tokenize_and_filter(text)

    detected_symptoms = []

    for token in tokens:
        best_match = find_best_match(token, available_symptoms)
        if best_match:
            detected_symptoms.append(best_match)

    return detected_symptoms

def create_multilabel_data(user_symptoms):
    # Initialize an empty DataFrame with symptom columns
    user_data = pd.DataFrame(columns=symptoms)

    # Set the corresponding columns to 1 based on the user's closest matches
    for closest_match in user_symptoms:
        user_data.loc[0, closest_match] = 1  # Assuming 0 is the row index

    return user_data

def get_top_diseases(probabilities, disease_labels, top_n=3, threshold=0.01):
    """
    Given the model's output probabilities, return the top N diseases which exceed a certain threshold.

    :param probabilities: List or ndarray of predicted probabilities.
    :param disease_labels: List of disease labels corresponding to indices in probabilities.
    :param top_n: Number of top diseases to return.
    :param threshold: Minimum probability threshold to consider a disease.

    :return: A list of tuples containing the top N diseases and their probabilities.
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
    # Create a dataframe with the user's symptoms
    user_symptoms = create_multilabel_data(symptoms_list)
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
