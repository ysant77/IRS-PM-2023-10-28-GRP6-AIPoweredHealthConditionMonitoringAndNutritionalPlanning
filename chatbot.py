
import pandas as pd
import numpy as np
import spacy
from fuzzywuzzy import process, fuzz
from joblib import load

nlp = spacy.load("en_core_web_md")


def clean_col_name(col_name):
  col_name = col_name.replace('.1','').replace('(typhos)','').replace('yellowish','yellow').replace('yellowing','yellow')
  return col_name


data_dir = './data/'

df_train = pd.read_csv('{}Training.csv'.format(data_dir))

model_dir = './models/'

classifier_name = '{}onevsrest_svm_classifier.joblib'.format(model_dir)
mlb_name = '{}onevsrest_svm_multilabel_binarizer.joblib'.format(model_dir)

mlb = load(mlb_name)
# Load the trained classifier
classifier = load(classifier_name)


df_train = df_train.drop(columns='Unnamed: 133')

df_train.columns = list(map(clean_col_name, list(df_train.columns)))

symptoms = list(df_train.columns)[:-1]

def get_relevant_symptoms(df, user_symptoms, top_n=3):
    # Compute conditional probabilities for all symptoms given the user's symptoms.
    disease_given_symptoms = df[df[user_symptoms].all(axis=1)]

    # Compute the conditional probability of each symptom given the user's symptoms
    prob_symptoms_given_user_symptoms = disease_given_symptoms.mean()

    # Remove symptoms that the user already mentioned
    prob_symptoms_given_user_symptoms = prob_symptoms_given_user_symptoms.drop(user_symptoms, errors='ignore')

    # Sort symptoms based on their conditional probabilities
    relevant_symptoms = prob_symptoms_given_user_symptoms.sort_values(ascending=False).index[:top_n].tolist()

    return relevant_symptoms

def tokenize_and_filter(input_text):
    doc = nlp(input_text.lower())

    # Exclude stopwords and punctuation
    tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]

    refined_tokens = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and '_'.join(tokens[i:i+2]) in symptoms:
            refined_tokens.append('_'.join(tokens[i:i+2]))
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
            semantic_similarity = calculate_semantic_similarity(user_symptom.split("_"), possible_symptom.split("_"))

            # A weighted sum of fuzzy and semantic similarity
            combined_score = 0.5 * fuzzy_score + 0.5 * semantic_similarity

            if combined_score > best_semantic_score + best_fuzzy_score:
                best_fuzzy_score = fuzzy_score
                best_semantic_score = semantic_similarity
                best_match = possible_symptom

    return best_match

def extract_symptoms(text, available_symptoms):

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


# Function to validate user input
def is_valid_input(detected_symptoms):
    if not detected_symptoms:
        return False, "I'm sorry, I couldn't detect any known symptoms. Can you please rephrase or provide more information?"
    return True, ""

# Function to ask for feedback
def ask_for_feedback():
    feedback = input("Was this information helpful? (yes/no) ")
    if feedback.lower() not in ["yes", "no"]:
        return "Sorry, I didn't catch that. Please answer with 'yes' or 'no'."
    elif feedback.lower() == "no":
        return "Thank you for your feedback. We're continuously working to improve."
    else:
        return "Thank you for your feedback! Stay healthy!"

# Fallback responses
def handle_fallback(user_input):
    non_symptom_phrases = ["hello", "how are you", "thanks", "bye", "who are you", "what can you do"]
    for phrase in non_symptom_phrases:
        if phrase in user_input.lower():
            return True, f"Hello! Please describe your symptoms, and I'll try to help."
    return False, "I'm sorry, I didn't understand that. Please describe your symptoms."

# Disclaimer
def display_disclaimer():
    return ("Please note that this tool's recommendations are based on input data and model training. "
            "It's not a substitute for professional medical advice.")

def chatbot():
    name = input("Hello! What's your name? ")
    print(f"\nNice to meet you, {name}! I'm here to help you understand potential diseases based on your symptoms.")
    while True:

        print(display_disclaimer())
        print("\nEnter your symptoms (e.g. 'I am having fever and cold') or type 'exit' to quit:")
        user_input = input()
        user_input = user_input.lower()

        if user_input == 'exit':
            print("Goodbye {name}. Stay healthy!")
            break
        is_fallback, fallback_message = handle_fallback(user_input)
        if is_fallback:
          print(fallback_message)
          continue

        # Extract symptoms from the user's text
        symptoms_detected = extract_symptoms(user_input, symptoms)
        if not symptoms_detected:
            print("Sorry, I couldn't recognize your symptoms. Try again.")
            continue

        print(f"Detected symptoms: {', '.join(symptoms_detected)}")

        # Suggest additional symptoms
        additional_symptoms = get_relevant_symptoms(df_train, symptoms_detected)
        print(f"Based on your input, you might also have: {', '.join(additional_symptoms)}. Do any of these apply?")

        more_symptoms = input("Enter any additional symptoms from the list above (comma-separated), or press enter to continue: ")
        symptoms_detected.extend([sym.strip() for sym in more_symptoms.split(",") if sym])

        user_symptoms = create_multilabel_data(symptoms_detected)


        user_symptoms = user_symptoms.fillna(0)

        # Make multi-label predictions
        predicted_probs = classifier.predict_proba(user_symptoms.values)

        top_diseases = get_top_diseases(predicted_probs, mlb.classes_, top_n=3, threshold=0.01)
        print(top_diseases)

        print("\nBased on the symptoms, here's the diagnosis (example):\nDisease A: 80% probability, Disease B: 50% probability.")
        print("Consult a doctor/hospital for a comprehensive diagnosis.")

        print("\nWould you like to enter more symptoms or exit? Type 'continue' to keep going or 'exit' to quit.")
        decision = input()
        if decision.lower() == 'exit':
            print("Goodbye {name}. Stay healthy!")
            break

chatbot()

