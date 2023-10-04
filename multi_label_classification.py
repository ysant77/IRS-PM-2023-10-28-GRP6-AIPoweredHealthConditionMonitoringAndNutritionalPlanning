
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

df_train = pd.read_csv('/content/drive/MyDrive/IRS_practice_module/Training.csv')
df_test = pd.read_csv('/content/drive/MyDrive/IRS_practice_module/Testing.csv')

df_train = df_train.drop(columns='Unnamed: 133')

df_train.columns[df_train.isnull().any()]

df_test.columns[df_test.isnull().any()]

def clean_col_name(col_name):
  col_name = col_name.replace('.1','').replace('(typhos)','').replace('yellowish','yellow').replace('yellowing','yellow')
  return col_name

df_train.columns = list(map(clean_col_name, list(df_train.columns)))

df_test.columns = list(map(clean_col_name, list(df_test.columns)))

from scipy.stats import chi2_contingency
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

disease_columns = list(df_train['prognosis'])
symptom_columns = list(df_train.columns[:-1])

associations_df = pd.DataFrame(columns=['Symptom', 'Disease', 'Probability'])
disease_column = 'prognosis'
p_disease = df_train[disease_column].value_counts(normalize=True)

# Calculate P(Symptom)
p_symptom = df_train[symptom_columns].mean()

# Create an empty dictionary to store conditional probabilities
disease_given_symptom_dict = {}

# Calculate conditional probabilities for each unique disease
for disease in df_train[disease_column].unique():
    mask = (df_train[disease_column] == disease)
    symptoms_given_disease = df_train[mask][symptom_columns].mean()

    # Calculate P(Disease | Symptom) using Bayes' theorem
    disease_given_symptom = (symptoms_given_disease * p_disease[disease]) / p_symptom

    # Store the result in the dictionary
    disease_given_symptom_dict[disease] = disease_given_symptom

# Create a DataFrame from the dictionary
disease_given_symptom_df = pd.DataFrame(disease_given_symptom_dict).T


"""# 3. OneVsRest Decision Tree Classifier"""

from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.metrics import classification_report, log_loss, roc_auc_score, confusion_matrix, multilabel_confusion_matrix
from sklearn.metrics import roc_curve, auc

X_train = df_train.iloc[:, :-1].values
y_train = df_train.iloc[:, -1].values
X_test = df_test.iloc[:, :-1].values
y_test = df_test.iloc[:, -1].values

mlb_diseases = MultiLabelBinarizer()

all_labels =  pd.concat([df_train['prognosis'], df_test['prognosis']], axis=0)
disease_labels_all = mlb_diseases.fit_transform(all_labels.apply(lambda x: [x]))


y_train_encoded = mlb_diseases.transform(df_train['prognosis'].apply(lambda x: [x]))

y_test_encoded = mlb_diseases.transform(df_test['prognosis'].apply(lambda x: [x]))

base_classifier = DecisionTreeClassifier(random_state=42)

# Create a OneVsRestClassifier to handle multi-label classification
classifier = OneVsRestClassifier(base_classifier)

param_grid = {
    'estimator__max_depth': [10, 20, 30],  # Example hyperparameter values for max_depth
    'estimator__min_samples_split': [2, 5, 10],  # Example hyperparameter values for min_samples_split
}

grid_search = GridSearchCV(classifier, param_grid, cv=10, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train_encoded)

# Get the best estimator from grid search
best_classifier = grid_search.best_estimator_

Y_pred_prob = best_classifier.predict_proba(X_test)

# Calculate the ROC AUC score for each disease label
roc_auc_scores = []
for i in range(y_test_encoded.shape[1]):
    auc_score = roc_auc_score(y_test_encoded[:, i], Y_pred_prob[:, i])
    roc_auc_scores.append(auc_score)

# ROC AUC score for the entire model
roc_auc_avg = roc_auc_score(y_test_encoded, Y_pred_prob, average="weighted")

# Log loss for the model
logloss = log_loss(y_test_encoded, Y_pred_prob)

Y_pred = best_classifier.predict(X_test)
confusion = confusion_matrix(y_test_encoded.argmax(axis=1), Y_pred.argmax(axis=1))

# Classification Report
class_report = classification_report(y_test_encoded.argmax(axis=1), Y_pred.argmax(axis=1))

# Plot ROC curves
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(y_test_encoded.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(y_test_encoded[:, i], Y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve for a specific disease (change the index)
plt.figure()
plt.plot(fpr[0], tpr[0], color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc[0])
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# Print metrics
print(f"ROC AUC Score (Average): {roc_auc_avg:.2f}")
print(f"Log Loss: {logloss:.2f}")
print("Confusion Matrix:")
print(confusion)
print("Classification Report:")
print(class_report)

plt.figure(figsize=(10, 6))
index = 1

plt.title('Validation Curve')
for param_name, param_range in param_grid.items():

  train_scores, valid_scores = validation_curve(best_classifier, X_train, y_train_encoded, param_name=param_name, param_range=param_range, cv=10, scoring='accuracy')

  # Calculate mean and standard deviation of training and validation scores
  train_scores_mean = np.mean(train_scores, axis=1)
  train_scores_std = np.std(train_scores, axis=1)
  valid_scores_mean = np.mean(valid_scores, axis=1)
  valid_scores_std = np.std(valid_scores, axis=1)
  plt.subplot(index, 1, 1)
  index += 1
  plt.xlabel(param_name)
  plt.ylabel('Accuracy')
  plt.grid()
  plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='r')
  plt.fill_between(param_range, valid_scores_mean - valid_scores_std, valid_scores_mean + valid_scores_std, alpha=0.1, color='g')
  plt.plot(param_range, train_scores_mean, 'o-', color='r', label='Training Score')
  plt.plot(param_range, valid_scores_mean, 'o-', color='g', label='Validation Score')
  plt.legend(loc='best')
  plt.show()

"""## Saving the model and label encoders"""

import joblib
dir_name = '/content/drive/MyDrive/models/'
joblib.dump(best_classifier, '{}onevsrest_decision_tree_classifier.joblib'.format(dir_name))

#joblib.dump(label_encoder, '{}onevsrest_decision_tree_label_encoder.joblib'.format(dir_name))

# Save the MultiLabelBinarizer
joblib.dump(mlb_diseases, '{}onevsrest_decision_tree_multilabel_binarizer.joblib'.format(dir_name))

tmp = dict(df_train.iloc[0])
user_input = []
for k, v in tmp.items():
  if k != 'prognosis':
    if v.all() == 1:
      user_input.append(k)


import pandas as pd
import spacy
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from itertools import product

# Load a spaCy model for semantic similarity (you can choose a model like 'en_core_web_md')
nlp = spacy.load('en_core_web_md')

symptom_columns = list(df_train.columns[:-1])
user_symptoms = ["itching", "skin_rash", "joint_pain"]

# Create a DataFrame with the fixed symptom columns
all_possible_symptoms_df = pd.DataFrame(columns=symptom_columns)

# Initialize a list to store the closest matching symptoms
closest_matches = []

# Find the closest matching symptoms for each user symptom
for user_symptom in user_symptoms:
    # Tokenize user's symptom by splitting on underscores
    user_tokens = user_symptom.split("_")

    # Initialize variables to store the best match and its similarity score
    best_match = None
    best_similarity = 0

    # Iterate through all possible symptoms to find the best match
    for possible_symptom in symptom_columns:
        # Tokenize the possible symptom by splitting on underscores
        possible_tokens = possible_symptom.split("_")

        # Calculate semantic similarity using spaCy's similarity score
        user_text = " ".join(user_tokens)
        possible_text = " ".join(possible_tokens)
        semantic_similarity = nlp(user_text).similarity(nlp(possible_text))

        # Calculate syntactic similarity using fuzzy string matching (fuzzywuzzy)
        syntactic_similarity = fuzz.ratio(user_text, possible_text)

        # Calculate an overall similarity score as a combination of semantic and syntactic similarity
        overall_similarity = (semantic_similarity + syntactic_similarity) / 2

        # Update the best match if the current symptom has a higher similarity
        if overall_similarity > best_similarity:
            best_match = possible_symptom
            best_similarity = overall_similarity

    # Add the closest matching symptom to the list
    closest_matches.append(best_match)

# Print the closest matching symptoms for each user symptom
for user_symptom, closest_match in zip(user_symptoms, closest_matches):
    print(f"User Symptom: {user_symptom} | Closest Match: {closest_match}")

# Define a list of different example user symptoms
user_symptoms_list = [
    ["itching", "rashes", "joint_pain"],
    ["fever", "chills", "nausea"],
    ["stomach_pain", "vomiting"],
]

# Create a DataFrame with the fixed symptom columns
all_possible_symptoms_df = pd.DataFrame(columns=symptom_columns)

# Initialize a list to store the closest matching symptoms for each user symptom list
all_closest_matches = []

# Function to calculate semantic similarity using spaCy
def calculate_semantic_similarity(token_list1, token_list2):
    text1 = " ".join(token_list1)
    text2 = " ".join(token_list2)
    return nlp(text1).similarity(nlp(text2))

# Iterate through different example user symptom lists
for user_symptoms in user_symptoms_list:
    # Normalize case (convert to lowercase) for user symptoms
    user_symptoms = [symptom.lower() for symptom in user_symptoms]

    # Initialize a list to store the closest matching symptoms for each user symptom
    closest_matches = []

    # Iterate through user symptoms
    for user_symptom in user_symptoms:
        # Tokenize user's symptom by splitting on underscores
        user_tokens = user_symptom.split("_")

        # Initialize variables to store the best match and its similarity score
        best_match = None
        best_similarity = 0

        # Iterate through all possible symptoms to find the best match
        for possible_symptom in symptom_columns:
            # Tokenize the possible symptom by splitting on underscores
            possible_tokens = possible_symptom.split("_")

            # Calculate semantic similarity using spaCy's similarity score
            semantic_similarity = calculate_semantic_similarity(user_tokens, possible_tokens)

            # Calculate syntactic similarity using fuzzy string matching (fuzzywuzzy)
            syntactic_similarity = fuzz.ratio(user_symptom, possible_symptom)

            # Calculate an overall similarity score as a combination of semantic and syntactic similarity
            overall_similarity = (semantic_similarity + syntactic_similarity) / 2

            # Update the best match if the current symptom has a higher similarity
            if overall_similarity > best_similarity:
                best_match = possible_symptom
                best_similarity = overall_similarity

        # Add the closest matching symptom to the list
        closest_matches.append(best_match)

    # Add the list of closest matching symptoms to the overall list
    all_closest_matches.append(closest_matches)

def create_multilabel_data(user_symptoms):
    # Initialize an empty DataFrame with symptom columns
    user_data = pd.DataFrame(columns=symptom_columns)

    # Set the corresponding columns to 1 based on the user's closest matches
    for closest_match in user_symptoms:
        user_data.loc[0, closest_match] = 1  # Assuming 0 is the row index

    return user_data

classifier_name = '{}onevsrest_decision_tree_classifier.joblib'.format(dir_name)
mlb_name = '{}onevsrest_decision_tree_multilabel_binarizer.joblib'.format(dir_name)

mlb = load(mlb_name)
# Load the trained classifier
classifier = load(classifier_name)


def predict_diseases(user_input):
    user_symptoms = create_multilabel_data(user_input)

    print(user_symptoms)
    user_symptoms = user_symptoms.fillna(0)
    # Make multi-label predictions
    predicted_probs = classifier.predict_proba(user_symptoms.values)

    print(predicted_probs)

    disease_labels = mlb_diseases.inverse_transform(predicted_probs)
    results = {}
    # Create a dictionary to store results
    for i, label in enumerate(disease_labels):
        results[label] = predicted_probs[0][i]

    sorted_results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)}

    return sorted_results

prediction = predict_diseases(user_input)
for disease, probability in prediction.items():
    print(f"Disease: {disease}, Probability: {probability:.2f}")



"""# OneVsRest Logistic Regression"""

mlb_diseases = MultiLabelBinarizer()

all_labels =  pd.concat([df_train['prognosis'], df_test['prognosis']], axis=0)
disease_labels_all = mlb_diseases.fit_transform(all_labels.apply(lambda x: [x]))


y_train_encoded = mlb_diseases.transform(df_train['prognosis'].apply(lambda x: [x]))

y_test_encoded = mlb_diseases.transform(df_test['prognosis'].apply(lambda x: [x]))

from sklearn.linear_model import LogisticRegression

estimator__solvers = ['newton-cg', 'lbfgs', 'liblinear', 'newton-cholesky']
estimator__penalty = ['l2']
base_classifier_logreg = LogisticRegression()
estimator__c_values = [ 0.01, 0.001, 0.0001]

param_grid = dict(estimator__solver=estimator__solvers,estimator__penalty=estimator__penalty,estimator__C=estimator__c_values)

# Create a OneVsRestClassifier to handle multi-label classification
classifier = OneVsRestClassifier(base_classifier_logreg)

grid_search_logreg = GridSearchCV(classifier, param_grid, cv=10, verbose=1, n_jobs=-1)
grid_search_logreg.fit(X_train, y_train_encoded)

# Get the best estimator from grid search
best_classifier_logreg = grid_search_logreg.best_estimator_

Y_pred_prob = best_classifier_logreg.predict_proba(X_test)

# Calculate the ROC AUC score for each disease label
roc_auc_scores = []
for i in range(y_test_encoded.shape[1]):
    auc_score = roc_auc_score(y_test_encoded[:, i], Y_pred_prob[:, i])
    roc_auc_scores.append(auc_score)

# ROC AUC score for the entire model
roc_auc_avg = roc_auc_score(y_test_encoded, Y_pred_prob, average="weighted")

# Log loss for the model
logloss = log_loss(y_test_encoded, Y_pred_prob)

Y_pred = best_classifier_logreg.predict(X_test)
confusion = confusion_matrix(y_test_encoded.argmax(axis=1), Y_pred.argmax(axis=1))

# Classification Report
class_report = classification_report(y_test_encoded.argmax(axis=1), Y_pred.argmax(axis=1))

# Plot ROC curves
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(y_test_encoded.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(y_test_encoded[:, i], Y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve for a specific disease (change the index)
plt.figure()
plt.plot(fpr[0], tpr[0], color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc[0])
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# Print metrics
print(f"ROC AUC Score (Average): {roc_auc_avg:.2f}")
print(f"Log Loss: {logloss:.2f}")
print("Confusion Matrix:")
print(confusion)
print("Classification Report:")
print(class_report)

plt.figure(figsize=(10, 6))
index = 1

plt.title('Validation Curve')
for param_name, param_range in param_grid.items():

  train_scores, valid_scores = validation_curve(best_classifier_logreg, X_train, y_train_encoded, param_name=param_name, param_range=param_range, cv=10, scoring='accuracy')

  # Calculate mean and standard deviation of training and validation scores
  train_scores_mean = np.mean(train_scores, axis=1)
  train_scores_std = np.std(train_scores, axis=1)
  valid_scores_mean = np.mean(valid_scores, axis=1)
  valid_scores_std = np.std(valid_scores, axis=1)
  plt.subplot(index, 1, 1)
  index += 1
  plt.xlabel(param_name)
  plt.ylabel('Accuracy')
  plt.grid()
  plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='r')
  plt.fill_between(param_range, valid_scores_mean - valid_scores_std, valid_scores_mean + valid_scores_std, alpha=0.1, color='g')
  plt.plot(param_range, train_scores_mean, 'o-', color='r', label='Training Score')
  plt.plot(param_range, valid_scores_mean, 'o-', color='g', label='Validation Score')
  plt.legend(loc='best')
  plt.show()



"""## Saving the model and label encoders"""

import joblib
dir_name = '/content/drive/MyDrive/models/'
joblib.dump(best_classifier_logreg, '{}onevsrest_logreg_classifier.joblib'.format(dir_name))

#joblib.dump(label_encoder, '{}onevsrest_decision_tree_label_encoder.joblib'.format(dir_name))

# Save the MultiLabelBinarizer
joblib.dump(mlb_diseases, '{}onevsrest_logreg_multilabel_binarizer.joblib'.format(dir_name))

from joblib import load

classifier_name = '{}onevsrest_logreg_classifier.joblib'.format(dir_name)
mlb_name = '{}onevsrest_logreg_multilabel_binarizer.joblib'.format(dir_name)

mlb = load(mlb_name)
# Load the trained classifier
classifier = load(classifier_name)


def predict_diseases(user_input):
    user_symptoms = create_multilabel_data(user_input)

    print(user_symptoms)
    user_symptoms = user_symptoms.fillna(0)
    # Make multi-label predictions
    predicted_probs = classifier.predict_proba(user_symptoms.values)

    print(predicted_probs)

    #disease_labels = mlb.inverse_transform(predicted_probs)
    results = {}
    # Create a dictionary to store results
    #for i, label in enumerate(disease_labels):
    #    results[label] = predicted_probs[0][i]

    sorted_results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)}

    return sorted_results

prediction = predict_diseases(['itching', 'skin_rash'])
#for disease, probability in prediction.items():
#    print(f"Disease: {disease}, Probability: {probability:.2f}")


#TODO: can make use of logistic regression, add some threshold, and then we can even make use of a subset of symptoms too

"""# OneVsRest Multinomial Naive Bayes"""

mlb_diseases = MultiLabelBinarizer()

all_labels =  pd.concat([df_train['prognosis'], df_test['prognosis']], axis=0)
disease_labels_all = mlb_diseases.fit_transform(all_labels.apply(lambda x: [x]))


y_train_encoded = mlb_diseases.transform(df_train['prognosis'].apply(lambda x: [x]))

y_test_encoded = mlb_diseases.transform(df_test['prognosis'].apply(lambda x: [x]))

from sklearn.naive_bayes import MultinomialNB

base_classifier_nb = MultinomialNB()
alpha = [0.01, 0.1, 0.5, 1.0, 2.0]

param_grid = dict(estimator__alpha=alpha)

# Create a OneVsRestClassifier to handle multi-label classification
classifier = OneVsRestClassifier(base_classifier_nb)

grid_search_nb = GridSearchCV(classifier, param_grid, cv=10, verbose=1, n_jobs=-1)
grid_search_nb.fit(X_train, y_train_encoded)

# Get the best estimator from grid search
best_classifier_nb = grid_search_nb.best_estimator_

Y_pred_prob = best_classifier_nb.predict_proba(X_test)

# Calculate the ROC AUC score for each disease label
roc_auc_scores = []
for i in range(y_test_encoded.shape[1]):
    auc_score = roc_auc_score(y_test_encoded[:, i], Y_pred_prob[:, i])
    roc_auc_scores.append(auc_score)

# ROC AUC score for the entire model
roc_auc_avg = roc_auc_score(y_test_encoded, Y_pred_prob, average="weighted")

# Log loss for the model
logloss = log_loss(y_test_encoded, Y_pred_prob)

Y_pred = best_classifier_nb.predict(X_test)
confusion = confusion_matrix(y_test_encoded.argmax(axis=1), Y_pred.argmax(axis=1))

# Classification Report
class_report = classification_report(y_test_encoded.argmax(axis=1), Y_pred.argmax(axis=1))

# Plot ROC curves
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(y_test_encoded.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(y_test_encoded[:, i], Y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve for a specific disease (change the index)
plt.figure()
plt.plot(fpr[0], tpr[0], color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc[0])
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# Print metrics
print(f"ROC AUC Score (Average): {roc_auc_avg:.2f}")
print(f"Log Loss: {logloss:.2f}")
print("Confusion Matrix:")
print(confusion)
print("Classification Report:")
print(class_report)

plt.figure(figsize=(10, 6))
index = 1

plt.title('Validation Curve')
for param_name, param_range in param_grid.items():

  train_scores, valid_scores = validation_curve(best_classifier_nb, X_train, y_train_encoded, param_name=param_name, param_range=param_range, cv=10, scoring='accuracy')

  # Calculate mean and standard deviation of training and validation scores
  train_scores_mean = np.mean(train_scores, axis=1)
  train_scores_std = np.std(train_scores, axis=1)
  valid_scores_mean = np.mean(valid_scores, axis=1)
  valid_scores_std = np.std(valid_scores, axis=1)
  plt.subplot(index, 1, 1)
  index += 1
  plt.xlabel(param_name)
  plt.ylabel('Accuracy')
  plt.grid()
  plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='r')
  plt.fill_between(param_range, valid_scores_mean - valid_scores_std, valid_scores_mean + valid_scores_std, alpha=0.1, color='g')
  plt.plot(param_range, train_scores_mean, 'o-', color='r', label='Training Score')
  plt.plot(param_range, valid_scores_mean, 'o-', color='g', label='Validation Score')
  plt.legend(loc='best')
  plt.show()

"""## Saving the model and label encoders"""

import joblib
dir_name = '/content/drive/MyDrive/models/'
joblib.dump(best_classifier_nb, '{}onevsrest_naive_bayes_classifier.joblib'.format(dir_name))

#joblib.dump(label_encoder, '{}onevsrest_decision_tree_label_encoder.joblib'.format(dir_name))

# Save the MultiLabelBinarizer
joblib.dump(mlb_diseases, '{}onevsrest_naive_bayes_multilabel_binarizer.joblib'.format(dir_name))

from joblib import load

classifier_name = '{}onevsrest_naive_bayes_classifier.joblib'.format(dir_name)
mlb_name = '{}onevsrest_naive_bayes_multilabel_binarizer.joblib'.format(dir_name)

mlb = load(mlb_name)
# Load the trained classifier
classifier = load(classifier_name)


def predict_diseases(user_input):
    user_symptoms = create_multilabel_data(user_input)

    print(user_symptoms)
    user_symptoms = user_symptoms.fillna(0)
    # Make multi-label predictions
    predicted_probs = classifier.predict_proba(user_symptoms.values)

    print(predicted_probs)

    #disease_labels = mlb.inverse_transform(predicted_probs)
    results = {}
    # Create a dictionary to store results
    #for i, label in enumerate(disease_labels):
    #    results[label] = predicted_probs[0][i]

    sorted_results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)}

    return sorted_results

prediction = predict_diseases(['itching', 'skin_rash'])
#for disease, probability in prediction.items():
#    print(f"Disease: {disease}, Probability: {probability:.2f}")


#TODO: can make use of logistic regression, add some threshold, and then we can even make use of a subset of symptoms too

"""# OneVsRest SVM"""

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

mlb_diseases = MultiLabelBinarizer()

all_labels =  pd.concat([df_train['prognosis'], df_test['prognosis']], axis=0)
disease_labels_all = mlb_diseases.fit_transform(all_labels.apply(lambda x: [x]))


y_train_encoded = mlb_diseases.transform(df_train['prognosis'].apply(lambda x: [x]))

y_test_encoded = mlb_diseases.transform(df_test['prognosis'].apply(lambda x: [x]))

base_classifier_svm = SVC(probability=True)

# Define hyperparameters and their possible values for grid search
param_grid = {
    'estimator__C': [0.1, 1, 10],  # Regularization parameter
    'estimator__kernel': ['linear', 'rbf', 'poly'],  # Kernel types to try
}

classifier = OneVsRestClassifier(base_classifier_svm)

grid_search_svm = GridSearchCV(classifier, param_grid, cv=10, verbose=1, n_jobs=-1)
grid_search_svm.fit(X_train, y_train_encoded)

# Get the best estimator from grid search
best_classifier_svm = grid_search_svm.best_estimator_

Y_pred_prob = best_classifier_svm.predict_proba(X_test)

# Calculate the ROC AUC score for each disease label
roc_auc_scores = []
for i in range(y_test_encoded.shape[1]):
    auc_score = roc_auc_score(y_test_encoded[:, i], Y_pred_prob[:, i])
    roc_auc_scores.append(auc_score)

# ROC AUC score for the entire model
roc_auc_avg = roc_auc_score(y_test_encoded, Y_pred_prob, average="weighted")

# Log loss for the model
logloss = log_loss(y_test_encoded, Y_pred_prob)

Y_pred = best_classifier_svm.predict(X_test)
confusion = confusion_matrix(y_test_encoded.argmax(axis=1), Y_pred.argmax(axis=1))

# Classification Report
class_report = classification_report(y_test_encoded.argmax(axis=1), Y_pred.argmax(axis=1))

# Plot ROC curves
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(y_test_encoded.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(y_test_encoded[:, i], Y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve for a specific disease (change the index)
plt.figure()
plt.plot(fpr[0], tpr[0], color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc[0])
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# Print metrics
print(f"ROC AUC Score (Average): {roc_auc_avg:.2f}")
print(f"Log Loss: {logloss:.2f}")
print("Confusion Matrix:")
print(confusion)
print("Classification Report:")
print(class_report)

plt.figure(figsize=(10, 6))
index = 1

plt.title('Validation Curve')
for param_name, param_range in param_grid.items():

  train_scores, valid_scores = validation_curve(best_classifier_svm, X_train, y_train_encoded, param_name=param_name, param_range=param_range, cv=10, scoring='accuracy')

  # Calculate mean and standard deviation of training and validation scores
  train_scores_mean = np.mean(train_scores, axis=1)
  train_scores_std = np.std(train_scores, axis=1)
  valid_scores_mean = np.mean(valid_scores, axis=1)
  valid_scores_std = np.std(valid_scores, axis=1)
  plt.subplot(index, 1, 1)
  index += 1
  plt.xlabel(param_name)
  plt.ylabel('Accuracy')
  plt.grid()
  plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='r')
  plt.fill_between(param_range, valid_scores_mean - valid_scores_std, valid_scores_mean + valid_scores_std, alpha=0.1, color='g')
  plt.plot(param_range, train_scores_mean, 'o-', color='r', label='Training Score')
  plt.plot(param_range, valid_scores_mean, 'o-', color='g', label='Validation Score')
  plt.legend(loc='best')
  plt.show()

"""## Saving the model and label encoders"""

import joblib
dir_name = '/content/drive/MyDrive/models/'
joblib.dump(best_classifier_svm, '{}onevsrest_svm_classifier.joblib'.format(dir_name))

#joblib.dump(label_encoder, '{}onevsrest_decision_tree_label_encoder.joblib'.format(dir_name))

# Save the MultiLabelBinarizer
joblib.dump(mlb_diseases, '{}onevsrest_svm_multilabel_binarizer.joblib'.format(dir_name))

from joblib import load

classifier_name = '{}onevsrest_svm_classifier.joblib'.format(dir_name)
mlb_name = '{}onevsrest_svm_multilabel_binarizer.joblib'.format(dir_name)

mlb = load(mlb_name)
# Load the trained classifier
classifier = load(classifier_name)


def predict_diseases(user_input):
    user_symptoms = create_multilabel_data(user_input)

    print(user_symptoms)
    user_symptoms = user_symptoms.fillna(0)
    # Make multi-label predictions
    predicted_probs = classifier.predict_proba(user_symptoms.values)

    print(predicted_probs)

    #disease_labels = mlb.inverse_transform(predicted_probs)
    results = {}
    # Create a dictionary to store results
    #for i, label in enumerate(disease_labels):
    #    results[label] = predicted_probs[0][i]

    sorted_results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)}

    return sorted_results

prediction = predict_diseases(['itching', 'skin_rash'])
#for disease, probability in prediction.items():
#    print(f"Disease: {disease}, Probability: {probability:.2f}")


#TODO: can make use of logistic regression, add some threshold, and then we can even make use of a subset of symptoms too



"""# BinaryRelevance Decision tree"""


from skmultilearn.problem_transform import BinaryRelevance

mlb_diseases = MultiLabelBinarizer()

all_labels =  pd.concat([df_train['prognosis'], df_test['prognosis']], axis=0)
#disease_labels_all = mlb_diseases.fit_transform(all_labels.apply(lambda x: [x]))


y_train_encoded = mlb_diseases.fit_transform(df_train['prognosis'].apply(lambda x: [x]))

y_test_encoded = mlb_diseases.transform(df_test['prognosis'].apply(lambda x: [x]))

base_classifier = DecisionTreeClassifier(random_state=42)

# Create a OneVsRestClassifier to handle multi-label classification
classifier = OneVsRestClassifier(base_classifier)

param_grid = {
    'estimator__max_depth': [10, 20, 30],  # Example hyperparameter values for max_depth
    'estimator__min_samples_split': [2, 5, 10],  # Example hyperparameter values for min_samples_split
}

grid_search = GridSearchCV(classifier, param_grid, cv=10, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train_encoded)

# Get the best estimator from grid search
best_classifier = grid_search.best_estimator_

Y_pred_prob = best_classifier.predict_proba(X_test)

# Calculate the ROC AUC score for each disease label
roc_auc_scores = []
for i in range(y_test_encoded.shape[1]):
    auc_score = roc_auc_score(y_test_encoded[:, i], Y_pred_prob[:, i])
    roc_auc_scores.append(auc_score)

# ROC AUC score for the entire model
roc_auc_avg = roc_auc_score(y_test_encoded, Y_pred_prob, average="weighted")

# Log loss for the model
logloss = log_loss(y_test_encoded, Y_pred_prob)

Y_pred = best_classifier.predict(X_test)
confusion = confusion_matrix(y_test_encoded.argmax(axis=1), Y_pred.argmax(axis=1))

# Classification Report
class_report = classification_report(y_test_encoded.argmax(axis=1), Y_pred.argmax(axis=1))

# Plot ROC curves
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(y_test_encoded.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(y_test_encoded[:, i], Y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve for a specific disease (change the index)
plt.figure()
plt.plot(fpr[0], tpr[0], color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc[0])
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# Print metrics
print(f"ROC AUC Score (Average): {roc_auc_avg:.2f}")
print(f"Log Loss: {logloss:.2f}")
print("Confusion Matrix:")
print(confusion)
print("Classification Report:")
print(class_report)

plt.figure(figsize=(10, 6))
index = 1

plt.title('Validation Curve')
for param_name, param_range in param_grid.items():

  train_scores, valid_scores = validation_curve(best_classifier, X_train, y_train_encoded, param_name=param_name, param_range=param_range, cv=10, scoring='accuracy')

  # Calculate mean and standard deviation of training and validation scores
  train_scores_mean = np.mean(train_scores, axis=1)
  train_scores_std = np.std(train_scores, axis=1)
  valid_scores_mean = np.mean(valid_scores, axis=1)
  valid_scores_std = np.std(valid_scores, axis=1)
  plt.subplot(index, 1, 1)
  index += 1
  plt.xlabel(param_name)
  plt.ylabel('Accuracy')
  plt.grid()
  plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='r')
  plt.fill_between(param_range, valid_scores_mean - valid_scores_std, valid_scores_mean + valid_scores_std, alpha=0.1, color='g')
  plt.plot(param_range, train_scores_mean, 'o-', color='r', label='Training Score')
  plt.plot(param_range, valid_scores_mean, 'o-', color='g', label='Validation Score')
  plt.legend(loc='best')
  plt.show()

"""## Saving the model and label encoders"""

import joblib
dir_name = '/content/drive/MyDrive/models/'
joblib.dump(best_classifier, '{}binaryrelevance_decision_tree_classifier.joblib'.format(dir_name))

#joblib.dump(label_encoder, '{}onevsrest_decision_tree_label_encoder.joblib'.format(dir_name))

# Save the MultiLabelBinarizer
joblib.dump(mlb_diseases, '{}binaryrelevance_decision_tree_multilabel_binarizer.joblib'.format(dir_name))

classifier_name = '{}binaryrelevance_decision_tree_classifier.joblib'.format(dir_name)
mlb_name = '{}binaryrelevance_decision_tree_multilabel_binarizer.joblib'.format(dir_name)

mlb = load(mlb_name)
# Load the trained classifier
classifier = load(classifier_name)


def predict_diseases(user_input):
    user_symptoms = create_multilabel_data(user_input)

    print(user_symptoms)
    user_symptoms = user_symptoms.fillna(0)
    # Make multi-label predictions
    predicted_probs = classifier.predict_proba(user_symptoms.values)

    print(predicted_probs)

    disease_labels = mlb_diseases.inverse_transform(predicted_probs)
    results = {}
    # Create a dictionary to store results
    for i, label in enumerate(disease_labels):
        results[label] = predicted_probs[0][i]

    sorted_results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)}

    return sorted_results

prediction = predict_diseases(user_input)
for disease, probability in prediction.items():
    print(f"Disease: {disease}, Probability: {probability:.2f}")



"""# BinaryRelevance Logistic Regression"""

mlb_diseases = MultiLabelBinarizer()

all_labels =  pd.concat([df_train['prognosis'], df_test['prognosis']], axis=0)
disease_labels_all = mlb_diseases.fit_transform(all_labels.apply(lambda x: [x]))


y_train_encoded = mlb_diseases.transform(df_train['prognosis'].apply(lambda x: [x]))

y_test_encoded = mlb_diseases.transform(df_test['prognosis'].apply(lambda x: [x]))

from sklearn.linear_model import LogisticRegression

estimator__solvers = ['newton-cg', 'lbfgs', 'liblinear', 'newton-cholesky']
estimator__penalty = ['l2']
base_classifier_logreg = LogisticRegression()
estimator__c_values = [ 0.01, 0.001, 0.0001]

param_grid = dict(classifier__solver=estimator__solvers,  classifier__penalty=estimator__penalty,  classifier__C=estimator__c_values)

# Create a OneVsRestClassifier to handle multi-label classification
classifier = BinaryRelevance(base_classifier_logreg)

grid_search_logreg = GridSearchCV(classifier, param_grid, cv=10, verbose=1, n_jobs=-1)
grid_search_logreg.fit(X_train, y_train_encoded)

# Get the best estimator from grid search
best_classifier_logreg = grid_search_logreg.best_estimator_

Y_pred_prob = best_classifier_logreg.predict_proba(X_test)


# Calculate the ROC AUC score for each disease label
Y_pred_prob_dense = Y_pred_prob.toarray()



roc_auc_scores = []

for i in range(y_test_encoded.shape[1]):
    auc_score = roc_auc_score(y_test_encoded[:, i], Y_pred_prob_dense[:, i])
    roc_auc_scores.append(auc_score)
# ROC AUC score for the entire model
roc_auc_avg = roc_auc_score(y_test_encoded, Y_pred_prob_dense, average="weighted")

# Log loss for the model
logloss = log_loss(y_test_encoded, Y_pred_prob_dense)

Y_pred = best_classifier_logreg.predict(X_test)
Y_pred = Y_pred.toarray()


confusion = confusion_matrix(y_test_encoded.argmax(axis=1), Y_pred.argmax(axis=1))

# Classification Report
class_report = classification_report(y_test_encoded.argmax(axis=1), Y_pred.argmax(axis=1))

# Plot ROC curves
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(y_test_encoded.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(y_test_encoded[:, i], Y_pred_prob_dense[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve for a specific disease (change the index)
plt.figure()
plt.plot(fpr[0], tpr[0], color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc[0])
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# Print metrics
print(f"ROC AUC Score (Average): {roc_auc_avg:.2f}")
print(f"Log Loss: {logloss:.2f}")
print("Confusion Matrix:")
print(confusion)
print("Classification Report:")
print(class_report)

plt.figure(figsize=(10, 6))
index = 1

plt.title('Validation Curve')
for param_name, param_range in param_grid.items():

  train_scores, valid_scores = validation_curve(best_classifier_logreg, X_train, y_train_encoded, param_name=param_name, param_range=param_range, cv=10, scoring='accuracy')

  # Calculate mean and standard deviation of training and validation scores
  train_scores_mean = np.mean(train_scores, axis=1)
  train_scores_std = np.std(train_scores, axis=1)
  valid_scores_mean = np.mean(valid_scores, axis=1)
  valid_scores_std = np.std(valid_scores, axis=1)
  plt.subplot(index, 1, 1)
  index += 1
  plt.xlabel(param_name)
  plt.ylabel('Accuracy')
  plt.grid()
  plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='r')
  plt.fill_between(param_range, valid_scores_mean - valid_scores_std, valid_scores_mean + valid_scores_std, alpha=0.1, color='g')
  plt.plot(param_range, train_scores_mean, 'o-', color='r', label='Training Score')
  plt.plot(param_range, valid_scores_mean, 'o-', color='g', label='Validation Score')
  plt.legend(loc='best')
  plt.show()

"""## Saving the model and label encoders"""

import joblib
dir_name = '/content/drive/MyDrive/models/'
joblib.dump(best_classifier_logreg, '{}binaryrelevance_logreg_classifier.joblib'.format(dir_name))

#joblib.dump(label_encoder, '{}onevsrest_decision_tree_label_encoder.joblib'.format(dir_name))

# Save the MultiLabelBinarizer
joblib.dump(mlb_diseases, '{}binaryrelevance_logreg_multilabel_binarizer.joblib'.format(dir_name))

classifier_name = '{}binaryrelevance_logreg_classifier.joblib'.format(dir_name)
mlb_name = '{}binaryrelevance_logreg_multilabel_binarizer.joblib'.format(dir_name)

mlb = load(mlb_name)
# Load the trained classifier
classifier = load(classifier_name)


def predict_diseases(user_input):
    user_symptoms = create_multilabel_data(user_input)

    print(user_symptoms)
    user_symptoms = user_symptoms.fillna(0)
    # Make multi-label predictions
    predicted_probs = classifier.predict_proba(user_symptoms.values)

    print(predicted_probs)

    #disease_labels = mlb.inverse_transform(predicted_probs)
    results = {}
    # Create a dictionary to store results
    #for i, label in enumerate(disease_labels):
    #    results[label] = predicted_probs[0][i]

    sorted_results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)}

    return sorted_results

prediction = predict_diseases(['itching', 'skin_rash'])
#for disease, probability in prediction.items():
#    print(f"Disease: {disease}, Probability: {probability:.2f}")

"""# BinaryRelevance Multinomial Naive Bayes"""

mlb_diseases = MultiLabelBinarizer()

all_labels =  pd.concat([df_train['prognosis'], df_test['prognosis']], axis=0)
disease_labels_all = mlb_diseases.fit_transform(all_labels.apply(lambda x: [x]))


y_train_encoded = mlb_diseases.transform(df_train['prognosis'].apply(lambda x: [x]))

y_test_encoded = mlb_diseases.transform(df_test['prognosis'].apply(lambda x: [x]))

from sklearn.naive_bayes import MultinomialNB

base_classifier_nb = MultinomialNB()
alpha = [0.01, 0.1, 0.5, 1.0, 2.0]

param_grid = dict(estimator__alpha=alpha)

# Create a OneVsRestClassifier to handle multi-label classification
classifier = OneVsRestClassifier(base_classifier_nb)

grid_search_nb = GridSearchCV(classifier, param_grid, cv=10, verbose=1, n_jobs=-1)
grid_search_nb.fit(X_train, y_train_encoded)

# Get the best estimator from grid search
best_classifier_nb = grid_search_nb.best_estimator_

Y_pred_prob = best_classifier_nb.predict_proba(X_test)

# Calculate the ROC AUC score for each disease label
roc_auc_scores = []
for i in range(y_test_encoded.shape[1]):
    auc_score = roc_auc_score(y_test_encoded[:, i], Y_pred_prob[:, i])
    roc_auc_scores.append(auc_score)

# ROC AUC score for the entire model
roc_auc_avg = roc_auc_score(y_test_encoded, Y_pred_prob, average="weighted")

# Log loss for the model
logloss = log_loss(y_test_encoded, Y_pred_prob)

Y_pred = best_classifier_nb.predict(X_test)
confusion = confusion_matrix(y_test_encoded.argmax(axis=1), Y_pred.argmax(axis=1))

# Classification Report
class_report = classification_report(y_test_encoded.argmax(axis=1), Y_pred.argmax(axis=1))

# Plot ROC curves
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(y_test_encoded.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(y_test_encoded[:, i], Y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve for a specific disease (change the index)
plt.figure()
plt.plot(fpr[0], tpr[0], color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc[0])
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# Print metrics
print(f"ROC AUC Score (Average): {roc_auc_avg:.2f}")
print(f"Log Loss: {logloss:.2f}")
print("Confusion Matrix:")
print(confusion)
print("Classification Report:")
print(class_report)

plt.figure(figsize=(10, 6))
index = 1


plt.title('Validation Curve')
for param_name, param_range in param_grid.items():

  train_scores, valid_scores = validation_curve(best_classifier_nb, X_train, y_train_encoded, param_name=param_name, param_range=param_range, cv=10, scoring='accuracy')

  # Calculate mean and standard deviation of training and validation scores
  train_scores_mean = np.mean(train_scores, axis=1)
  train_scores_std = np.std(train_scores, axis=1)
  valid_scores_mean = np.mean(valid_scores, axis=1)
  valid_scores_std = np.std(valid_scores, axis=1)
  plt.subplot(index, 1, 1)
  index += 1
  plt.xlabel(param_name)
  plt.ylabel('Accuracy')
  plt.grid()
  plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='r')
  plt.fill_between(param_range, valid_scores_mean - valid_scores_std, valid_scores_mean + valid_scores_std, alpha=0.1, color='g')
  plt.plot(param_range, train_scores_mean, 'o-', color='r', label='Training Score')
  plt.plot(param_range, valid_scores_mean, 'o-', color='g', label='Validation Score')
  plt.legend(loc='best')
  plt.show()

"""## Saving the model and label encoders"""

import joblib
dir_name = '/content/drive/MyDrive/models/'
joblib.dump(best_classifier_nb, '{}binaryrelevance_naive_bayes_classifier.joblib'.format(dir_name))

#joblib.dump(label_encoder, '{}onevsrest_decision_tree_label_encoder.joblib'.format(dir_name))

# Save the MultiLabelBinarizer
joblib.dump(mlb_diseases, '{}binaryrelevance_naive_bayes_multilabel_binarizer.joblib'.format(dir_name))

from joblib import load

classifier_name = '{}binaryrelevance_naive_bayes_classifier.joblib'.format(dir_name)
mlb_name = '{}binaryrelevance_naive_bayes_multilabel_binarizer.joblib'.format(dir_name)

mlb = load(mlb_name)
# Load the trained classifier
classifier = load(classifier_name)


def predict_diseases(user_input):
    user_symptoms = create_multilabel_data(user_input)

    print(user_symptoms)
    user_symptoms = user_symptoms.fillna(0)
    # Make multi-label predictions
    predicted_probs = classifier.predict_proba(user_symptoms.values)

    print(predicted_probs)

    #disease_labels = mlb.inverse_transform(predicted_probs)
    results = {}
    # Create a dictionary to store results
    #for i, label in enumerate(disease_labels):
    #    results[label] = predicted_probs[0][i]

    sorted_results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)}

    return sorted_results

prediction = predict_diseases(['itching', 'skin_rash'])
#for disease, probability in prediction.items():
#    print(f"Disease: {disease}, Probability: {probability:.2f}")


#TODO: can make use of logistic regression, add some threshold, and then we can even make use of a subset of symptoms too

"""# BinaryRelevance SVM"""

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

mlb_diseases = MultiLabelBinarizer()

all_labels =  pd.concat([df_train['prognosis'], df_test['prognosis']], axis=0)
disease_labels_all = mlb_diseases.fit_transform(all_labels.apply(lambda x: [x]))


y_train_encoded = mlb_diseases.transform(df_train['prognosis'].apply(lambda x: [x]))

y_test_encoded = mlb_diseases.transform(df_test['prognosis'].apply(lambda x: [x]))

base_classifier_svm = SVC(probability=True)

# Define hyperparameters and their possible values for grid search
param_grid = {
    'estimator__C': [0.1, 1, 10],  # Regularization parameter
    'estimator__kernel': ['linear', 'rbf', 'poly'],  # Kernel types to try
}

classifier = OneVsRestClassifier(base_classifier_svm)

grid_search_svm = GridSearchCV(classifier, param_grid, cv=10, verbose=1, n_jobs=-1)
grid_search_svm.fit(X_train, y_train_encoded)

# Get the best estimator from grid search
best_classifier_svm = grid_search_svm.best_estimator_

Y_pred_prob = best_classifier_svm.predict_proba(X_test)

# Calculate the ROC AUC score for each disease label
roc_auc_scores = []
for i in range(y_test_encoded.shape[1]):
    auc_score = roc_auc_score(y_test_encoded[:, i], Y_pred_prob[:, i])
    roc_auc_scores.append(auc_score)

# ROC AUC score for the entire model
roc_auc_avg = roc_auc_score(y_test_encoded, Y_pred_prob, average="weighted")

# Log loss for the model
logloss = log_loss(y_test_encoded, Y_pred_prob)

Y_pred = best_classifier_svm.predict(X_test)
confusion = confusion_matrix(y_test_encoded.argmax(axis=1), Y_pred.argmax(axis=1))

# Classification Report
class_report = classification_report(y_test_encoded.argmax(axis=1), Y_pred.argmax(axis=1))

# Plot ROC curves
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(y_test_encoded.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(y_test_encoded[:, i], Y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve for a specific disease (change the index)
plt.figure()
plt.plot(fpr[0], tpr[0], color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc[0])
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# Print metrics
print(f"ROC AUC Score (Average): {roc_auc_avg:.2f}")
print(f"Log Loss: {logloss:.2f}")
print("Confusion Matrix:")
print(confusion)
print("Classification Report:")
print(class_report)

plt.figure(figsize=(10, 6))
index = 1

plt.title('Validation Curve')
for param_name, param_range in param_grid.items():

  train_scores, valid_scores = validation_curve(best_classifier_svm, X_train, y_train_encoded, param_name=param_name, param_range=param_range, cv=10, scoring='accuracy')

  # Calculate mean and standard deviation of training and validation scores
  train_scores_mean = np.mean(train_scores, axis=1)
  train_scores_std = np.std(train_scores, axis=1)
  valid_scores_mean = np.mean(valid_scores, axis=1)
  valid_scores_std = np.std(valid_scores, axis=1)
  plt.subplot(index, 1, 1)
  index += 1
  plt.xlabel(param_name)
  plt.ylabel('Accuracy')
  plt.grid()
  plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='r')
  plt.fill_between(param_range, valid_scores_mean - valid_scores_std, valid_scores_mean + valid_scores_std, alpha=0.1, color='g')
  plt.plot(param_range, train_scores_mean, 'o-', color='r', label='Training Score')
  plt.plot(param_range, valid_scores_mean, 'o-', color='g', label='Validation Score')
  plt.legend(loc='best')
  plt.show()

"""## Saving the model and label encoders"""

import joblib
dir_name = '/content/drive/MyDrive/models/'
joblib.dump(best_classifier_svm, '{}binaryrelevance_svm_classifier.joblib'.format(dir_name))

#joblib.dump(label_encoder, '{}onevsrest_decision_tree_label_encoder.joblib'.format(dir_name))

# Save the MultiLabelBinarizer
joblib.dump(mlb_diseases, '{}binaryrelevance_svm_multilabel_binarizer.joblib'.format(dir_name))

from joblib import load

classifier_name = '{}binaryrelevance_svm_classifier.joblib'.format(dir_name)
mlb_name = '{}binaryrelevance_svm_multilabel_binarizer.joblib'.format(dir_name)

mlb = load(mlb_name)
# Load the trained classifier
classifier = load(classifier_name)


def predict_diseases(user_input):
    user_symptoms = create_multilabel_data(user_input)

    print(user_symptoms)
    user_symptoms = user_symptoms.fillna(0)
    # Make multi-label predictions
    predicted_probs = classifier.predict_proba(user_symptoms.values)

    print(predicted_probs)

    #disease_labels = mlb.inverse_transform(predicted_probs)
    results = {}
    # Create a dictionary to store results
    #for i, label in enumerate(disease_labels):
    #    results[label] = predicted_probs[0][i]

    sorted_results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)}

    return sorted_results

prediction = predict_diseases(['itching', 'skin_rash'])
#for disease, probability in prediction.items():
#    print(f"Disease: {disease}, Probability: {probability:.2f}")


#TODO: can make use of logistic regression, add some threshold, and then we can even make use of a subset of symptoms too

"""# ClassifierChain Decision Tree"""

from skmultilearn.problem_transform import ClassifierChain

mlb_diseases = MultiLabelBinarizer()

all_labels =  pd.concat([df_train['prognosis'], df_test['prognosis']], axis=0)
disease_labels_all = mlb_diseases.fit_transform(all_labels.apply(lambda x: [x]))


y_train_encoded = mlb_diseases.transform(df_train['prognosis'].apply(lambda x: [x]))

y_test_encoded = mlb_diseases.transform(df_test['prognosis'].apply(lambda x: [x]))

base_classifier = DecisionTreeClassifier(random_state=42)

# Create a OneVsRestClassifier to handle multi-label classification
classifier = ClassifierChain(base_classifier)

param_grid = {
    'classifier__max_depth': [10, 20, 30],  # Example hyperparameter values for max_depth
    'classifier__min_samples_split': [2, 5, 10],  # Example hyperparameter values for min_samples_split
}

grid_search = GridSearchCV(classifier, param_grid, cv=10, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train_encoded)

# Get the best estimator from grid search
best_classifier = grid_search.best_estimator_

Y_pred_prob = best_classifier.predict_proba(X_test)
Y_pred_prob = Y_pred_prob.toarray()
# Calculate the ROC AUC score for each disease label
roc_auc_scores = []
for i in range(y_test_encoded.shape[1]):
    auc_score = roc_auc_score(y_test_encoded[:, i], Y_pred_prob[:, i])
    roc_auc_scores.append(auc_score)

# ROC AUC score for the entire model
roc_auc_avg = roc_auc_score(y_test_encoded, Y_pred_prob, average="weighted")

# Log loss for the model
logloss = log_loss(y_test_encoded, Y_pred_prob)

Y_pred = best_classifier.predict(X_test)
Y_pred = Y_pred.toarray()

confusion = confusion_matrix(y_test_encoded.argmax(axis=1), Y_pred.argmax(axis=1))

# Classification Report
class_report = classification_report(y_test_encoded.argmax(axis=1), Y_pred.argmax(axis=1))

# Plot ROC curves
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(y_test_encoded.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(y_test_encoded[:, i], Y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve for a specific disease (change the index)
plt.figure()
plt.plot(fpr[0], tpr[0], color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc[0])
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# Print metrics
print(f"ROC AUC Score (Average): {roc_auc_avg:.2f}")
print(f"Log Loss: {logloss:.2f}")
print("Confusion Matrix:")
print(confusion)
print("Classification Report:")
print(class_report)

plt.figure(figsize=(10, 6))
index = 1

plt.title('Validation Curve')
for param_name, param_range in param_grid.items():

  train_scores, valid_scores = validation_curve(best_classifier, X_train, y_train_encoded, param_name=param_name, param_range=param_range, cv=10, scoring='accuracy')

  # Calculate mean and standard deviation of training and validation scores
  train_scores_mean = np.mean(train_scores, axis=1)
  train_scores_std = np.std(train_scores, axis=1)
  valid_scores_mean = np.mean(valid_scores, axis=1)
  valid_scores_std = np.std(valid_scores, axis=1)
  plt.subplot(index, 1, 1)
  index += 1
  plt.xlabel(param_name)
  plt.ylabel('Accuracy')
  plt.grid()
  plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='r')
  plt.fill_between(param_range, valid_scores_mean - valid_scores_std, valid_scores_mean + valid_scores_std, alpha=0.1, color='g')
  plt.plot(param_range, train_scores_mean, 'o-', color='r', label='Training Score')
  plt.plot(param_range, valid_scores_mean, 'o-', color='g', label='Validation Score')
  plt.legend(loc='best')
  plt.show()



"""## Saving the model and label encoders"""

import joblib
dir_name = '/content/drive/MyDrive/models/'
joblib.dump(best_classifier, '{}classifierchain_decision_tree_classifier.joblib'.format(dir_name))

#joblib.dump(label_encoder, '{}onevsrest_decision_tree_label_encoder.joblib'.format(dir_name))

# Save the MultiLabelBinarizer
joblib.dump(mlb_diseases, '{}classifierchain_decision_tree_multilabel_binarizer.joblib'.format(dir_name))

classifier_name = '{}classifierchain_decision_tree_classifier.joblib'.format(dir_name)
mlb_name = '{}classifierchain_decision_tree_multilabel_binarizer.joblib'.format(dir_name)

mlb = load(mlb_name)
# Load the trained classifier
classifier = load(classifier_name)


def predict_diseases(user_input):
    user_symptoms = create_multilabel_data(user_input)

    user_symptoms = user_symptoms.fillna(0)

    print(user_symptoms.values)
    # Make multi-label predictions
    predicted_probs = classifier.predict_proba(user_symptoms.values[0])
    predicted_probs = predicted_probs.toarray()
    print(predicted_probs)

    disease_labels = mlb_diseases.inverse_transform(predicted_probs)
    results = {}
    # Create a dictionary to store results
    for i, label in enumerate(disease_labels):
        results[label] = predicted_probs[0][i]

    sorted_results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)}

    return sorted_results

prediction = predict_diseases(user_input)
for disease, probability in prediction.items():
    print(f"Disease: {disease}, Probability: {probability:.2f}")

"""# ClassifierChain Logistic Regression"""

mlb_diseases = MultiLabelBinarizer()

all_labels =  pd.concat([df_train['prognosis'], df_test['prognosis']], axis=0)
disease_labels_all = mlb_diseases.fit_transform(all_labels.apply(lambda x: [x]))


y_train_encoded = mlb_diseases.transform(df_train['prognosis'].apply(lambda x: [x]))

y_test_encoded = mlb_diseases.transform(df_test['prognosis'].apply(lambda x: [x]))

from sklearn.linear_model import LogisticRegression

estimator__solvers = ['newton-cg', 'lbfgs', 'liblinear', 'newton-cholesky']
estimator__penalty = ['l2']
base_classifier_logreg = LogisticRegression()
estimator__c_values = [ 0.01, 0.001, 0.0001]

param_grid = dict(classifier__solver=estimator__solvers,classifier__penalty=estimator__penalty,classifier__C=estimator__c_values)

# Create a OneVsRestClassifier to handle multi-label classification
classifier = ClassifierChain(base_classifier_logreg)

grid_search_logreg = GridSearchCV(classifier, param_grid, cv=10, verbose=1, n_jobs=-1)
grid_search_logreg.fit(X_train, y_train_encoded)

# Get the best estimator from grid search
best_classifier_logreg = grid_search_logreg.best_estimator_

Y_pred_prob = best_classifier_logreg.predict_proba(X_test)
Y_pred_prob = Y_pred_prob.toarray()
# Calculate the ROC AUC score for each disease label
roc_auc_scores = []
for i in range(y_test_encoded.shape[1]):
    auc_score = roc_auc_score(y_test_encoded[:, i], Y_pred_prob[:, i])
    roc_auc_scores.append(auc_score)

# ROC AUC score for the entire model
roc_auc_avg = roc_auc_score(y_test_encoded, Y_pred_prob, average="weighted")

# Log loss for the model
logloss = log_loss(y_test_encoded, Y_pred_prob)

Y_pred = best_classifier_logreg.predict(X_test)
Y_pred = Y_pred.toarray()
confusion = confusion_matrix(y_test_encoded.argmax(axis=1), Y_pred.argmax(axis=1))

# Classification Report
class_report = classification_report(y_test_encoded.argmax(axis=1), Y_pred.argmax(axis=1))

# Plot ROC curves
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(y_test_encoded.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(y_test_encoded[:, i], Y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve for a specific disease (change the index)
plt.figure()
plt.plot(fpr[0], tpr[0], color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc[0])
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# Print metrics
print(f"ROC AUC Score (Average): {roc_auc_avg:.2f}")
print(f"Log Loss: {logloss:.2f}")
print("Confusion Matrix:")
print(confusion)
print("Classification Report:")
print(class_report)

plt.figure(figsize=(10, 6))
index = 1

plt.title('Validation Curve')
for param_name, param_range in param_grid.items():

  train_scores, valid_scores = validation_curve(best_classifier_logreg, X_train, y_train_encoded, param_name=param_name, param_range=param_range, cv=10, scoring='accuracy')

  # Calculate mean and standard deviation of training and validation scores
  train_scores_mean = np.mean(train_scores, axis=1)
  train_scores_std = np.std(train_scores, axis=1)
  valid_scores_mean = np.mean(valid_scores, axis=1)
  valid_scores_std = np.std(valid_scores, axis=1)
  plt.subplot(index, 1, 1)
  index += 1
  plt.xlabel(param_name)
  plt.ylabel('Accuracy')
  plt.grid()
  plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='r')
  plt.fill_between(param_range, valid_scores_mean - valid_scores_std, valid_scores_mean + valid_scores_std, alpha=0.1, color='g')
  plt.plot(param_range, train_scores_mean, 'o-', color='r', label='Training Score')
  plt.plot(param_range, valid_scores_mean, 'o-', color='g', label='Validation Score')
  plt.legend(loc='best')
  plt.show()

"""## Saving the model and label encoders"""

import joblib
dir_name = '/content/drive/MyDrive/models/'
joblib.dump(best_classifier_logreg, '{}classifierchain_logreg_classifier.joblib'.format(dir_name))

#joblib.dump(label_encoder, '{}onevsrest_decision_tree_label_encoder.joblib'.format(dir_name))

# Save the MultiLabelBinarizer
joblib.dump(mlb_diseases, '{}classifierchain_logreg_multilabel_binarizer.joblib'.format(dir_name))

from joblib import load

classifier_name = '{}classifierchain_logreg_classifier.joblib'.format(dir_name)
mlb_name = '{}classifierchain_logreg_multilabel_binarizer.joblib'.format(dir_name)

mlb = load(mlb_name)
# Load the trained classifier
classifier = load(classifier_name)


def predict_diseases(user_input):
    user_symptoms = create_multilabel_data(user_input)

    print(user_symptoms)
    user_symptoms = user_symptoms.fillna(0)
    # Make multi-label predictions
    predicted_probs = classifier.predict_proba(user_symptoms.values)

    print(predicted_probs)

    #disease_labels = mlb.inverse_transform(predicted_probs)
    results = {}
    # Create a dictionary to store results
    #for i, label in enumerate(disease_labels):
    #    results[label] = predicted_probs[0][i]

    sorted_results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)}

    return sorted_results

prediction = predict_diseases(['itching', 'skin_rash'])
#for disease, probability in prediction.items():
#    print(f"Disease: {disease}, Probability: {probability:.2f}")


#TODO: can make use of logistic regression, add some threshold, and then we can even make use of a subset of symptoms too

"""# ClassifierChain Multinomial Naive Bayes"""

mlb_diseases = MultiLabelBinarizer()

all_labels =  pd.concat([df_train['prognosis'], df_test['prognosis']], axis=0)
disease_labels_all = mlb_diseases.fit_transform(all_labels.apply(lambda x: [x]))


y_train_encoded = mlb_diseases.transform(df_train['prognosis'].apply(lambda x: [x]))

y_test_encoded = mlb_diseases.transform(df_test['prognosis'].apply(lambda x: [x]))

from sklearn.naive_bayes import MultinomialNB

base_classifier_nb = MultinomialNB()
alpha = [0.01, 0.1, 0.5, 1.0, 2.0]

param_grid = dict(classifier__alpha=alpha)

# Create a OneVsRestClassifier to handle multi-label classification
classifier = ClassifierChain(base_classifier_nb)

grid_search_nb = GridSearchCV(classifier, param_grid, cv=10, verbose=1, n_jobs=-1)
grid_search_nb.fit(X_train, y_train_encoded)

# Get the best estimator from grid search
best_classifier_nb = grid_search_nb.best_estimator_

Y_pred_prob = best_classifier_nb.predict_proba(X_test)
Y_pred_prob = Y_pred_prob.toarray()
# Calculate the ROC AUC score for each disease label
roc_auc_scores = []
for i in range(y_test_encoded.shape[1]):
    auc_score = roc_auc_score(y_test_encoded[:, i], Y_pred_prob[:, i])
    roc_auc_scores.append(auc_score)

# ROC AUC score for the entire model
roc_auc_avg = roc_auc_score(y_test_encoded, Y_pred_prob, average="weighted")

# Log loss for the model
logloss = log_loss(y_test_encoded, Y_pred_prob)

Y_pred = best_classifier_nb.predict(X_test)
Y_pred = Y_pred.toarray()

confusion = confusion_matrix(y_test_encoded.argmax(axis=1), Y_pred.argmax(axis=1))

# Classification Report
class_report = classification_report(y_test_encoded.argmax(axis=1), Y_pred.argmax(axis=1))

# Plot ROC curves
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(y_test_encoded.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(y_test_encoded[:, i], Y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve for a specific disease (change the index)
plt.figure()
plt.plot(fpr[0], tpr[0], color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc[0])
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# Print metrics
print(f"ROC AUC Score (Average): {roc_auc_avg:.2f}")
print(f"Log Loss: {logloss:.2f}")
print("Confusion Matrix:")
print(confusion)
print("Classification Report:")
print(class_report)

plt.figure(figsize=(10, 6))
index = 1

plt.title('Validation Curve')
for param_name, param_range in param_grid.items():

  train_scores, valid_scores = validation_curve(best_classifier_nb, X_train, y_train_encoded, param_name=param_name, param_range=param_range, cv=10, scoring='accuracy')

  # Calculate mean and standard deviation of training and validation scores
  train_scores_mean = np.mean(train_scores, axis=1)
  train_scores_std = np.std(train_scores, axis=1)
  valid_scores_mean = np.mean(valid_scores, axis=1)
  valid_scores_std = np.std(valid_scores, axis=1)
  plt.subplot(index, 1, 1)
  index += 1
  plt.xlabel(param_name)
  plt.ylabel('Accuracy')
  plt.grid()
  plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='r')
  plt.fill_between(param_range, valid_scores_mean - valid_scores_std, valid_scores_mean + valid_scores_std, alpha=0.1, color='g')
  plt.plot(param_range, train_scores_mean, 'o-', color='r', label='Training Score')
  plt.plot(param_range, valid_scores_mean, 'o-', color='g', label='Validation Score')
  plt.legend(loc='best')
  plt.show()

"""## Saving the model and label encoders"""

import joblib
dir_name = '/content/drive/MyDrive/models/'
joblib.dump(best_classifier_nb, '{}classifierchain_naive_bayes_classifier.joblib'.format(dir_name))

#joblib.dump(label_encoder, '{}onevsrest_decision_tree_label_encoder.joblib'.format(dir_name))

# Save the MultiLabelBinarizer
joblib.dump(mlb_diseases, '{}classifierchain_naive_bayes_multilabel_binarizer.joblib'.format(dir_name))

from joblib import load

classifier_name = '{}classifierchain_naive_bayes_classifier.joblib'.format(dir_name)
mlb_name = '{}classifierchain_naive_bayes_multilabel_binarizer.joblib'.format(dir_name)

mlb = load(mlb_name)
# Load the trained classifier
classifier = load(classifier_name)


def predict_diseases(user_input):
    user_symptoms = create_multilabel_data(user_input)

    print(user_symptoms)
    user_symptoms = user_symptoms.fillna(0)
    # Make multi-label predictions
    predicted_probs = classifier.predict_proba(user_symptoms.values)

    print(predicted_probs)

    #disease_labels = mlb.inverse_transform(predicted_probs)
    results = {}
    # Create a dictionary to store results
    #for i, label in enumerate(disease_labels):
    #    results[label] = predicted_probs[0][i]

    sorted_results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)}

    return sorted_results

prediction = predict_diseases(['itching', 'skin_rash'])
#for disease, probability in prediction.items():
#    print(f"Disease: {disease}, Probability: {probability:.2f}")


#TODO: can make use of logistic regression, add some threshold, and then we can even make use of a subset of symptoms too

"""# ClassifierChain SVM"""

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

mlb_diseases = MultiLabelBinarizer()

all_labels =  pd.concat([df_train['prognosis'], df_test['prognosis']], axis=0)
disease_labels_all = mlb_diseases.fit_transform(all_labels.apply(lambda x: [x]))


y_train_encoded = mlb_diseases.transform(df_train['prognosis'].apply(lambda x: [x]))

y_test_encoded = mlb_diseases.transform(df_test['prognosis'].apply(lambda x: [x]))

base_classifier_svm = SVC(probability=True)

# Define hyperparameters and their possible values for grid search
param_grid = {
    'classifier__C': [0.1, 1, 10],  # Regularization parameter
    'classifier__kernel': ['linear', 'rbf', 'poly'],  # Kernel types to try
}

classifier = ClassifierChain(base_classifier_svm)

grid_search_svm = GridSearchCV(classifier, param_grid, cv=10, verbose=1, n_jobs=-1)
grid_search_svm.fit(X_train, y_train_encoded)

# Get the best estimator from grid search
best_classifier_svm = grid_search_svm.best_estimator_

Y_pred_prob = best_classifier_svm.predict_proba(X_test)
Y_pred_prob = Y_pred_prob.toarray()
# Calculate the ROC AUC score for each disease label
roc_auc_scores = []
for i in range(y_test_encoded.shape[1]):
    auc_score = roc_auc_score(y_test_encoded[:, i], Y_pred_prob[:, i])
    roc_auc_scores.append(auc_score)

# ROC AUC score for the entire model
roc_auc_avg = roc_auc_score(y_test_encoded, Y_pred_prob, average="weighted")

# Log loss for the model
logloss = log_loss(y_test_encoded, Y_pred_prob)

Y_pred = best_classifier_svm.predict(X_test)
Y_pred = Y_pred.toarray()
confusion = confusion_matrix(y_test_encoded.argmax(axis=1), Y_pred.argmax(axis=1))

# Classification Report
class_report = classification_report(y_test_encoded.argmax(axis=1), Y_pred.argmax(axis=1))

# Plot ROC curves
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(y_test_encoded.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(y_test_encoded[:, i], Y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve for a specific disease (change the index)
plt.figure()
plt.plot(fpr[0], tpr[0], color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc[0])
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# Print metrics
print(f"ROC AUC Score (Average): {roc_auc_avg:.2f}")
print(f"Log Loss: {logloss:.2f}")
print("Confusion Matrix:")
print(confusion)
print("Classification Report:")
print(class_report)

plt.figure(figsize=(10, 6))
index = 1

plt.title('Validation Curve')
for param_name, param_range in param_grid.items():

  train_scores, valid_scores = validation_curve(best_classifier_svm, X_train, y_train_encoded, param_name=param_name, param_range=param_range, cv=10, scoring='accuracy')

  # Calculate mean and standard deviation of training and validation scores
  train_scores_mean = np.mean(train_scores, axis=1)
  train_scores_std = np.std(train_scores, axis=1)
  valid_scores_mean = np.mean(valid_scores, axis=1)
  valid_scores_std = np.std(valid_scores, axis=1)
  plt.subplot(index, 1, 1)
  index += 1
  plt.xlabel(param_name)
  plt.ylabel('Accuracy')
  plt.grid()
  plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='r')
  plt.fill_between(param_range, valid_scores_mean - valid_scores_std, valid_scores_mean + valid_scores_std, alpha=0.1, color='g')
  plt.plot(param_range, train_scores_mean, 'o-', color='r', label='Training Score')
  plt.plot(param_range, valid_scores_mean, 'o-', color='g', label='Validation Score')
  plt.legend(loc='best')
  plt.show()

"""## Saving the model and label encoders"""

import joblib
dir_name = '/content/drive/MyDrive/models/'
joblib.dump(best_classifier_svm, '{}classifierchain_svm_classifier.joblib'.format(dir_name))

#joblib.dump(label_encoder, '{}onevsrest_decision_tree_label_encoder.joblib'.format(dir_name))

# Save the MultiLabelBinarizer
joblib.dump(mlb_diseases, '{}classifierchain_svm_multilabel_binarizer.joblib'.format(dir_name))

from joblib import load

classifier_name = '{}classifierchain_svm_classifier.joblib'.format(dir_name)
mlb_name = '{}classifierchain_svm_multilabel_binarizer.joblib'.format(dir_name)

mlb = load(mlb_name)
# Load the trained classifier
classifier = load(classifier_name)


def predict_diseases(user_input):
    user_symptoms = create_multilabel_data(user_input)

    print(user_symptoms)
    user_symptoms = user_symptoms.fillna(0)
    # Make multi-label predictions
    predicted_probs = classifier.predict_proba(user_symptoms.values)

    print(predicted_probs)

    #disease_labels = mlb.inverse_transform(predicted_probs)
    results = {}
    # Create a dictionary to store results
    #for i, label in enumerate(disease_labels):
    #    results[label] = predicted_probs[0][i]

    sorted_results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)}

    return sorted_results

prediction = predict_diseases(['itching', 'skin_rash'])
#for disease, probability in prediction.items():
#    print(f"Disease: {disease}, Probability: {probability:.2f}")


#TODO: can make use of logistic regression, add some threshold, and then we can even make use of a subset of symptoms too