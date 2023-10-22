from django.apps import AppConfig
import pandas as pd
from joblib import load
from .utils import clean_col_name, clean_col_name_food
import spacy


class DiagnosisConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "diagnosis"
    data_dir_name = "../data"
    model_dir_name = "../models"

    def ready(self):
        """
        This function helps to load all the necessary datasets/models needed by the entire application. Loading them in this function
        makes it available to the entire diagnosis app and does not load them on every user request.
        """
        ## =================diagnosis================= ##
        self.dataframe = pd.read_csv("{}/Training.csv".format(self.data_dir_name))
        self.dataframe = self.dataframe.drop(columns="Unnamed: 133")
        self.dataframe.columns = list(map(clean_col_name, list(self.dataframe.columns)))
        self.symptoms = list(self.dataframe.columns)[:-1]

        self.disease_filter = pd.read_csv("{}/dataset.csv".format(self.data_dir_name))
        self.disease_filter = self.disease_filter[['Disease','Nutrition_Rec_1']]

        self.precautions_df = pd.read_csv("{}/symptom_precaution.csv".format(self.data_dir_name))
        self.precautions_df.columns = list(map(clean_col_name, list(self.precautions_df.columns)))
        self.precautions_df = self.precautions_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

        self.mlb = load("{}/classifierchain_svm_multilabel_binarizer.joblib".format(self.model_dir_name))
        self.classifier = load("{}/classifierchain_svm_classifier.joblib".format(self.model_dir_name))

        self.nlp = spacy.load("en_core_web_md")


        ## ===========meal planner============= ##
        self.food_df_for_checker = pd.read_csv('{}/Food.csv'.format(self.data_dir_name), encoding='latin1')
        self.food_df_for_checker.drop(['Calcium', 'Iron', 'Sodium', 'Vitamin A', 'Vitamin B1 (Thiamine)', 
                                'Vitamin B2 (Rivoflavin)', 'Vitamin C', 'Linoleic Acid', 
                                'Alpha_Linolenic_Acid', 'Fat/weight', 'Protein/weight', 'Fat_type', 
                                'Protein_type', 'Meal_Type', 'Class', 'Diet_Restrictions'], axis=1, inplace=True)
        self.food_df_for_checker.columns = list(map(clean_col_name_food, list(self.food_df_for_checker.columns)))

        self.food_df_for_ga = pd.read_csv('{}/Food.csv'.format(self.data_dir_name), encoding='latin1')
        self.food_df_for_ga.drop(['Class', 'Total Dietary (g)', 'Calcium', 'Iron', 'Sodium', 'Vitamin A', 'Vitamin B1 (Thiamine)', 'Vitamin B2 (Rivoflavin)', 
                                'Vitamin C', 'Linoleic Acid', 'Alpha_Linolenic_Acid'], axis=1, inplace=True)
        self.food_df_for_ga.columns = list(map(clean_col_name_food, list(self.food_df_for_ga.columns)))
