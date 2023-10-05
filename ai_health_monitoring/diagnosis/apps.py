from django.apps import AppConfig
import pandas as pd
from joblib import load
from .utils import clean_col_name
import spacy


class DiagnosisConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "diagnosis"
    data_dir_name = "/home/yatharth/AI-Driven-Personal-Health-Assistant-with-Decision-Support/data"
    model_dir_name = "/home/yatharth/AI-Driven-Personal-Health-Assistant-with-Decision-Support/models"
    def ready(self):
        self.dataframe = pd.read_csv("{}/Training.csv".format(self.data_dir_name))
        self.dataframe = self.dataframe.drop(columns="Unnamed: 133")
        self.dataframe.columns = list(map(clean_col_name, list(self.dataframe.columns)))
        self.symptoms = list(self.dataframe.columns)[:-1]
        self.mlb = load("{}/onevsrest_svm_multilabel_binarizer.joblib".format(self.model_dir_name))
        self.classifier = load("{}/onevsrest_svm_classifier.joblib".format(self.model_dir_name))

        self.nlp = spacy.load("en_core_web_md")



