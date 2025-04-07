import pandas as pd
import os
from classifier import logger
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit, GridSearchCV
from classifier.entity.config_entity import ModelTrainerConfig
from classifier.Mylib import myfuncs
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from xgboost import XGBClassifier
from scipy.stats import randint
import random
from lightgbm import LGBMClassifier
from sklearn.model_selection import ParameterSampler
from sklearn import metrics
from sklearn.base import clone
import time


class ManyModelsTypeModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def load_data_to_train(self):
        # Load các training data
        self.train_feature_data = myfuncs.load_python_object(
            self.config.train_feature_path
        )
        self.train_target_data = myfuncs.load_python_object(
            self.config.train_target_path
        )
        self.val_feature_data = myfuncs.load_python_object(self.config.val_feature_path)
        self.val_target_data = myfuncs.load_python_object(self.config.val_target_path)

        # Load models
        self.models = [
            myfuncs.convert_string_to_object_4(model) for model in self.config.models
        ]

    def train_model(self):
        for model in self.models:
            model.fit(self.train_feature_data, self.train_target_data)

    def evaluate_model(
        self, model, train_feature, train_target, val_feature, val_target, scoring
    ):
        if scoring == "accuracy":
            train_prediction = model.predict(train_feature)
            val_prediction = model.predict(val_feature)
            return metrics.accuracy_score(
                train_target, train_prediction
            ), metrics.accuracy_score(val_target, val_prediction)
        elif scoring == "log_loss":
            train_prediction = model.predict_proba(train_feature)
            val_prediction = model.predict_proba(val_feature)
            return metrics.log_loss(train_target, train_prediction), metrics.log_loss(
                val_target, val_prediction
            )
        else:
            raise ValueError(
                "===== Chỉ mới định nghĩa cho accuracy và log_loss =============="
            )

    def find_best_model_and_scoring(self):
        train_val_scorings = [
            self.evaluate_model(
                model,
                self.train_feature_data,
                self.train_target_data,
                self.val_feature_data,
                self.val_target_data,
                self.config.scoring,
            )
            for model in self.models
        ]
        train_scorings = [item[0] for item in train_val_scorings]
        val_scorings = [item[1] for item in train_val_scorings]

        index_best_model = None
        if self.config.scoring == "accuracy":
            index_best_model = np.argmax(val_scorings)
        elif self.config.scoring == "log_loss":
            index_best_model = np.argmin(val_scorings)
        else:
            raise ValueError(
                "===== Chỉ mới định nghĩa cho accuracy và log_loss =============="
            )

        self.best_model = self.models[index_best_model]
        self.train_scoring = train_scorings[index_best_model]
        self.val_scoring = val_scorings[index_best_model]

    def save_best_model_results(self):
        # Tìm model tốt nhất và chỉ số scoring
        self.find_best_model_and_scoring()

        # Các chỉ số đánh giá của model
        self.best_model_results_text = (
            "========KẾT QUẢ MODEL TỐT NHẤT================\n"
        )

        ## Chỉ số scoring
        self.best_model_results_text += f"====CHỈ SỐ SCORING====\n"
        self.best_model_results_text += (
            f"Train {self.config.scoring}: {self.train_scoring}"
        )
        self.best_model_results_text += f"Val {self.config.scoring}: {self.val_scoring}"

        # Các chỉ số khác bao gồm accuracy + classfication report
        train_prediction = self.best_model.predict(self.train_feature_data)
        val_prediction = self.best_model.predict(self.val_feature_data)
        train_accuracy = metrics.accuracy_score(
            self.train_target_data, train_prediction
        )
        val_accuracy = metrics.accuracy_score(self.val_target_data, val_prediction)

        train_classification_report = metrics.classification_report(
            self.train_target_data, train_prediction
        )
        val_classification_report = metrics.classification_report(
            self.val_target_data, val_prediction
        )

        self.best_model_results_text += "\n====CHỈ SỐ ĐÁNH GIÁ KHÁC====\n"
        self.best_model_results_text += f"Train accuracy: {train_accuracy}"
        self.best_model_results_text += f"Val accuracy: {val_accuracy}\n"
        self.best_model_results_text += (
            f"Train classification_report: \n{train_classification_report}\n"
        )
        self.best_model_results_text += (
            f"Val classification_report: \n{val_classification_report}"
        )

        # In ra kết quả đánh giá
        print(self.best_model_results_text)

        # Lưu chỉ số đánh giá vào file results.txt
        with open(self.config.results_path, mode="w") as file:
            file.write(self.best_model_results_text)

        # Lưu lại model tốt nhất
        myfuncs.save_python_object(self.config.best_model_path, self.best_model)

    def save_list_monitor_components(self):
        # Chuyển đổi train, val scoring để hiển thị lên biểu đồ
        if self.config.scoring == "accuracy":
            self.train_scoring = self.train_scoring * 100
            self.val_scoring = self.val_scoring * 100

        if os.path.exists(self.config.list_monitor_components_path):
            self.list_monitor_components = myfuncs.load_python_object(
                self.config.list_monitor_components_path
            )

        else:
            self.list_monitor_components = []

        self.list_monitor_components += [
            (
                self.config.model_name,
                self.train_scoring,
                self.val_scoring,
                self.best_model_results_text,
            )
        ]

        myfuncs.save_python_object(
            self.config.list_monitor_components_path, self.list_monitor_components
        )
