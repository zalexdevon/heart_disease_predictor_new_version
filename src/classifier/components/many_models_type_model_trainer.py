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

        self.num_models = len(self.models)

        # Load classes
        self.class_names = myfuncs.load_python_object(self.config.class_names_path)

    def train_model(self):
        print(
            f"\n========TIEN HANH TRAIN {self.num_models} MODELS !!!!!!================\n"
        )
        self.train_scorings = []
        self.val_scorings = []
        for index, model in enumerate(self.models):
            model.fit(self.train_feature_data, self.train_target_data)
            train_scoring = myfuncs.evaluate_model_on_one_scoring_17(
                model,
                self.train_feature_data,
                self.train_target_data,
                self.config.scoring,
            )
            val_scoring = myfuncs.evaluate_model_on_one_scoring_17(
                model,
                self.val_feature_data,
                self.val_target_data,
                self.config.scoring,
            )

            # In kết quả
            print(
                f"Model no. {index} -> Train {self.config.scoring}: {train_scoring}, Val {self.config.scoring}: {val_scoring}\n"
            )

            # TODO: d
            print(f"C = {model.C}")
            # d

            self.train_scorings.append(train_scoring)
            self.val_scorings.append(val_scoring)

        print(
            f"\n========KET THUC TRAIN {self.num_models} MODELS !!!!!!================\n"
        )

    def find_train_val_scorings_to_find_the_best(self):
        sign_for_score = 1  # Nếu scoring cần min thì lấy âm -> quy về tìm lớn nhất thôi
        if self.config.scoring in ["log_loss", "mse", "mae"]:
            self.config.target_score = -self.config.target_score
            sign_for_score = -1

        self.train_scorings_to_find_the_best = np.asarray(
            [item * sign_for_score for item in self.train_scorings]
        )
        self.val_scorings_to_find_the_best = np.asarray(
            [item * sign_for_score for item in self.val_scorings]
        )

    def find_best_model_and_scoring(self):
        """TÌm model tốt nhất và scoring tương ứng

        Examples:
            Với **monitor = val_accuracy và indicator = 0.99**

            Tìm model thỏa val_accuracy > 0.99 và train_accuracy > 0.99 (1) và val_accuracy là lớn nhất trong số đó

            Nếu không thỏa (1) thì lấy theo val_accuracy lớn nhất
        """

        # Tìm index của best model
        indexs_good_model = np.where(
            (self.val_scorings_to_find_the_best > self.config.target_score)
            & (self.train_scorings_to_find_the_best > self.config.target_score)
        )[0]

        index_best_model = None
        if (
            len(indexs_good_model) == 0
        ):  # Nếu ko có model nào đạt chỉ tiêu thì lấy cái tốt nhất
            index_best_model = np.argmax(self.val_scorings_to_find_the_best)
        else:
            val_series = pd.Series(
                self.val_scorings_to_find_the_best[indexs_good_model],
                index=indexs_good_model,
            )
            index_best_model = val_series.idxmax()

        self.best_model = self.models[index_best_model]
        self.train_scoring = self.train_scorings[index_best_model]
        self.val_scoring = self.val_scorings[index_best_model]

    def save_best_model_results(self):
        # Tìm model tốt nhất và chỉ số scoring
        self.find_train_val_scorings_to_find_the_best()
        self.find_best_model_and_scoring()

        # Các chỉ số đánh giá của model
        self.best_model_results_text = (
            "========KẾT QUẢ MODEL TỐT NHẤT================\n"
        )

        ## Chỉ số scoring
        self.best_model_results_text += f"====CHỈ SỐ SCORING====\n"
        self.best_model_results_text += (
            f"Train {self.config.scoring}: {self.train_scoring}\n"
        )
        self.best_model_results_text += (
            f"Val {self.config.scoring}: {self.val_scoring}\n"
        )

        # Các chỉ số khác bao gồm accuracy + classfication report
        self.best_model_results_text += "====CÁC CHỈ SỐ KHÁC===========\n"

        evaluate_func = (
            myfuncs.evaluate_classifier_15
            if self.config.predictor_type == "c"
            else myfuncs.evaluate_regressor_16
        )
        self.best_model_results_text += evaluate_func(
            self.best_model,
            self.train_feature_data,
            self.train_target_data,
            self.val_feature_data,
            self.val_target_data,
            self.class_names,
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
