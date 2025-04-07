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


class ModelTrainer:
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

        # Gộp train, val vào đúng 1 df
        self.features, self.target, self.trainval_splitter = (
            myfuncs.get_features_target_spliter_for_CV_train_val(
                self.train_feature_data,
                self.train_target_data,
                self.val_feature_data,
                self.val_target_data,
            )
        )

        # Load base model (chưa có tham số)
        self.base_model = myfuncs.convert_string_to_object_4(self.config.base_model)

        # Load params thực hiện fine tune model
        self.param_grid = myfuncs.get_param_grid_model(self.config.param_grid)

        # Load scoring để sử dụng vào RandomizedSearchCV, GridSearchCV (vd: log_loss -> neg_log_loss)
        self.scoring = self.config.scoring
        if self.config.scoring == "log_loss":
            self.scoring = "neg_log_loss"
        elif self.config.scoring == "mse":
            self.scoring = "neg_mean_squared_error"
        elif self.config.scoring == "mae":
            self.scoring = "neg_mean_absolute_error"

        # Load searcher
        if self.config.model_training_type == "r":
            self.searcher = RandomizedSearchCV(
                self.base_model,
                param_distributions=self.param_grid,
                n_iter=self.config.n_iter,
                cv=self.trainval_splitter,
                random_state=42,
                scoring=self.scoring,
                return_train_score=True,
                verbose=2,
            )
        elif self.config.model_training_type == "g":
            self.searcher = GridSearchCV(
                self.base_model,
                param_grid=self.param_grid,
                cv=self.trainval_splitter,
                scoring=self.scoring,
                return_train_score=True,
                verbose=2,
            )
        elif self.config.model_training_type == "rcv":
            self.searcher = RandomizedSearchCV(
                self.base_model,
                param_distributions=self.param_grid,
                n_iter=self.config.n_iter,
                cv=5,
                random_state=42,
                scoring=self.scoring,
                return_train_score=True,
                verbose=2,
            )
        elif self.config.model_training_type == "gcv":
            self.searcher = GridSearchCV(
                self.base_model,
                param_grid=self.param_grid,
                cv=5,
                scoring=self.scoring,
                return_train_score=True,
                verbose=2,
            )
        else:
            raise ValueError(
                "===== Giá trị model_training_type không hợp lệ =============="
            )

        # Load classes
        self.class_names = myfuncs.load_python_object(self.config.class_names_path)

    def train_model(self):
        self.searcher.fit(self.features, self.target)

    def find_model_scoring(self):
        cv_results = zip(
            self.cv_results["mean_test_score"], self.cv_results["mean_train_score"]
        )
        cv_results = sorted(cv_results, key=lambda x: x[0], reverse=True)
        self.val_scoring, self.train_scoring = cv_results[0]

        if self.config.scoring == "log_loss":
            self.val_scoring, self.train_scoring = (
                -self.val_scoring,
                -self.train_scoring,
            )

    def save_best_model_results(self):
        self.best_model = self.searcher.best_estimator_
        self.cv_results = self.searcher.cv_results_

        # Các chỉ số đánh giá của model
        self.best_model_results_text = (
            "========KET QUA CUA MO HINH TOT NHAT================\n"
        )

        ## Chỉ số scoring
        self.find_model_scoring()
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
