import pandas as pd
import os
from classifier import logger
from classifier.entity.config_entity import ModelEvaluationConfig
from classifier.Mylib import myfuncs
from sklearn import metrics


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def evaluate_model(self):
        # Load data
        df = myfuncs.load_python_object(self.config.test_data_path)
        preprocessor = myfuncs.load_python_object(self.config.preprocessor_path)
        self.model = myfuncs.load_python_object(self.config.model_path)
        self.class_names = myfuncs.load_python_object(self.config.class_names_path)

        # Transform test data
        df_transformed = preprocessor.transform(df)
        self.df_feature = df_transformed.drop(columns=[self.config.target_col])
        self.df_target = df_transformed[self.config.target_col]

        # Các chỉ số đánh giá của model
        self.model_results_text = "========KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH================\n"

        ## Chỉ số scoring
        test_score = myfuncs.evaluate_model_on_one_scoring_17(
            self.model, self.df_feature, self.df_target, self.config.scoring
        )
        self.model_results_text += f"====CHỈ SỐ SCORING====\n"
        self.model_results_text += f"{self.config.scoring}: {test_score}"

        # Các chỉ số khác
        self.model_results_text += "====CÁC CHỈ SỐ KHÁC===========\n"

        evaluate_func = (
            myfuncs.evaluate_classifier_on_test_data_18
            if self.config.predictor_type == "c"
            else myfuncs.evaluate_regressor_on_test_data_18
        )
        self.model_results_text += evaluate_func(
            self.model, self.df_feature, self.df_target, self.class_names
        )

        # In ra kết quả đánh giá
        print(self.model_results_text)

        # Lưu chỉ số đánh giá vào file results.txt
        with open(self.config.results_path, mode="w") as file:
            file.write(self.model_results_text)
