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

        # Transform test data
        df_transformed = preprocessor.transform(df)
        self.df_feature = df_transformed.drop(columns=[self.config.target_col])
        self.df_target = df_transformed[self.config.target_col]

        # Các chỉ số đánh giá của model
        self.best_model_results_text = (
            "========KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH================\n"
        )

        ## Chỉ số scoring
        self.find_model_scoring()
        self.best_model_results_text += f"====CHỈ SỐ SCORING====\n"
        self.best_model_results_text += f"{self.config.scoring}: {self.test_score}"

        # Các chỉ số khác bao gồm accuracy + classfication report
        feature_prediction = self.model.predict(self.df_feature)
        accuracy = metrics.accuracy_score(self.df_target, feature_prediction)

        classification_report = metrics.classification_report(
            self.df_target, feature_prediction
        )

        self.best_model_results_text += "\n====CHỈ SỐ ĐÁNH GIÁ KHÁC====\n"
        self.best_model_results_text += f"Accuracy: {accuracy}"
        self.best_model_results_text += (
            f"Classification_report: \n{classification_report}\n"
        )

        # In ra kết quả đánh giá
        print(self.best_model_results_text)

        # Lưu chỉ số đánh giá vào file results.txt
        with open(self.config.results_path, mode="w") as file:
            file.write(self.best_model_results_text)

    def find_model_scoring(self):
        if self.config.scoring == "accuracy":
            df_feature_prediction = self.model.predict(self.df_feature)
            self.test_score = metrics.accuracy_score(
                self.df_target, df_feature_prediction
            )
        elif self.config.scoring == "log_loss":
            df_feature_prediction = self.model.predict_proba(self.df_feature)
            self.test_score = metrics.log_loss(self.df_target, df_feature_prediction)
        else:
            raise ValueError(
                "===== Chỉ mới định nghĩa cho accuracy và log_loss =============="
            )
