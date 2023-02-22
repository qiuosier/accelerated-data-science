from collections import defaultdict
import json
import unittest
import copy

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
import sklearn
import xgboost as xgb
from ads.common.model import ADSModel
from ads.model.extractor.model_info_extractor_factory import ModelInfoExtractorFactory
from ads.dataset.factory import DatasetFactory
from unittest.mock import patch, MagicMock, PropertyMock
from numpy import nan
from sklearn import datasets, svm
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tests.ads_unit_tests.utils import get_test_dataset_path
from ads.model.extractor.tensorflow_extractor import TensorflowExtractor
from ads.model.extractor.pytorch_extractor import PytorchExtractor


class TestModelInfoExtractor(unittest.TestCase):
    X, y = make_regression(
        n_samples=10000, n_features=10, n_informative=2, random_state=42
    )
    rf_clf = RandomForestRegressor(n_estimators=10).fit(X, y)
    rf_clf.fit(X, y)

    def test_generic_sklearn_model_estimator_pipeline(self):
        X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
        Y = np.array([1, 1, 2, 2])
        # Always scale the input. The most convenient way is to use a pipeline.
        clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3))
        clf.fit(X, Y)
        original_dict = clf.get_params()
        original_dict_copy = copy.copy(clf.get_params())
        metadata_taxonomy = ModelInfoExtractorFactory.extract_info(clf)

        assert isinstance(metadata_taxonomy, dict)
        assert metadata_taxonomy["Framework"] == "scikit-learn"
        assert metadata_taxonomy["FrameworkVersion"] == sklearn.__version__
        assert metadata_taxonomy["Algorithm"] == "Pipeline"
        assert json.dumps(metadata_taxonomy["Hyperparameters"]) != ""

    def test_generic_sklearn_model_estimator_classifier(self):
        digits = datasets.load_digits()
        clf = svm.SVC(gamma=0.001, C=100.0)
        clf.fit(digits.data[:-1], digits.target[:-1])
        metadata_taxonomy = ModelInfoExtractorFactory.extract_info(clf)
        assert isinstance(metadata_taxonomy, dict)
        assert metadata_taxonomy["Framework"] == "scikit-learn"
        assert metadata_taxonomy["FrameworkVersion"] == sklearn.__version__
        assert metadata_taxonomy["Algorithm"] == "SVC"
        assert json.dumps(metadata_taxonomy["Hyperparameters"]) != ""

    def test_generic_sklearn_model_estimator_model_selection(self):
        X_digits, y_digits = datasets.load_digits(return_X_y=True)
        Cs = np.logspace(-6, -1, 10)
        svc = svm.SVC(kernel="linear")
        clf = GridSearchCV(estimator=svc, param_grid=dict(C=Cs), n_jobs=-1)
        clf.fit(X_digits[:1000], y_digits[:1000])
        original_dict = clf.get_params()
        original_dict_copy = copy.copy(clf.get_params())
        metadata_taxonomy = ModelInfoExtractorFactory.extract_info(clf)
        assert isinstance(metadata_taxonomy, dict)
        assert metadata_taxonomy["Framework"] == "scikit-learn"
        assert metadata_taxonomy["FrameworkVersion"] == sklearn.__version__
        assert metadata_taxonomy["Algorithm"] == "GridSearchCV"
        assert json.dumps(metadata_taxonomy["Hyperparameters"]) != ""

    def test_ADS_sklearn_model(self):
        oracle_fraud_dataset = pd.read_csv(
            get_test_dataset_path("vor_oracle_fraud_dataset_5k.csv")
        )
        ds = DatasetFactory.open(oracle_fraud_dataset, target="anomalous")
        transformed_ds = ds.auto_transform(fix_imbalance=False)
        train, test = transformed_ds.train_test_split(test_size=0.15)
        rf_clf = RandomForestClassifier(n_estimators=10).fit(
            train.X.values, train.y.values
        )
        rf_model = ADSModel.from_estimator(rf_clf)
        metadata_taxonomy = ModelInfoExtractorFactory.extract_info(rf_model)
        assert isinstance(metadata_taxonomy, dict)
        assert metadata_taxonomy["Framework"] == "scikit-learn"
        assert metadata_taxonomy["FrameworkVersion"] == sklearn.__version__

        metadata_hyperparameters = metadata_taxonomy["Hyperparameters"]
        assert json.dumps(metadata_taxonomy["Hyperparameters"]) != ""

    def test_generic_xgboost_model(self):
        X, y = make_regression(
            n_samples=10000, n_features=10, n_informative=2, random_state=42
        )
        xg_reg = xgb.XGBRegressor(
            objective="reg:linear",
            colsample_bytree=0.3,
            learning_rate=0.1,
            max_depth=5,
            alpha=10,
            n_estimators=10,
        )
        metadata_taxonomy = ModelInfoExtractorFactory.extract_info(xg_reg)
        assert isinstance(metadata_taxonomy, dict)
        assert metadata_taxonomy["Framework"] == "xgboost"
        assert metadata_taxonomy["FrameworkVersion"] == xgb.__version__
        assert metadata_taxonomy["Algorithm"] == "XGBRegressor"
        assert json.dumps(metadata_taxonomy["Hyperparameters"]) != ""

    def test_generic_lightgbm_model(self):
        titanic_dataset = pd.read_csv(get_test_dataset_path("vor_titanic.csv"))
        # Declare feature vector and target variable
        X = titanic_dataset[["Pclass", "Age", "Fare"]]
        y = titanic_dataset["Survived"]
        # split the dataset into the training set and test set
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=0
        )
        clf = lgb.LGBMClassifier()
        clf.fit(X_train, y_train)

        metadata_taxonomy = ModelInfoExtractorFactory.extract_info(clf)
        assert isinstance(metadata_taxonomy, dict)
        assert metadata_taxonomy["Framework"] == "lightgbm"
        assert metadata_taxonomy["FrameworkVersion"] == lgb.__version__
        assert metadata_taxonomy["Algorithm"] == "LGBMClassifier"
        assert json.dumps(metadata_taxonomy["Hyperparameters"]) != ""

    @pytest.mark.skip(
        reason="wait for proper testing pipeline for tensorflow related tests"
    )
    def test_generic_keras_model(self):

        import tensorflow

        mnist = tensorflow.keras.datasets.mnist

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        model = tensorflow.keras.models.Sequential(
            [
                tensorflow.keras.layers.Flatten(input_shape=(28, 28)),
                tensorflow.keras.layers.Dense(128, activation="relu"),
                tensorflow.keras.layers.Dropout(0.2),
                tensorflow.keras.layers.Dense(10),
            ]
        )

        loss_fn = tensorflow.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        )

        model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

        model.fit(x_train, y_train, epochs=10, batch_size=1024)

        metadata_taxonomy = ModelInfoExtractorFactory.extract_info(model)
        assert isinstance(metadata_taxonomy, dict)
        assert metadata_taxonomy["Framework"] == "lightGBM"
        assert metadata_taxonomy["FrameworkVersion"] == lgb.__version__
        assert metadata_taxonomy["Framework"] == "keras"
        assert metadata_taxonomy["FrameworkVersion"] == keras.__version__
        assert (
            metadata_taxonomy["Algorithm"]
            == tensorflow.python.keras.engine.sequential.Sequential
        )
        assert json.dumps(metadata_taxonomy["Hyperparameters"]) != ""

    @patch("ads.common.utils.get_base_modules")
    @patch(
        "ads.model.extractor.tensorflow_extractor.TensorflowExtractor.version",
        new_callable=PropertyMock,
    )
    def test_tensorflow_extractors(self, mock_version, mock_get_base_modules):
        class any:
            __module__ = "tensorflow.python.module.module"

        mock_get_base_modules.return_value = [any()]
        mock_version.return_value = "1.0.0"

        metadata_taxonomy = ModelInfoExtractorFactory.extract_info(self.rf_clf)
        assert isinstance(metadata_taxonomy, dict)
        assert metadata_taxonomy["Framework"] == "tensorflow"
        assert metadata_taxonomy["FrameworkVersion"] == "1.0.0"
        assert metadata_taxonomy["Algorithm"] is not None

    @patch("ads.common.utils.get_base_modules")
    @patch(
        "ads.model.extractor.pytorch_extractor.PytorchExtractor.version",
        new_callable=PropertyMock,
    )
    def test_torch_extractors(self, mock_version, mock_get_base_modules):
        class any:
            __module__ = "torch.nn.modules.module.Module"

        mock_get_base_modules.return_value = [any()]
        mock_version.return_value = "1.0.0"

        metadata_taxonomy = ModelInfoExtractorFactory.extract_info(self.rf_clf)
        assert isinstance(metadata_taxonomy, dict)
        assert metadata_taxonomy["Framework"] == "pytorch"
        assert metadata_taxonomy["FrameworkVersion"] == "1.0.0"
        assert metadata_taxonomy["Algorithm"] is not None
