#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Unit tests for model frameworks. Includes tests for:
 - TensorFlowModel
"""
import base64
import os
import shutil
from io import BytesIO

import numpy as np
import onnxruntime as rt
import pandas as pd
import pytest
import tensorflow as tf
from ads.model.framework.tensorflow_model import TensorFlowModel

tmp_model_dir = "/tmp/model/"
mnist = tf.keras.datasets.mnist
mnist.load_data()


def setup_module():
    os.makedirs(tmp_model_dir, exist_ok=True)


class MyTFModel:

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    x_train, y_train = x_train[:1000], y_train[:1000]

    def training(self):
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(10),
            ]
        )
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
        model.fit(self.x_train, self.y_train, epochs=1)

        return model


class TestTensorFlowModel:
    """Unittests for the TensorFlowModel class."""

    def setup_class(cls):
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        cls.x_train, cls.x_test = x_train / 255.0, x_test / 255.0

        cls.myTFModel = MyTFModel().training()
        cls.dummy_input = (tf.TensorSpec((None, 28, 28), tf.float64, name="input"),)

        cls.inference_conda_env = "oci://service-conda-packs@ociodscdev/service_pack/cpu/General_Machine_Learning_for_CPUs/1.0/mlcpuv1"
        cls.inference_python_version = "3.6"
        cls.training_conda_env = "oci://service-conda-packs@ociodscdev/service_pack/cpu/Oracle_Database_for_CPU_Python_3.7/1.0/database_p37_cpu_v1"
        cls.training_python_version = "3.7"

    def test_serialize_with_incorrect_model_file_name_onnx(self):
        """
        Test wrong model_file_name format.
        """
        test_tf_model = TensorFlowModel(
            self.myTFModel,
            tmp_model_dir,
        )
        with pytest.raises(ValueError):
            test_tf_model._handle_model_file_name(
                as_onnx=True, model_file_name="model.xxx"
            )

    def test_serialize_with_incorrect_model_file_name_pt(self):
        """
        Test wrong model_file_name format.
        """
        test_tf_model = TensorFlowModel(
            self.myTFModel,
            tmp_model_dir,
        )
        with pytest.raises(ValueError):
            test_tf_model._handle_model_file_name(
                as_onnx=False, model_file_name="model.xxx"
            )

    def test_serialize_using_tf_without_modelname(self):
        """
        Test serialize_model using tf without model_file_name
        """
        test_tf_model = TensorFlowModel(
            self.myTFModel,
            tmp_model_dir,
        )
        test_tf_model.model_file_name = test_tf_model._handle_model_file_name(
            as_onnx=False, model_file_name=None
        )
        test_tf_model.serialize_model(as_onnx=False)
        assert os.path.isfile(tmp_model_dir + "model.h5")

    def test_serialize_using_tf_with_modelname(self):
        """
        Test serialize_model using tf with correct model_file_name
        """
        test_tf_model = TensorFlowModel(
            self.myTFModel,
            tmp_model_dir,
        )
        test_tf_model.model_file_name = "test1.h5"
        test_tf_model.serialize_model(as_onnx=False)
        assert os.path.isfile(tmp_model_dir + "test1.h5")

    def test_serialize_using_onnx_without_modelname(self):
        """
        Test serialize_model using onnx without model_file_name
        """
        test_tf_model = TensorFlowModel(
            self.myTFModel,
            tmp_model_dir,
        )
        test_tf_model.model_file_name = test_tf_model._handle_model_file_name(
            as_onnx=True, model_file_name=None
        )
        test_tf_model.serialize_model(as_onnx=True, dummy_input=self.dummy_input)
        assert os.path.exists(tmp_model_dir + "model.onnx")

    def test_serialize_using_onnx_with_modelname(self):
        """
        Test serialize_model using onnx with correct model_file_name
        """
        test_tf_model = TensorFlowModel(
            self.myTFModel,
            tmp_model_dir,
        )
        test_tf_model.model_file_name = "test2.onnx"
        test_tf_model.serialize_model(as_onnx=True, dummy_input=self.dummy_input)
        assert os.path.exists(tmp_model_dir + "test2.onnx")

    def test_to_onnx(self):
        """
        Test if TensorFlowModel.to_onnx generate onnx model result.
        """
        test_tf_model = TensorFlowModel(
            self.myTFModel,
            tmp_model_dir,
        )
        test_tf_model.to_onnx(
            tmp_model_dir + "test2.onnx", input_signature=self.dummy_input
        )
        assert os.path.exists(tmp_model_dir + "test2.onnx")

    def test_to_onnx_reload(self):
        """
        Test if TensorFlowModel.to_onnx generated model can be reloaded.
        """
        test_tf_model = TensorFlowModel(
            self.myTFModel,
            tmp_model_dir,
        )
        test_tf_model.to_onnx(
            tmp_model_dir + "model1.onnx", input_signature=self.dummy_input
        )
        assert (
            rt.InferenceSession(os.path.join(tmp_model_dir, "model1.onnx")) is not None
        )

    def test_to_onnx_without_dummy_input(self):
        """
        Test if TensorFlowModel.to_onnx raise expected error
        """
        test_tf_model = TensorFlowModel(
            self.myTFModel,
            tmp_model_dir,
        )
        test_tf_model.to_onnx(tmp_model_dir + "model1.onnx")
        assert os.path.exists(tmp_model_dir + "model1.onnx")

    def test_to_onnx_without_path(self):
        """
        Test if TensorFlowModel.to_onnx raise expected error
        """
        test_tf_model = TensorFlowModel(
            self.myTFModel,
            tmp_model_dir,
        )
        with pytest.raises(ValueError):
            test_tf_model.to_onnx(input_signature=self.dummy_input)

    @pytest.mark.parametrize(
        "test_data",
        [pd.Series([1, 2, 3]), [1, 2, 3]],
    )
    def test_get_data_serializer_with_convert_to_list(self, test_data):
        """
        Test if TensorFlowModel.to_onnx raise expected error
        """
        test_tf_model = TensorFlowModel(self.myTFModel, tmp_model_dir)
        serialized_data = test_tf_model.get_data_serializer(test_data).to_dict()
        assert serialized_data["data"] == [1, 2, 3]
        assert serialized_data["data_type"] == str(type(test_data))

    def test_get_data_serializer_helper_numpy(self):
        test_data = np.array([1, 2, 3])
        test_tf_model = TensorFlowModel(self.myTFModel, tmp_model_dir)
        serialized_data = test_tf_model.get_data_serializer(test_data).to_dict()
        load_bytes = BytesIO(base64.b64decode(serialized_data["data"].encode("utf-8")))
        deserialized_data = np.load(load_bytes, allow_pickle=True)
        assert (deserialized_data == test_data).any()

    @pytest.mark.parametrize(
        "test_data",
        [
            pd.DataFrame({"a": [1, 2], "b": [2, 3], "c": [3, 4]}),
        ],
    )
    def test_get_data_serializer_with_pandasdf(self, test_data):
        test_tf_model = TensorFlowModel(self.myTFModel, tmp_model_dir)
        serialized_data = test_tf_model.get_data_serializer(test_data).to_dict()
        assert (
            serialized_data["data"]
            == '{"a":{"0":1,"1":2},"b":{"0":2,"1":3},"c":{"0":3,"1":4}}'
        )
        assert serialized_data["data_type"] == "<class 'pandas.core.frame.DataFrame'>"

    @pytest.mark.parametrize(
        "test_data",
        ["I have an apple", {"a": [1], "b": [2], "c": [3]}],
    )
    def test_get_data_serializer_with_no_change(self, test_data):
        test_tf_model = TensorFlowModel(self.myTFModel, tmp_model_dir)
        serialized_data = test_tf_model.get_data_serializer(test_data).to_dict()
        assert serialized_data["data"] == test_data

    def test_get_data_serializer_raise_error(self):
        class TestData:
            pass

        test_data = TestData()
        test_tf_model = TensorFlowModel(self.myTFModel, tmp_model_dir)
        with pytest.raises(TypeError):
            serialized_data = test_tf_model.get_data_serializer(test_data).to_dict()

    def test_framework(self):
        """Test framework attribute"""
        test_tf_model = TensorFlowModel(self.myTFModel, tmp_model_dir)
        assert test_tf_model.framework == "tensorflow"

    def test_prepare_default(self):
        """
        Test if TensorFlowModel.prepare default serialization
        """
        test_tf_model = TensorFlowModel(self.myTFModel, tmp_model_dir)
        test_tf_model.prepare(
            inference_conda_env=self.inference_conda_env,
            inference_python_version=self.inference_python_version,
            training_conda_env=self.training_conda_env,
            training_python_version=self.training_python_version,
            force_overwrite=True,
        )
        assert os.path.exists(tmp_model_dir + "model.h5")

    def test_prepare_onnx_with_input_signature(self):
        """
        Test if TensorFlowModel.prepare onnx serialization
        """
        test_tf_model = TensorFlowModel(self.myTFModel, tmp_model_dir)
        test_tf_model.prepare(
            inference_conda_env=self.inference_conda_env,
            inference_python_version=self.inference_python_version,
            training_conda_env=self.training_conda_env,
            training_python_version=self.training_python_version,
            force_overwrite=True,
            as_onnx=True,
            input_signature=self.dummy_input,
        )
        assert os.path.exists(tmp_model_dir + "model.onnx")

    def test_prepare_onnx_with_X_sample(self):
        """
        Test if TensorFlowModel.prepare raise expected error
        """
        test_tf_model = TensorFlowModel(self.myTFModel, tmp_model_dir)
        test_tf_model.prepare(
            inference_conda_env=self.inference_conda_env,
            inference_python_version=self.inference_python_version,
            training_conda_env=self.training_conda_env,
            training_python_version=self.training_python_version,
            force_overwrite=True,
            as_onnx=True,
            X_sample=self.x_test[:1],
        )
        assert isinstance(test_tf_model.verify(self.x_test[:1].tolist()), dict)

    def test_verify_onnx_without_input(self):
        """
        Test if TensorFlowModel.verify in onnx serialization
        """
        test_tf_model = TensorFlowModel(self.myTFModel, tmp_model_dir)
        test_tf_model.prepare(
            inference_conda_env=self.inference_conda_env,
            inference_python_version=self.inference_python_version,
            training_conda_env=self.training_conda_env,
            training_python_version=self.training_python_version,
            force_overwrite=True,
            as_onnx=True,
        )
        assert isinstance(test_tf_model.verify(self.x_test[:1].tolist()), dict)

    def test_verify_default(self):
        """
        Test if TensorFlowModel.verify in default serialization
        """
        test_tf_model = TensorFlowModel(self.myTFModel, tmp_model_dir)
        test_tf_model.prepare(
            inference_conda_env=self.inference_conda_env,
            inference_python_version=self.inference_python_version,
            training_conda_env=self.training_conda_env,
            training_python_version=self.training_python_version,
            force_overwrite=True,
        )
        assert isinstance(test_tf_model.verify(self.x_test[:1]), dict)


def teardown_module():
    shutil.rmtree(tmp_model_dir)
