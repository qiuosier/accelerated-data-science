#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


import base64
import os
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from ads.common import logger
from ads.model.extractor.pytorch_extractor import PytorchExtractor
from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)
from ads.common.data_serializer import InputDataSerializer
from ads.model.generic_model import FrameworkSpecificModel
from ads.model.model_properties import ModelProperties

ONNX_MODEL_FILE_NAME = "model.onnx"
PYTORCH_MODEL_FILE_NAME = "model.pt"


class PyTorchModel(FrameworkSpecificModel):
    """PyTorchModel class for estimators from Pytorch framework.

    Attributes
    ----------
    algorithm: str
        The algorithm of the model.
    artifact_dir: str
        Artifact directory to store the files needed for deployment.
    auth: Dict
        Default authentication is set using the `ads.set_auth` API. To override the
        default, use the `ads.common.auth.api_keys` or `ads.common.auth.resource_principal` to create
        an authentication signer to instantiate an IdentityClient object.
    ds_client: DataScienceClient
        The data science client used by model deployment.
    estimator: Callable
        A trained pytorch estimator/model using Pytorch.
    framework: str
        "pytorch", the framework name of the model.
    hyperparameter: dict
        The hyperparameters of the estimator.
    metadata_custom: ModelCustomMetadata
        The model custom metadata.
    metadata_provenance: ModelProvenanceMetadata
        The model provenance metadata.
    metadata_taxonomy: ModelTaxonomyMetadata
        The model taxonomy metadata.
    model_artifact: ModelArtifact
        This is built by calling prepare.
    model_deployment: ModelDeployment
        A ModelDeployment instance.
    model_file_name: str
        Name of the serialized model.
    model_id: str
        The model ID.
    properties: ModelProperties
        ModelProperties object required to save and deploy model.
    runtime_info: RuntimeInfo
        A RuntimeInfo instance.
    schema_input: Schema
        Schema describes the structure of the input data.
    schema_output: Schema
        Schema describes the structure of the output data.
    serialize: bool
        Whether to serialize the model to pkl file by default. If False, you need to serialize the model manually,
        save it under artifact_dir and update the score.py manually.
    version: str
        The framework version of the model.

    Methods
    -------
    delete_deployment(...)
        Deletes the current model deployment.
    deploy(..., **kwargs)
        Deploys a model.
    from_model_artifact(uri, model_file_name, artifact_dir, ..., **kwargs)
        Loads model from the specified folder, or zip/tar archive.
    from_model_catalog(model_id, model_file_name, artifact_dir, ..., **kwargs)
        Loads model from model catalog.
    introspect(...)
        Runs model introspection.
    predict(data, ...)
        Returns prediction of input data run against the model deployment endpoint.
    prepare(..., **kwargs)
        Prepare and save the score.py, serialized model and runtime.yaml file.
    reload(...)
        Reloads the model artifact files: `score.py` and the `runtime.yaml`.
    save(..., **kwargs)
        Saves model artifacts to the model catalog.
    summary_status(...)
        Gets a summary table of the current status.
    verify(data, ...)
        Tests if deployment works in local environment.

    Examples
    --------
    >>> torch_model = PyTorchModel(estimator=torch_estimator,
    ... artifact_dir=tmp_model_dir)
    >>> inference_conda_env = "generalml_p37_cpu_v1"

    >>> torch_model.prepare(inference_conda_env=inference_conda_env, force_overwrite=True)
    >>> torch_model.reload()
    >>> torch_model.verify(...)
    >>> torch_model.save()
    >>> model_deployment = torch_model.deploy(wait_for_completion=False)
    >>> torch_model.predict(...)
    """

    _PREFIX = "pytorch"

    @runtime_dependency(module="torch", install_from=OptionalDependency.PYTORCH)
    def __init__(
        self,
        estimator: callable,
        artifact_dir: str,
        properties: Optional[ModelProperties] = None,
        auth: Dict = None,
        **kwargs,
    ):
        """
        Initiates a PyTorchModel instance.

        Parameters
        ----------
        estimator: callable
            Any model object generated by pytorch framework
        artifact_dir: str
            artifact directory to store the files needed for deployment.
        properties: (ModelProperties, optional). Defaults to None.
            ModelProperties object required to save and deploy model.
        auth :(Dict, optional). Defaults to None.
            The default authetication is set using `ads.set_auth` API. If you need to override the
            default, use the `ads.common.auth.api_keys` or `ads.common.auth.resource_principal` to create appropriate
            authentication signer and kwargs required to instantiate IdentityClient object.

        Returns
        -------
        PyTorchModel
            PyTorchModel instance.
        """
        super().__init__(
            estimator=estimator,
            artifact_dir=artifact_dir,
            properties=properties,
            auth=auth,
            **kwargs,
        )
        self._extractor = PytorchExtractor(estimator)
        self.framework = self._extractor.framework
        self.algorithm = self._extractor.algorithm
        self.version = self._extractor.version
        self.hyperparameter = self._extractor.hyperparameter
        self.version = torch.__version__

    def _handle_model_file_name(self, as_onnx: bool, model_file_name: str) -> str:
        """
        Process file name for saving model.
        For ONNX model file name must be ending with ".onnx".
        For joblib model file name must be ending with ".joblib".
        If not specified, use "model.onnx" for ONNX model and "model.joblib" for joblib model.

        Parameters
        ----------
        as_onnx: bool
            If set as True, convert into ONNX model.
        model_file_name: str
            File name for saving model.

        Returns
        -------
        str
            Processed file name.
        """
        if not model_file_name:
            return ONNX_MODEL_FILE_NAME if as_onnx else PYTORCH_MODEL_FILE_NAME
        if as_onnx:
            if model_file_name and not model_file_name.endswith(".onnx"):
                raise ValueError(
                    "`model_file_name` has to be ending with `.onnx` for onnx format."
                )
        else:
            if model_file_name and not (
                model_file_name.endswith(".pt") or model_file_name.endswith(".pth")
            ):
                raise ValueError(
                    "`model_file_name` has to be ending with `.pt` or `.pth` "
                    "for Pytorch format."
                )
        return model_file_name

    @runtime_dependency(module="torch", install_from=OptionalDependency.PYTORCH)
    def serialize_model(
        self,
        as_onnx: bool = False,
        force_overwrite: bool = False,
        X_sample: Optional[
            Union[
                Dict,
                str,
                List,
                Tuple,
                np.ndarray,
                pd.core.series.Series,
                pd.core.frame.DataFrame,
            ]
        ] = None,
        use_torch_script: bool = None,
        **kwargs,
    ) -> None:
        """
        Serialize and save Pytorch model using ONNX or model specific method.

        Parameters
        ----------
        as_onnx: (bool, optional). Defaults to False.
            If set as True, convert into ONNX model.
        force_overwrite: (bool, optional). Defaults to False.
            If set as True, overwrite serialized model if exists.
        X_sample: Union[list, tuple, pd.Series, np.ndarray, pd.DataFrame]. Defaults to None.
            A sample of input data that will be used to generate input schema and detect onnx_args.
        use_torch_script:  (bool, optional). Defaults to None (If the default value has not been changed, it will be set as `False`).
            If set as `True`, the model will be serialized as a TorchScript program. Check https://pytorch.org/tutorials/beginner/saving_loading_models.html#export-load-model-in-torchscript-format for more details.
            If set as `False`, it will only save the trained model’s learned parameters, and the score.py
            need to be modified to construct the model class instance first. Check https://pytorch.org/tutorials/beginner/saving_loading_models.html#save-load-state-dict-recommended for more details.
        **kwargs: optional params used to serialize pytorch model to onnx,
        including the following:
            onnx_args: (tuple or torch.Tensor), default to None
            Contains model inputs such that model(onnx_args) is a valid
            invocation of the model. Can be structured either as: 1) ONLY A
            TUPLE OF ARGUMENTS; 2) A TENSOR; 3) A TUPLE OF ARGUMENTS ENDING
            WITH A DICTIONARY OF NAMED ARGUMENTS
            input_names: (List[str], optional). Names to assign to the input
            nodes of the graph, in order.
            output_names: (List[str], optional). Names to assign to the output nodes of the graph, in order.
            dynamic_axes: (dict, optional), default to None. Specify axes of tensors as dynamic (i.e. known only at run-time).

        Returns
        -------
        None
            Nothing.
        """
        model_path = os.path.join(self.artifact_dir, self.model_file_name)

        if os.path.exists(model_path) and not force_overwrite:
            raise ValueError(
                f"The {model_path} already exists, set force_overwrite to True if you wish to overwrite."
            )

        os.makedirs(self.artifact_dir, exist_ok=True)

        if use_torch_script is None:
            logger.warning(
                "In future the models will be saved in TorchScript format by default. Currently saving it using torch.save method."
                "Set `use_torch_script` as `True` to serialize the model as a TorchScript program by `torch.jit.save()` "
                "and loaded using `torch.jit.load()` in score.py. "
                "You don't need to modify `load_model()` in score.py to load the model."
                "Check https://pytorch.org/tutorials/beginner/saving_loading_models.html#export-load-model-in-torchscript-format for more details."
                "Set `use_torch_script` as `False` to save only the model parameters."
                "The model class instance must be constructed before "
                "loading parameters in the perdict function of score.py."
                "Check https://pytorch.org/tutorials/beginner/saving_loading_models.html#save-load-state-dict-recommended for more details."
            )
            use_torch_script = False

        if as_onnx:
            onnx_args = kwargs.get("onnx_args", None)

            input_names = kwargs.get("input_names", ["input"])
            output_names = kwargs.get("output_names", ["output"])
            dynamic_axes = kwargs.get("dynamic_axes", None)

            self.to_onnx(
                path=model_path,
                onnx_args=onnx_args,
                X_sample=X_sample,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )

        elif use_torch_script:
            compiled_model = torch.jit.script(self.estimator)
            torch.jit.save(compiled_model, model_path)

        else:
            torch.save(self.estimator.state_dict(), model_path)

    @runtime_dependency(module="torch", install_from=OptionalDependency.PYTORCH)
    def to_onnx(
        self,
        path: str = None,
        onnx_args=None,
        X_sample: Optional[
            Union[
                Dict,
                str,
                List,
                Tuple,
                np.ndarray,
                pd.core.series.Series,
                pd.core.frame.DataFrame,
            ]
        ] = None,
        input_names: List[str] = ["input"],
        output_names: List[str] = ["output"],
        dynamic_axes=None,
    ):
        """
        Exports the given Pytorch model into ONNX format.

        Parameters
        ----------
        path: str, default to None
            Path to save the serialized model.
        onnx_args: (tuple or torch.Tensor), default to None
            Contains model inputs such that model(onnx_args) is a valid
            invocation of the model. Can be structured either as: 1) ONLY A
            TUPLE OF ARGUMENTS; 2) A TENSOR; 3) A TUPLE OF ARGUMENTS ENDING
            WITH A DICTIONARY OF NAMED ARGUMENTS
        X_sample: Union[list, tuple, pd.Series, np.ndarray, pd.DataFrame]. Defaults to None.
            A sample of input data that will be used to generate input schema and detect onnx_args.
        input_names: (List[str], optional). Defaults to ["input"].
            Names to assign to the input nodes of the graph, in order.
        output_names: (List[str], optional). Defaults to ["output"].
            Names to assign to the output nodes of the graph, in order.
        dynamic_axes: (dict, optional). Defaults to None.
            Specify axes of tensors as dynamic (i.e. known only at run-time).

        Returns
        -------
        None
            Nothing

        Raises
        ------
        AssertionError
            if onnx module is not support by the current version of torch
        ValueError
            if X_sample is not provided
            if path is not provided
        """

        assert hasattr(torch, "onnx"), (
            f"This version of pytorch {torch.__version__} does not appear to support onnx "
            "conversion."
        )

        if onnx_args is None:
            if X_sample is not None:
                logger.warning(
                    "Since `onnx_args` is not provided, `onnx_args` is "
                    "detected from `X_sample` to export pytorch model as onnx."
                )
                onnx_args = X_sample
            else:
                raise ValueError(
                    "`onnx_args` can not be detected. The parameter `onnx_args` must be provided to export pytorch model as onnx."
                )

        if not path:
            raise ValueError(
                "The parameter `path` must be provided to save the model file."
            )

        torch.onnx.export(
            self.estimator,
            args=onnx_args,
            f=path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

    @runtime_dependency(module="torch", install_from=OptionalDependency.PYTORCH)
    def get_data_serializer(
        self,
        data: Union[
            Dict,
            str,
            List,
            np.ndarray,
            pd.core.series.Series,
            pd.core.frame.DataFrame,
            "torch.Tensor",
        ],
        data_type: str = None,
    ):
        """Returns serializable input data.

        Parameters
        ----------
        data: Union[Dict, str, list, numpy.ndarray, pd.core.series.Series,
        pd.core.frame.DataFrame, torch.Tensor]
            Data expected by the model deployment predict API.
        data_type: str
            Type of the data.

        Returns
        -------
        InputDataSerializer
            A class containing serialized input data and original data type
            information.

        Raises
        ------
        TypeError
            if provided data type is not supported.
        """
        data_type = data_type if data_type else type(data)
        if data_type == "image":
            try:
                import torchvision.transforms as transforms

                convert_tensor = transforms.ToTensor()
                data = convert_tensor(data)
                data_type = str(type(data))
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    f"The `torchvision` module was not found. Please run "
                    f"`pip install {OptionalDependency.PYTORCH}`."
                )
        if isinstance(data, torch.Tensor):
            buffer = BytesIO()
            torch.save(data, buffer)
            data = base64.b64encode(buffer.getvalue()).decode("utf-8")
        try:
            return InputDataSerializer(data, data_type=data_type)
        except:
            raise TypeError(
                "The supported data types are Dict, str, list, bytes,"
                "numpy.ndarray, pd.core.series.Series, "
                "pd.core.frame.DataFrame, torch.Tensor. Please "
                "convert to the supported data types first. "
            )
