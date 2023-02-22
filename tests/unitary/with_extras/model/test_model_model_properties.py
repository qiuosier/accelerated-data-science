import os
from unittest import mock

from ads.model.model_properties import ModelProperties

mock_env_variables = {
    "PROJECT_OCID": "test_project_id",
    "JOB_RUN_OCID": "test_job_run_ocid",
    "NB_SESSION_OCID": "test_nb_session_ocid",
    "NB_SESSION_COMPARTMENT_OCID": "test_nb_session_compartment_ocid",
}


class TestModelProperties:
    @mock.patch.dict(os.environ, mock_env_variables, clear=True)
    def test__adjust_with_env(self):
        """Tests adjustment env variables."""
        model_properties = ModelProperties()
        assert model_properties.project_id == None
        assert model_properties.training_resource_id == None
        assert model_properties.compartment_id == None
        model_properties._adjust_with_env()
        assert model_properties.project_id == mock_env_variables["PROJECT_OCID"]
        assert (
            model_properties.training_resource_id == mock_env_variables["JOB_RUN_OCID"]
        )
        assert (
            model_properties.compartment_id
            == mock_env_variables["NB_SESSION_COMPARTMENT_OCID"]
        )
