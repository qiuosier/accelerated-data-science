import os
import tempfile
from pathlib import Path
from unittest.mock import patch
import json

import pytest

from ads.opctl.backend.ads_dataflow import DataFlowBackend


class TestDataFlowBackend:
    def test_dataflow_apply(self):
        config = {
            "execution": {
                "backend": "dataflow",
                "oci_profile": "DEFAULT",
                "oci_config": "~/.oci/config",
            }
        }
        with pytest.raises(ValueError):
            DataFlowBackend(config).apply()

    @patch("ads.jobs.builders.infrastructure.dataflow.DataFlowApp.create")
    @patch("ads.opctl.backend.ads_dataflow.Job.run")
    @patch("ads.jobs.DataFlow._upload_file")
    def test_dataflow_run(self, file_upload, job_run, job_create):
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "src"))
            Path(os.path.join(td, "src", "main.py")).touch()
            config = {
                "execution": {
                    "backend": "dataflow",
                    "source_folder": os.path.join(td, "src"),
                    "entrypoint": "main.py",
                    "command": "--opt v arg",
                    "auth": "api_key",
                    "oci_profile": "DEFAULT",
                    "oci_config": "~/.oci/config",
                    "archive": "archive.zip",
                },
                "infrastructure": {
                    "compartment_id": "ocid1.compartment.oc1..aaaaaaaapvb3hearqum6wjvlcpzm5ptfxqa7xfftpth4h72xx46ygavkqteq",
                    "driver_shape": "VM.Standard2.1",
                    "executor_shape": "VM.Standard2.1",
                    "logs_bucket_uri": "oci://jize-dev@ociodscdev/",
                    "configurations": json.dumps({"spark.driver.memory": "512m"}),
                    "script_bucket": "oci://jize-dev2@ociodscdev/test-dataflow/",
                    "archive_bucket": "oci://jize-dev2@ociodscdev/test-dataflow/",
                },
            }

            DataFlowBackend(config).run()
            job_create.assert_called()
            job_run.assert_called()
            file_upload.assert_any_call(
                os.path.join(td, "src", "main.py"),
                "oci://jize-dev2@ociodscdev/test-dataflow/",
                False,
            )
            file_upload.assert_any_call(
                "archive.zip",
                "oci://jize-dev2@ociodscdev/test-dataflow/",
                False,
            )
