#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.vault.vault import Vault
from collections import namedtuple
from datetime import datetime, timezone
from oci.secrets.models import SecretBundle, Base64SecretBundleContentDetails
from oci.vault.models import Secret
from unittest.mock import MagicMock
import base64
import json


class TestVault:
    """Contains test cases for ads.vault.vault.py"""

    vault = Vault(
        vault_id="ocid1.vault.oc1.iad.bbpuia5yaacuu.abuwcljroewpr3krxa2yajvqos6my3b2lkuim664cjvnidzqj4vqkw62cxla",
        key_id="ocid1.key.oc1.iad.bbpuia5yaacuu.abuwcljshdypfrt5lhd4xm66oyj3fthpzihzniev4cjobijcfmahvedqrjvq",
    )
    vault.secret_client = MagicMock()
    vault.vaults_client_composite = MagicMock()

    credential = {
        "database_name": "db201910031555_high",
        "username": "admin",
        "password": "MySecretPassword",
    }

    updated_credential = {
        "database_name": "db201910031555_high",
        "username": "admin",
        "password": "updated_password",
    }

    secret_ocid = "ocid1.vaultsecret.oc1.iad.amaaaaaav66vvniawm3hpm7kqxke63c7hpv4o5ugajv45mjvyuajhlminh7q"
    date_time = datetime(2021, 7, 13, 18, 24, 42, 110000, tzinfo=timezone.utc)

    def test_create_secret(self):
        """Test vault.create_secret()."""
        wrapper = namedtuple("wrapper", ["data"])
        secret_response = wrapper(data=Secret(id=self.secret_ocid))
        self.vault.vaults_client_composite.create_secret_and_wait_for_state = MagicMock(
            return_value=secret_response
        )
        secret_id = self.vault.create_secret(self.credential)
        assert isinstance(secret_id, str)
        assert secret_id == self.secret_ocid

    def test_get_secret(self):
        """Test vault.get_secret()."""
        content = base64.b64encode(json.dumps(self.credential).encode("ascii")).decode(
            "ascii"
        )
        secret_bundle_content = Base64SecretBundleContentDetails(
            content_type="BASE64", content=content
        )
        wrapper = namedtuple("wrapper", ["data"])
        secret_response = wrapper(
            data=SecretBundle(
                secret_id=self.secret_ocid,
                time_created=self.date_time.isoformat(),
                version_name="Testing",
                version_number="111",
                secret_bundle_content=secret_bundle_content,
            )
        )
        self.vault.secret_client.get_secret_bundle = MagicMock(
            return_value=secret_response
        )
        secret_content = self.vault.get_secret(self.secret_ocid)
        assert isinstance(secret_content, dict)

    def test_update_secret(self):
        """Test vault.update_secret()."""
        wrapper = namedtuple("wrapper", ["data"])
        secret_response = wrapper(data=Secret(id=self.secret_ocid))
        self.vault.vaults_client_composite.update_secret_and_wait_for_state = MagicMock(
            return_value=secret_response
        )
        update_secret_ocid = self.vault.update_secret(
            self.secret_ocid, self.updated_credential
        )
        assert isinstance(update_secret_ocid, str)
        assert update_secret_ocid == self.secret_ocid
