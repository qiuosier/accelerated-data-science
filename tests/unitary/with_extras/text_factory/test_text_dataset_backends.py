#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import tempfile
import gc

import fsspec
import pytest
from ads.text_dataset.backends import OITCC, Base, Tika, PDFPlumber


sources = ["simple_content", "big5_encode", "ISO-8859-1_encode"]
expected_outputs = [
    [
        "this is a dummy text file with punctuations and numbers:",
        "1234",
        "~%()",
    ],
    [
        "失在浮淺，其辭多鄙俗。",
        "得其質直，其辭多古語。",
    ],
    [
        "DIE Drachenkˆpfe unserer Boote bogen um das gelbe Segel. Die Parade",
        "vollzog sich in elegantem Rauschen, wir wollten mit Ostwind an das",
        "andere Ende, bei Ostwind anderthalb Stunden dachten wir, es waren",
        "Kilometer. Die Flottille lag in einer Linie.",
    ],
]


def get_test_dataset_path(filename):
    return os.path.join(
        f"{os.path.dirname(os.path.abspath(__file__))}/../../../data/text_dataset_test_data",
        filename,
    )


def _test_read_line(fmt, backend, idx=None):
    if idx is None:
        idx = range(len(sources))
    for i in idx:
        fhandler = fsspec.open(get_test_dataset_path(sources[i] + "." + fmt))
        lines = [
            line.strip()
            for line in backend.read_line(fhandler)
            if len(line.strip()) > 0
        ]
        assert len(lines) > 0
        assert "".join(lines).replace("\n", "") == "".join(expected_outputs[i]).replace(
            "\n", ""
        )


def _test_read_text(fmt, backend, idx=None, check_meta=False):
    if idx is None:
        idx = range(len(sources))
    for i in idx:
        fhandler = fsspec.open(get_test_dataset_path(sources[i] + "." + fmt))
        for text in backend.read_text(fhandler):
            assert "".join(text.strip().replace("\n", "")) == "".join(
                expected_outputs[i]
            )
        if check_meta:
            assert len(backend.get_metadata(fhandler)) > 0


def _test_convert_text(fmt, backend, idx=None):
    if idx is None:
        idx = range(len(sources))
    for i in idx:
        fhandler = fsspec.open(get_test_dataset_path(sources[i] + "." + fmt))
        with tempfile.TemporaryDirectory() as d:
            dest = backend.convert_to_text(fhandler, d)
            for text in Base().read_text(fsspec.open(dest)):
                assert text.strip().replace("\n", "") == "".join(expected_outputs[i])


class TestBase:

    backend = Base()

    def test_read_line(self):
        _test_read_line("txt", self.backend, [0])

    def test_read_text(self):
        _test_read_text("txt", self.backend, [0])

    def test_convert_text(self):
        _test_convert_text("txt", self.backend, [0])


class TestTika:

    backend = Tika()

    def test_read_line_txt(self):
        _test_read_line("txt", self.backend)

    def test_read_line_pdf(self):
        _test_read_line("pdf", self.backend)

    def test_read_line_docx(self):
        _test_read_line("docx", self.backend)

    def test_read_text_txt(self):
        _test_read_text("txt", self.backend, check_meta=True)

    def test_read_text_pdf(self):
        _test_read_text("pdf", self.backend, check_meta=True)

    def test_read_text_docx(self):
        _test_read_text("docx", self.backend, check_meta=True)

    def test_convert_text_txt(self):
        _test_convert_text("txt", self.backend)

    def test_convert_text_pdf(self):
        _test_convert_text("pdf", self.backend)

    def test_convert_text_docx(self):
        _test_convert_text("docx", self.backend)


class TestPDFPlumber:

    backend = PDFPlumber()

    def test_read_line_pdf(self):
        _test_read_line("pdf", self.backend)

    def test_read_text_pdf(self):
        _test_read_text("pdf", self.backend, check_meta=True)

    def test_convert_text_pdf(self):
        _test_convert_text("pdf", self.backend)


@pytest.mark.skip(
    "OIT-CC has encoding/decoding issues. Communicating with the team on resolving those."
)
class TestOITCC:

    backend = OITCC()

    def test_read_line_docx(self):
        _test_read_line("docx", self.backend, [0])

    def test_read_text_docx(self):
        _test_read_text("docx", self.backend, [0])

    def test_convert_text_docx(self):
        _test_convert_text("docx", self.backend, [0])

    def test_read_line_pdf(self):
        _test_read_line("pdf", self.backend, [0])

    def test_read_text_pdf(self):
        _test_read_text("pdf", self.backend, [0])

    def test_convert_text_pdf(self):
        _test_convert_text("pdf", self.backend, [0])
