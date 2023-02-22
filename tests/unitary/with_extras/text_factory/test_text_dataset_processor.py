import os

import fsspec
import pytest
from ads.text_dataset.backends import Base, Tika, PDFPlumber
from ads.text_dataset.extractor import (
    FileProcessor,
    FileProcessorFactory,
    NotSupportedError,
    PDFProcessor,
    WordProcessor,
)


def get_test_dataset_path(filename):
    return os.path.join(
        f"{os.path.dirname(os.path.abspath(__file__))}/../../../data/text_dataset_test_data/{filename}"
    )


class DummyCustomBackend(Base):
    def read_line(self, fhandler):
        with fhandler as f:
            for line in f:
                yield line.decode()[::-1]

    def read_text(self, fhandler):
        with fhandler as f:
            yield f.read().decode()[::-1]


class DummyProcessor(FileProcessor):
    pass


class TestFileProcessor:
    def test_custom_backend(self):
        fhandler = fsspec.open(get_test_dataset_path("simple_content.txt"))
        processor = FileProcessor().backend(DummyCustomBackend())
        lines = [line for line in processor.read_line(fhandler)]
        assert len(lines) == 3
        assert (
            lines[0][::-1].strip()
            == "this is a dummy text file with punctuations and numbers:"
        )
        assert lines[1][::-1].strip() == "1234"
        assert lines[2][::-1].strip() == "~%()"

    def test_register_processor(self):
        FileProcessorFactory.register("abc", DummyProcessor)
        assert FileProcessorFactory.processor_map["abc"] == DummyProcessor

    def test_set_backend(self):
        processor = FileProcessor()
        processor.backend("tika")
        assert isinstance(processor._backend, Tika)
        processor.backend(Tika())
        assert isinstance(processor._backend, Tika)
        with pytest.raises(NotSupportedError):
            processor.backend("xxxx")

        processor = PDFProcessor()
        processor.backend("tika")
        assert isinstance(processor._backend, Tika)
        processor.backend("pdfplumber")
        assert isinstance(processor._backend, PDFPlumber)

        processor = WordProcessor()
        processor.backend("tika")
        assert isinstance(processor._backend, Tika)

    def test_set_format(self):
        factory = FileProcessorFactory()
        assert isinstance(factory.get_processor("pdf")(), PDFProcessor)
        assert isinstance(factory.get_processor("docx")(), WordProcessor)
        assert isinstance(factory.get_processor("xxx")(), FileProcessor)
