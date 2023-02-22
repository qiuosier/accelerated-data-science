import os
import tempfile
import pytest

from ads.text_dataset.dataset import TextDatasetFactory as textfactory
from ads.text_dataset.options import Options
from ads.text_dataset.extractor import NotSupportedError


def get_test_dataset_path(filename):
    return os.path.join(
        f"{os.path.dirname(os.path.abspath(__file__))}/../../../data/text_dataset_test_data/{filename}"
    )


class TestPDF:
    @pytest.fixture()
    def conda_prefix(self):
        with tempfile.TemporaryDirectory() as td:
            prefix = os.environ["CONDA_PREFIX"]
            yield td
            os.environ["CONDA_PREFIX"] = prefix

    def test_no_tika_jar_scenario(self, conda_prefix):
        os.environ["CONDA_PREFIX"] = conda_prefix
        with pytest.raises(NotSupportedError):
            textfactory.format("pdf").engine("pandas").option(
                Options.FILE_METADATA, {"extract": ["Author"]}
            ).read_line(
                get_test_dataset_path("pdfs/*.pdf"),
                n_lines_per_file=10,
            )

        del os.environ["CONDA_PREFIX"]
        with pytest.raises(NotSupportedError):
            textfactory.format("pdf").engine("pandas").option(
                Options.FILE_METADATA, {"extract": ["Author"]}
            ).read_line(
                get_test_dataset_path("pdfs/*.pdf"),
                n_lines_per_file=10,
            )

        textfactory.format("pdf").backend("pdfplumber").read_line(
            get_test_dataset_path("pdfs/*.pdf"),
            n_lines_per_file=10,
        )

    def test_read_line(self):
        ds_local = (
            textfactory.format("pdf")
            .engine("pandas")
            .option(Options.FILE_METADATA, {"extract": ["Author"]})
            .read_line(
                get_test_dataset_path("pdfs/*.pdf"),
                n_lines_per_file=10,
            )
        )

        assert ds_local.shape == (30, 2)

        ds_local = (
            textfactory.format("pdf")
            .option(Options.FILE_NAME)
            .backend("pdfplumber")
            .read_line(
                get_test_dataset_path("pdfs/*.pdf"),
                n_lines_per_file=10,
            )
        )

        assert len([row for row in ds_local]) == 30

    def test_read_line_with_regex(self):

        ds_local = (
            textfactory.format("pdf")
            .engine("pandas")
            .read_line(
                get_test_dataset_path("pdfs/COVID-0.pdf"),
                udf=r"[^.]*effective[^.]*",
            )
        )

        assert ds_local.shape == (2, 1)

        ds_local = (
            textfactory.format("pdf")
            .backend("pdfplumber")
            .backend("pdfplumber")
            .read_line(
                get_test_dataset_path("pdfs/COVID-0.pdf"),
                udf=r"[^.]*effective[^.]*",
            )
        )

        assert len([row for row in ds_local]) == 2

    def test_read_line_with_fn(self):

        ds_local = (
            textfactory.format("pdf")
            .engine("pandas")
            .read_line(
                get_test_dataset_path("pdfs/COVID-0.pdf"),
                udf=lambda line: line.split(),
                n_lines_per_file=10,
            )
        )

        assert ds_local.shape == (10, 7)

        ds_local = (
            textfactory.format("pdf")
            .backend("pdfplumber")
            .read_line(
                get_test_dataset_path("pdfs/COVID-0.pdf"),
                udf=lambda line: line.split(),
                n_lines_per_file=10,
            )
        )

        assert len([row for row in ds_local]) == 10

    def test_read_text(self):
        ds_local = (
            textfactory.format("pdf")
            .engine("pandas")
            .option(Options.FILE_METADATA, {"extract": ["Author"]})
            .read_text(
                get_test_dataset_path("pdfs/*.pdf"),
                total_files=2,
            )
        )

        assert ds_local.shape == (2, 2)

        ds_local = (
            textfactory.format("pdf")
            .backend("pdfplumber")
            .option(Options.FILE_NAME)
            .read_text(
                get_test_dataset_path("pdfs/*.pdf"),
                total_files=2,
            )
        )

        assert len([row for row in ds_local]) == 2

    def test_convert_to_text(self):
        with tempfile.TemporaryDirectory() as td:
            textfactory.format("pdf").convert_to_text(
                get_test_dataset_path("pdfs/COVID-0.pdf"), td
            )
            assert os.path.exists(os.path.join(td, "COVID-0.txt"))

        with tempfile.TemporaryDirectory() as td:
            textfactory.format("pdf").backend("pdfplumber").convert_to_text(
                get_test_dataset_path("pdfs/COVID-0.pdf"), td
            )
            assert os.path.exists(os.path.join(td, "COVID-0.txt"))

    def test_metadata(self):
        metadata = textfactory.format("pdf").metadata_all(
            get_test_dataset_path("pdfs/COVID-0.pdf")
        )
        assert next(metadata)["dc:creator"][0] == "Tung Thanh Le"

        metadata = (
            textfactory.format("pdf")
            .backend("pdfplumber")
            .metadata_all(get_test_dataset_path("pdfs/COVID-0.pdf"))
        )
        assert next(metadata)["Title"] == "The COVID-19 vaccine development landscape"

        schema = textfactory.format("pdf").metadata_schema(
            get_test_dataset_path("pdfs/COVID-0.pdf"),
            n_files=10,
        )
        assert len(schema) == len(set(schema))

        schema = (
            textfactory.format("pdf")
            .backend("pdfplumber")
            .metadata_schema(
                get_test_dataset_path("pdfs/COVID-0.pdf"),
                n_files=10,
            )
        )
        assert len(schema) == len(set(schema))


class TestMSWord:
    def test_read_line(self):
        ds_local = (
            textfactory.format("docx")
            .engine("pandas")
            .option(Options.FILE_METADATA, {"extract": ["Character Count"]})
            .option(Options.FILE_METADATA, {"extract": ["Paragraph-Count"]})
            .read_line(
                [
                    get_test_dataset_path("docs/[A-Za-z]*.docx"),
                    get_test_dataset_path("docs/[A-Za-z]*.doc"),
                ],
                n_lines_per_file=10,
            )
        )

        assert ds_local.shape == (30, 3)

    def test_read_line_with_regex(self):

        ds_local = (
            textfactory.format("docx")
            .option(Options.FILE_METADATA, {"extract": ["Character Count"]})
            .option(Options.FILE_METADATA, {"extract": ["Paragraph-Count"]})
            .read_line(
                [
                    get_test_dataset_path("docs/[A-Za-z]*.docx"),
                    get_test_dataset_path("docs/[A-Za-z]*.doc"),
                ],
                udf=r"[^.]*University[^.]*",
            )
        )

        assert len([row for row in ds_local]) == 4

    def test_read_line_with_fn(self):

        ds_local = (
            textfactory.format("docx")
            .engine("pandas")
            .option(Options.FILE_METADATA, {"extract": ["Character Count"]})
            .option(Options.FILE_METADATA, {"extract": ["Paragraph-Count"]})
            .read_line(
                [
                    get_test_dataset_path("docs/[A-Za-z]*.docx"),
                    get_test_dataset_path("docs/[A-Za-z]*.doc"),
                ],
                udf=lambda line: [
                    w.strip() for w in line.strip().split() if len(w.strip()) > 0
                ],
                n_lines_per_file=10,
            )
        )

        assert ds_local.shape == (30, 42)

    def test_read_text(self):
        ds_local = (
            textfactory.format("docx")
            .option(Options.FILE_METADATA, {"extract": ["Character Count"]})
            .option(Options.FILE_METADATA, {"extract": ["Paragraph-Count"]})
            .read_text(
                [
                    get_test_dataset_path("docs/[A-Za-z]*.docx"),
                    get_test_dataset_path("docs/[A-Za-z]*.doc"),
                ],
                total_files=2,
            )
        )

        assert len([row for row in ds_local]) == 2

    def test_convert_to_text(self):
        with tempfile.TemporaryDirectory() as td:
            textfactory.format("docx").convert_to_text(
                [
                    get_test_dataset_path("docs/[A-Za-z]*.docx"),
                    get_test_dataset_path("docs/[A-Za-z]*.doc"),
                ],
                td,
            )
            assert len(os.listdir(td)) == 3

    def test_metadata(self):
        metadata = textfactory.format("docx").metadata_all(
            get_test_dataset_path("docs/[A-Za-z]*.doc")
        )
        assert next(metadata)["meta:last-author"] == "Wouter Olsthoorn"

        schema = textfactory.format("docx").metadata_schema(
            get_test_dataset_path("docs/[A-Za-z]*.docx"),
            n_files=10,
        )
        assert len(schema) == len(set(schema))


class TestTxt:
    def test_read_line(self):
        ds_local = textfactory.format("txt").read_line(
            get_test_dataset_path("logs/*.log"),
            n_lines_per_file=10,
        )

        assert len([row for row in ds_local]) == 10

        ds_local = (
            textfactory.format("txt")
            .backend("tika")
            .engine("pandas")
            .read_line(
                get_test_dataset_path("logs/*.log"),
                n_lines_per_file=10,
            )
        )

        assert ds_local.shape == (10, 1)

    def test_read_line_with_regex(self):

        ds_local = (
            textfactory.format("txt")
            .engine("pandas")
            .read_line(
                get_test_dataset_path("logs/*.log"),
                udf=r"^\[(\S+)\s(\S+)\s(\d+)\s(\d+\:\d+\:\d+)\s(\d+)]\s(\S+)\s(\S+)\s(\S+)\s(\S+)",
                df_args={
                    "columns": [
                        "day",
                        "month",
                        "date",
                        "time",
                        "year",
                        "type",
                        "method",
                        "status",
                        "file",
                    ]
                },
                n_lines_per_file=10,
            )
        )

        assert ds_local.shape == (10, 9)

        ds_local = (
            textfactory.format("txt")
            .backend("tika")
            .read_line(
                get_test_dataset_path("logs/*.log"),
                udf=r"^\[(\S+)\s(\S+)\s(\d+)\s(\d+\:\d+\:\d+)\s(\d+)]\s(\S+)\s(\S+)\s(\S+)\s(\S+)",
                df_args={
                    "columns": [
                        "day",
                        "month",
                        "date",
                        "time",
                        "year",
                        "type",
                        "method",
                        "status",
                        "file",
                    ]
                },
                n_lines_per_file=10,
            )
        )

        assert len([row for row in ds_local]) == 10

    def test_read_line_with_fn(self):

        ds_local = (
            textfactory.format("txt")
            .engine("pandas")
            .read_line(
                get_test_dataset_path("logs/*.log"),
                udf=lambda line: line.split(),
                n_lines_per_file=10,
            )
        )

        assert ds_local.shape == (10, 14)

        ds_local = (
            textfactory.format("txt")
            .backend("tika")
            .read_line(
                get_test_dataset_path("logs/*.log"),
                udf=lambda line: line.split(),
                n_lines_per_file=10,
            )
        )

        assert len([row for row in ds_local]) == 10

    def test_read_text(self):
        ds_local = (
            textfactory.format("txt")
            .engine("pandas")
            .read_text(
                get_test_dataset_path("reviews/**/*.txt"),
                total_files=10,
            )
        )

        assert len(ds_local) == 10

        ds_local = (
            textfactory.format("txt")
            .backend("tika")
            .read_text(
                get_test_dataset_path("reviews/**/*.txt"),
                total_files=10,
            )
        )

        assert len([row for row in ds_local]) == 10
