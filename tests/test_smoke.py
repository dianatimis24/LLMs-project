import importlib
import pytest

REQUIRED_PACKAGES = [
    "pydantic",
    "chromadb",
    "sentence_transformers",
    "langchain_text_splitters",
    "rank_bm25",
    "fastmcp",
    "openai",
]

@pytest.mark.parametrize("package_name", REQUIRED_PACKAGES)
def test_dependencies_installed(package_name: str):
    """Verify that core project dependencies can be imported."""
    module = importlib.import_module(package_name)
    assert module is not None
