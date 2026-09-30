import pytest

from endurosense.config import data_path
from endurosense.data.load import load_flights, load_parameters
from endurosense.data.profile import build_flight_summary

RAW_AVAILABLE = (data_path("raw") / "flights.csv").exists()


def pytest_collection_modifyitems(config, items):
    if RAW_AVAILABLE:
        return
    skip = pytest.mark.skip(reason="raw CMU dataset not found in data/raw")
    for item in items:
        if "data" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def flight_summary():
    """Per-flight summary built from the raw data (built once per test session)."""
    return build_flight_summary(load_flights(), load_parameters())
