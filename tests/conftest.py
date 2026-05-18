import pytest

from egd_airbnb.spark_session import get_spark


@pytest.fixture(scope="session")
def spark():
    s = get_spark("tests", master="local[2]")
    yield s
    s.stop()
