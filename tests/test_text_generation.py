import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from tests import test_data


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


def test_text_generation(client):
    response = client.post(
        "/text-generation",
        json={
            'text': test_data.PREAMBLE_1,
        }
    )
    assert response.status_code == 200
    assert response.json() == {'generated_text': test_data.GENERATED_TEXT_1}
