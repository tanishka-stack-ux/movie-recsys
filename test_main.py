"""
Day 4: Automated tests for the recommendation API.
Run with: pytest test_main.py -v
"""

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_recommend_valid_user_returns_10_items():
    response = client.get("/recommend/196")
    assert response.status_code == 200
    data = response.json()
    assert data["cold_start"] is False
    assert data["count"] == 10
    assert len(data["recommendations"]) == 10


def test_recommend_unknown_user_falls_back_to_popular():
    response = client.get("/recommend/999999")
    assert response.status_code == 200
    data = response.json()
    assert data["cold_start"] is True
    assert len(data["recommendations"]) == 10


def test_recommend_invalid_n_rejected():
    response = client.get("/recommend/196?n=0")
    assert response.status_code == 400

    response = client.get("/recommend/196?n=100")
    assert response.status_code == 400


def test_recommend_custom_n():
    response = client.get("/recommend/196?n=5")
    assert response.status_code == 200
    assert response.json()["count"] == 5


def test_caching_marks_second_call_as_cached():
    response1 = client.get("/recommend/42?n=7")
    assert response1.json()["cached"] is False

    response2 = client.get("/recommend/42?n=7")
    assert response2.json()["cached"] is True

    assert response1.json()["recommendations"] == response2.json()["recommendations"]