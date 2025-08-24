
from fastapi.testclient import TestClient
from api.main import APP
def test_healthz():
  c = TestClient(APP)
  r = c.get("/healthz")
  assert r.status_code == 200
  assert r.json().get("ok") is True
