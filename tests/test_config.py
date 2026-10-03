import numpy as np

from endurosense.config import ROOT, data_path, load_config, set_seed


def test_config_has_decision_defaults():
    cfg = load_config()
    assert cfg["battery"]["reserve_v"] == 22.6
    assert cfg["decision"]["tau"] == 0.95
    assert cfg["uncertainty"]["quantiles"] == sorted(cfg["uncertainty"]["quantiles"])
    assert len(cfg["uncertainty"]["quantiles"]) == 23 and 0.5 in cfg["uncertainty"]["quantiles"]


def test_data_path_is_under_project_root():
    assert data_path("raw") == ROOT / "data" / "raw"


def test_set_seed_is_reproducible():
    set_seed(123)
    a = np.random.rand(3)
    set_seed(123)
    assert np.array_equal(a, np.random.rand(3))
