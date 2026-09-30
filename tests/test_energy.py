import numpy as np
import pytest

from endurosense.data.energy import energy_wh, step_distance_m


def test_constant_power_for_one_hour_gives_known_wh():
    t = np.arange(0, 3600.5, 0.5)            # 1 h at 2 Hz
    v = np.full_like(t, 24.0)
    i = np.full_like(t, 20.0)
    assert energy_wh(t, v, i) == pytest.approx(24.0 * 20.0)  # 480 Wh


def test_uneven_sampling_weights_by_interval():
    t = np.array([0.0, 1.0, 3.0])            # intervals of 1 s and 2 s
    v = np.array([10.0, 10.0, 10.0])
    i = np.array([0.0, 36.0, 72.0])          # 360 W for 1 s, 720 W for 2 s
    assert energy_wh(t, v, i) == pytest.approx((360 * 1 + 720 * 2) / 3600)


def test_single_reading_has_no_energy():
    assert energy_wh([0.0], [24.0], [20.0]) == 0.0


def test_step_distance_one_millidegree_of_latitude():
    d = step_distance_m(np.array([-79.78, -79.78]), np.array([40.458, 40.459]))
    assert d[0] == 0.0
    assert d[1] == pytest.approx(111.2, abs=0.2)
