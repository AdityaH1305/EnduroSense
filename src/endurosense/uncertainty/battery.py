"""Calibrated predictive distribution for Model A (energy available).

The GRU ensemble gives a mean ``mu`` and spread ``sigma`` per reading. The
standardised error ``(truth - mu) / sigma`` on held-out batteries tells us how
many "sigmas" the truth really falls away from the mean at each probability
level. Those calibrated multipliers ``q_hat`` replace the textbook normal
values (e.g. 1.645 for the 95th percentile):

    quantile_q(x) = mu(x) + sigma(x) * q_hat[q]

``scale="constant"`` ignores sigma (same-width ranges everywhere); it is kept
as a comparison to show what the per-reading spread adds.
"""
from __future__ import annotations

import numpy as np

from endurosense.uncertainty.quantiles import PredictiveDistribution, conformal_quantiles

SCALERS = ("med", "s_mu", "s_sd", "q_mu", "q_sd", "y_mu", "y_sd")


def supported(df) -> np.ndarray:
    """Readings the model is allowed to judge: the battery's rest voltage was measured
    before take-off (for this flight and when the battery was first used).

    A real drone always has this reading. In the dataset it is missing only where a
    recording started with the motors already running (3 chains, incl. the hover
    tests); there the starting charge is unknown, errors reach 10-20 Wh, and three
    chains are too few to calibrate a range. The model abstains instead of guessing.
    """
    return (df["v_rest_chain_start"].notna() & df["v_rest_flight_start"].notna()).to_numpy()


def scores(y, mu, sigma, scale: str = "sigma") -> np.ndarray:
    """Calibration scores: standardised errors (or raw errors for ``scale="constant"``)."""
    err = np.asarray(y, float) - np.asarray(mu, float)
    return err / np.asarray(sigma, float) if scale == "sigma" else err


def calibrate(y, mu, sigma, groups, levels, scale: str = "sigma", correction: str | None = None) -> np.ndarray:
    """Calibrated multipliers ``q_hat`` (one per level) from held-out predictions."""
    return conformal_quantiles(scores(y, mu, sigma, scale), groups, levels, correction)


def quantile_values(mu, sigma, q_hat, scale: str = "sigma") -> np.ndarray:
    """(rows, levels) array of predictive quantiles."""
    mu, sigma, q_hat = np.asarray(mu, float)[:, None], np.asarray(sigma, float)[:, None], np.asarray(q_hat, float)[None, :]
    return mu + (sigma if scale == "sigma" else 1.0) * q_hat


def leave_fold_out_quantiles(y, mu, sigma, groups, folds, levels, scale: str = "sigma",
                             correction: str | None = None) -> np.ndarray:
    """Held-out predictive quantiles for every row: each fold's rows are calibrated
    using only the *other* folds' held-out errors (``mu``/``sigma`` must be out-of-fold)."""
    y, mu, sigma, groups, folds = (np.asarray(v) for v in (y, mu, sigma, groups, folds))
    out = np.zeros((len(y), len(levels)))
    for k in np.unique(folds):
        te = folds == k
        q_hat = calibrate(y[~te], mu[~te], sigma[~te], groups[~te], levels, scale, correction)
        out[te] = quantile_values(mu[te], sigma[te], q_hat, scale)
    return out


class CalibratedBatteryModel:
    """Ensemble + calibrated multipliers: gives a distribution of remaining Wh per reading."""

    def __init__(self, ensemble, levels, q_hat):
        self.ensemble, self.levels, self.q_hat = ensemble, np.asarray(levels, float), np.asarray(q_hat, float)

    def quantiles(self, df) -> np.ndarray:
        """(rows, levels) predictive quantiles; rows outside the supported domain are NaN."""
        mu, sigma = self.ensemble.predict_dist(df)
        out = quantile_values(mu, sigma, self.q_hat)
        out[~supported(df)] = np.nan
        return out

    def distributions(self, df) -> list[PredictiveDistribution]:
        """One distribution per row; ``None`` where the model abstains (no pre-flight voltage)."""
        return [None if np.isnan(row).any() else PredictiveDistribution(self.levels, row) for row in self.quantiles(df)]


def save_calibrated(model: CalibratedBatteryModel, kind: str, config: dict, path) -> None:
    """Store the ensemble's weights, input scalers and calibration multipliers."""
    import torch

    members = model.ensemble.members
    torch.save({"kind": kind, "config": config, "features": members[0].features,
                "levels": model.levels.tolist(), "q_hat": model.q_hat.tolist(),
                "requires": "pre-flight rest voltage (see endurosense.uncertainty.battery.supported)",
                "members": [{"state": m.net.state_dict(), "seed": m.seed,
                             "scalers": {k: getattr(m, k) for k in SCALERS}} for m in members]}, path)


def load_calibrated(path, windows, device: str | None = None) -> CalibratedBatteryModel:
    """Rebuild the calibrated ensemble saved by ``save_calibrated`` (no retraining).

    ``windows`` is the ``WindowStore`` holding sequence inputs for the rows to be predicted.
    """
    import torch

    from endurosense.models.probabilistic import ProbabilisticSequenceModel, SequenceEnsemble, _ProbNet

    blob = torch.load(path, map_location="cpu", weights_only=False)
    ens = SequenceEnsemble(blob["kind"], windows, n=len(blob["members"]), features=blob["features"], **blob["config"])
    for m, saved in zip(ens.members, blob["members"]):
        assert isinstance(m, ProbabilisticSequenceModel)
        if device:
            m.device = device
        for k, v in saved["scalers"].items():
            setattr(m, k, v)
        m.net = _ProbNet(blob["kind"], windows.X.shape[2], len(blob["features"]), m.hidden, m.layers, m.dropout)
        m.net.load_state_dict(saved["state"])
        m.net.to(m.device).eval()
    return CalibratedBatteryModel(ens, blob["levels"], blob["q_hat"])
