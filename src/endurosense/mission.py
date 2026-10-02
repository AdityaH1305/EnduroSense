"""Mission description and phase-based energy assembly (Model B's output).

Plain idea: a mission's energy is built from its parts, each predicted by a
component model:

    E(mission) = E(climb) + sum over legs [ P(leg) x T(leg) ]
               + E(hover between legs) + E(descent) + E(ground)

Leg time ``T(leg)`` is distance / commanded speed plus the time lost
accelerating and braking, which the data shows grows with speed (about 0.5 s
per leg at 4 m/s, 4.4 s at 12 m/s). Because missions are built leg by leg, any
route shape and length can be estimated, e.g. a delivery out-and-back is two
legs with the payload dropped after the first.

Component models implement the ``Components`` protocol. Trained models plug in
from Phase 4; ``ConstantComponents`` is a simple stand-in for tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class MissionSpec:
    """A planned mission. Every field is known before take-off."""

    legs_m: tuple[float, ...]          # horizontal length of each straight leg, in flying order
    payload_g: tuple[float, ...]       # payload carried on each leg
    speed: float                       # commanded cruise ground speed, m/s
    altitude_m: float                  # cruise height above take-off, m
    wind: float = 3.0                  # expected ambient wind speed, m/s

    def __post_init__(self):
        if len(self.legs_m) == 0 or len(self.legs_m) != len(self.payload_g):
            raise ValueError("need one payload value per leg, and at least one leg")
        if min(self.legs_m) <= 0 or self.speed <= 0 or self.altitude_m <= 0:
            raise ValueError("leg lengths, speed and altitude must be positive")

    @property
    def n_legs(self) -> int:
        return len(self.legs_m)

    @property
    def distance_m(self) -> float:
        return float(sum(self.legs_m))

    @classmethod
    def delivery(cls, distance_m: float, payload_g: float, speed: float, altitude_m: float,
                 wind: float = 3.0) -> MissionSpec:
        """Fly out with the payload, drop it, fly back empty."""
        return cls((distance_m, distance_m), (payload_g, 0.0), speed, altitude_m, wind)

    @classmethod
    def loop(cls, legs_m, payload_g: float, speed: float, altitude_m: float, wind: float = 3.0) -> MissionSpec:
        """Closed route carrying the same payload throughout (like the recorded routes)."""
        legs_m = tuple(float(x) for x in legs_m)
        return cls(legs_m, (float(payload_g),) * len(legs_m), speed, altitude_m, wind)


class Components(Protocol):
    """Per-part energy predictors used to assemble a mission."""

    def climb_wh(self, altitude_m: float, payload_g: float, wind: float) -> float: ...
    def leg_time_s(self, distance_m: float, speed: float, payload_g: float, wind: float) -> float: ...
    def leg_power_w(self, speed: float, payload_g: float, altitude_m: float, wind: float) -> float: ...
    def hover_wh(self, n_legs: int, payload_g: float, wind: float) -> float: ...
    def descent_wh(self, altitude_m: float, payload_g: float, wind: float) -> float: ...
    def ground_wh(self) -> float: ...


def mission_energy(spec: MissionSpec, comps: Components) -> dict:
    """Energy breakdown (Wh) and total for a mission."""
    legs = []
    for d, pay in zip(spec.legs_m, spec.payload_g):
        t = comps.leg_time_s(d, spec.speed, pay, spec.wind)
        p = comps.leg_power_w(spec.speed, pay, spec.altitude_m, spec.wind)
        legs.append(p * t / 3600.0)
    mean_payload = sum(spec.payload_g) / spec.n_legs
    parts = {
        "climb_wh": comps.climb_wh(spec.altitude_m, spec.payload_g[0], spec.wind),
        "legs_wh": sum(legs),
        "hover_wh": comps.hover_wh(spec.n_legs, mean_payload, spec.wind),
        "descent_wh": comps.descent_wh(spec.altitude_m, spec.payload_g[-1], spec.wind),
        "ground_wh": comps.ground_wh(),
    }
    parts["total_wh"] = sum(parts.values())
    parts["leg_wh"] = legs
    return parts


@dataclass
class ConstantComponents:
    """Fixed per-unit costs, for tests and as the simplest possible baseline."""

    climb_wh_per_m: float = 0.06
    descent_wh_per_m: float = 0.085
    power_w: float = 510.0
    overhead_s: float = 2.0
    hover_wh_per_leg: float = 0.4
    ground: float = 1.2

    def climb_wh(self, altitude_m, payload_g, wind):
        return self.climb_wh_per_m * altitude_m

    def leg_time_s(self, distance_m, speed, payload_g, wind):
        return distance_m / speed + self.overhead_s

    def leg_power_w(self, speed, payload_g, altitude_m, wind):
        return self.power_w

    def hover_wh(self, n_legs, payload_g, wind):
        return self.hover_wh_per_leg * n_legs

    def descent_wh(self, altitude_m, payload_g, wind):
        return self.descent_wh_per_m * altitude_m

    def ground_wh(self):
        return self.ground
