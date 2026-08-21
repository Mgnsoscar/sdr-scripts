"""calkit — the transmit script's power-calibration consumer.

The sdr-agent resolves each unit's measured calibration and injects a flat artifact
at ``$SDR_CALIBRATION_FILE`` (see the agent's docs/calibration.md). This module turns
that artifact into a :class:`PowerMap`: the same ``--power`` (dBm) ↔ commanded-gain
mapping the scripts already use, but backed by the unit's MEASURED curve instead of a
single baked anchor.

When no artifact is present — the unit/signal isn't calibrated, or you're running
off-unit — :meth:`PowerMap.load` returns a map built from the script's baked
constants, which is byte-identical to the previous behaviour (a slope-1 line through
the single anchor). So adopting calkit is a no-op until a unit is actually
calibrated; once it is, ``--power`` becomes accurate (interpolated) and reads at the
unit's real operating plane (e.g. EIRP at the antenna).

The artifact schema (produced by ResolvedCalibration.to_public_dict):
    { "curve": [[gain_db, power_dbm], …],   # gain-sorted, strictly monotonic
      "min_gain_db", "max_gain_db",          # commanded-gain clamps (the ceiling)
      "amplitude",                           # baseband amplitude the curve needs
      "operating_plane", "quantity", … }     # for the banner label
"""
from __future__ import annotations

import json
import os

CALIBRATION_FILE_ENV = "SDR_CALIBRATION_FILE"


def _interp(x: float, xs: list[float], ys: list[float]) -> float:
    """Piecewise-linear y(x) over strictly-increasing xs, endpoint-clamped. A single
    sample degrades to a slope-1 line through it (1 dB gain ≈ 1 dB power) — the same
    single-point fallback the agent resolver uses."""
    n = len(xs)
    if n == 1:
        return ys[0] + (x - xs[0])
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(1, n):
        if x <= xs[i]:
            x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return ys[-1]


class PowerMap:
    """Maps a requested delivered/radiated power (dBm) to a commanded SDR gain (dB)
    and back, over a monotonic measured curve, clamped to the unit's gain limits."""

    def __init__(self, gains, powers, min_gain_db, max_gain_db, amplitude,
                 source, label):
        if not gains or len(gains) != len(powers):
            raise ValueError("calibration curve is empty or malformed")
        # keep gain-sorted; agent guarantees strict monotonicity, verify defensively
        pairs = sorted(zip((float(g) for g in gains), (float(p) for p in powers)))
        self._gains = [g for g, _ in pairs]
        self._powers = [p for _, p in pairs]
        for i in range(1, len(pairs)):
            if self._gains[i] <= self._gains[i - 1] or self._powers[i] <= self._powers[i - 1]:
                raise ValueError("calibration curve is not strictly monotonic "
                                 "(not invertible)")
        self.min_gain_db = float(min_gain_db)
        self.max_gain_db = float(max_gain_db)
        self.amplitude = float(amplitude)
        self.source = source                 # human tag: where the map came from
        self.label = label                   # what --power means (quantity, at plane)

    # ── the two functions the script calls ──────────────────────────────────────
    def gain_for_power(self, delivered_dbm: float) -> float:
        """Commanded gain (dB) for a requested power, clamped to [min, max]. Upward
        is clamped to the ceiling, never extrapolated past it."""
        g = _interp(float(delivered_dbm), self._powers, self._gains)  # invert
        return min(max(g, self.min_gain_db), self.max_gain_db)

    def power_for_gain(self, gain_db: float) -> float:
        """Delivered power (dBm) at the operating plane for an (actual) gain."""
        g = min(max(float(gain_db), self.min_gain_db), self.max_gain_db)
        return _interp(g, self._gains, self._powers)

    @property
    def max_power_dbm(self) -> float:
        return self.power_for_gain(self.max_gain_db)

    @property
    def min_power_dbm(self) -> float:
        return self.power_for_gain(self.min_gain_db)

    # ── constructors ────────────────────────────────────────────────────────────
    @classmethod
    def from_linear(cls, min_gain_db, max_gain_db, min_power_dbm, max_power_dbm,
                    amplitude, label="SDR port (uncalibrated)") -> "PowerMap":
        """Baked fallback: a straight line between (min_gain, min_power) and
        (max_gain, max_power). With the scripts' constants this is exactly the old
        slope-1 anchor model, so behaviour is unchanged when uncalibrated."""
        return cls([min_gain_db, max_gain_db], [min_power_dbm, max_power_dbm],
                   min_gain_db, max_gain_db, amplitude,
                   source="baked defaults", label=label)

    @classmethod
    def from_artifact(cls, art: dict, fallback_amplitude: float) -> "PowerMap":
        """Build from the agent's resolved artifact dict."""
        curve = art.get("curve") or []
        gains = [pt[0] for pt in curve]
        powers = [pt[1] for pt in curve]
        amp = art.get("amplitude")
        amp = fallback_amplitude if amp is None else amp
        plane = art.get("operating_plane", "")
        quantity = art.get("quantity") or "power"
        label = f"{quantity}, at {plane}" if plane else quantity
        return cls(gains, powers,
                   art.get("min_gain_db"), art.get("max_gain_db"), amp,
                   source="calibration file", label=label)

    @classmethod
    def load(cls, baked: "PowerMap", env_var: str = CALIBRATION_FILE_ENV) -> "PowerMap":
        """Return the injected calibration map if ``$SDR_CALIBRATION_FILE`` is set,
        else ``baked``. A path that is set but unreadable/malformed raises — the agent
        only ever writes a valid artifact, so a broken one is a real error, not a
        reason to silently fall back to a different power scale."""
        path = os.environ.get(env_var)
        if not path:
            return baked
        try:
            with open(path, encoding="utf-8") as fh:
                art = json.load(fh)
        except (OSError, ValueError) as exc:
            raise ValueError(f"{env_var}={path} could not be read: {exc}") from exc
        return cls.from_artifact(art, fallback_amplitude=baked.amplitude)

    def describe(self) -> str:
        """One-line banner summary, e.g. 'calibration file — EIRP, at antenna_eirp'."""
        return f"{self.source} — {self.label}"
