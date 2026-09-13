from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class MeasurementFrame:
    """One complete firmware measurement frame plus optional quality diagnostics."""

    omega: np.ndarray
    magnitude: np.ndarray
    phase: np.ndarray
    diagnostics: dict[str, np.ndarray] = field(default_factory=dict)
    source: str = "serial"
    raw_lines: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "omega", np.asarray(self.omega, dtype=float).flatten())
        object.__setattr__(self, "magnitude", np.asarray(self.magnitude, dtype=float).flatten())
        object.__setattr__(self, "phase", np.asarray(self.phase, dtype=float).flatten())
        object.__setattr__(
            self,
            "diagnostics",
            {name: np.asarray(values, dtype=float).flatten() for name, values in self.diagnostics.items()},
        )

    def validate(self) -> None:
        lengths = {len(self.omega), len(self.magnitude), len(self.phase)}
        if len(self.omega) == 0:
            raise ValueError("测量帧为空：omega 没有数据点。")
        if len(lengths) != 1:
            raise ValueError(
                "测量帧长度不一致："
                f"omega={len(self.omega)}, Magnitude_data={len(self.magnitude)}, "
                f"Phase_data_rad={len(self.phase)}"
            )

    def as_legacy_tuple(self):
        """Return the old tuple shape used by the Tkinter app."""

        return self.omega, self.magnitude, self.phase, self.diagnostics
