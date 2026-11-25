"""Mirror field map.

A magnetic mirror sampled on a coarse grid, written to netCDF and read back, against the
analytic field: the multilinear interpolation error falls with the square of the spacing.
"""

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from takeout import FieldMap

LENGTH, B0, RATIO = 0.1, 0.05, 3.0


def mirror(points: np.ndarray) -> np.ndarray:
    """Return the mirror field ``(N, 3)`` at positions ``(N, ndim)``."""
    s = 2.0 * points[:, 0] / LENGTH - 1.0
    bx = B0 * (1.0 + (RATIO - 1.0) * s**2)
    return np.stack([bx, np.zeros_like(bx), np.zeros_like(bx)], axis=1)


x = np.linspace(0.0, LENGTH, 501)[:, None]
fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(6, 5))
top.plot(x[:, 0], mirror(x)[:, 0], "k", label="analytic")
with tempfile.TemporaryDirectory() as folder:
    for nodes in (6, 11, 21, 41):
        written = FieldMap.from_function(
            mirror, {"x": np.linspace(0.0, LENGTH, nodes)}, source="mirror"
        ).write(Path(folder) / f"mirror_{nodes}.nc")
        field_map = FieldMap.read(written)
        values = field_map.magnetic_field(x)[:, 0]
        top.plot(x[:, 0], values, "--", label=f"{nodes} nodes")
        bottom.semilogy(
            x[:, 0], np.abs(values - mirror(x)[:, 0]) / B0, label=f"{nodes}"
        )
top.set_ylabel("$B_x$ (T)")
top.legend()
bottom.set_xlabel("$x$ (m)")
bottom.set_ylabel(r"$|B_x - B_\mathrm{exact}| / B_0$")
fig.tight_layout()
plt.show()
