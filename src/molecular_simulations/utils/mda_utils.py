from pathlib import Path

import MDAnalysis as mda
import numpy as np
from rust_simulation_tools import kabsch_align  # ty: ignore[unresolved-import]


def trim_trajectory(
    u: mda.Universe,
    out: Path,
    stride: int = 1,
    align: bool = False,
    rewrap: bool = False,
    sel: str | None = None,
    align_sel: str | None = None,
) -> None:
    selection = u.select_atoms(sel) if sel is not None else u.atoms
    assert selection is not None

    positions = np.zeros(
        (u.trajectory.n_frames // stride, selection.n_atoms, 3), dtype=np.float32
    )

    for i, _ in enumerate(u.trajectory[::stride]):
        positions[i, ...] = selection.positions.copy().astype(np.float32)

    if align:
        if align_sel is None:
            align_idx = selection.select_atoms('backbone or nucleicbackbone').ix
        else:
            align_idx = selection.select_atoms(align_sel).ix

        positions = kabsch_align(positions, positions[0], align_idx)

    if rewrap:
        pass

    with mda.Writer(str(out), n_atoms=selection.n_atoms) as w:
        for pos in positions:
            w.write(pos)
