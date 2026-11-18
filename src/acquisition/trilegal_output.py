"""Validate and normalize a completed TRILEGAL population."""

import math
from typing import List, Tuple

from trex.populations.background import (
    TRILEGAL_FOOTER_MARKER, TRILEGAL_LOGG_COLUMN, TRILEGAL_LOGTE_COLUMN,
    TRILEGAL_MASS_COLUMN, TRILEGAL_METALLICITY_COLUMN, TRILEGAL_TESS_COLUMN,
)
from .trilegal_errors import TrilegalUnavailableError

TRILEGAL_NORMAL_TERMINATION = "#TRILEGAL normally terminated"
# Exactly the columns ``populations.background.load_trilegal_population`` requires
# as finite numbers; other columns are retained as text and checked there.
_REQUIRED_COLUMNS = (TRILEGAL_TESS_COLUMN, TRILEGAL_MASS_COLUMN, TRILEGAL_LOGG_COLUMN, TRILEGAL_LOGTE_COLUMN,
                     TRILEGAL_METALLICITY_COLUMN)


def trilegal_output_to_csv(text: str) -> Tuple[str, int]:
    """Convert TRILEGAL's whitespace table to the CSV cache ``load_trilegal_population`` reads.

    Header and cell text are kept verbatim; only the delimiter changes and the
    single normal-termination footer is removed. TRILEGAL cells never contain
    whitespace; a cell containing a comma is refused because it would shift
    the CSV columns. Returns the CSV text and the number of simulated stars.

    Raises:
        TrilegalUnavailableError: Empty output, a missing required column, an
            unexpected comment row, a ragged row, a non-numeric or non-finite
            required cell, an abnormal, missing or repeated footer, data after
            the footer, or no simulated star.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise TrilegalUnavailableError("TRILEGAL output is empty")
    header = lines[0].split()
    columns = {name: index for index, name in enumerate(header)}
    missing = [name for name in _REQUIRED_COLUMNS if name not in columns]
    if missing:
        raise TrilegalUnavailableError("TRILEGAL output lacks required column(s): {0}".format(", ".join(missing)))
    required = tuple((name, columns[name]) for name in _REQUIRED_COLUMNS)
    rows: List[str] = [",".join(header)]
    terminated = False
    for line in lines[1:]:
        if line.startswith(TRILEGAL_FOOTER_MARKER):
            if not line.startswith(TRILEGAL_NORMAL_TERMINATION):
                raise TrilegalUnavailableError("TRILEGAL terminated abnormally: {0}".format(line))
            if terminated:
                raise TrilegalUnavailableError("TRILEGAL output repeats its termination footer")
            terminated = True
            continue
        if terminated:
            raise TrilegalUnavailableError("TRILEGAL output continues after its termination footer")
        if line.startswith("#"):
            raise TrilegalUnavailableError("TRILEGAL output has an unexpected comment row")
        cells = line.split()
        if len(cells) != len(header):
            raise TrilegalUnavailableError("TRILEGAL output has a row with a different column count")
        if any("," in cell for cell in cells):
            raise TrilegalUnavailableError("TRILEGAL output has a cell containing a comma")
        for name, index in required:
            try:
                value = float(cells[index])
            except ValueError as exc:
                raise TrilegalUnavailableError("TRILEGAL output has a non-numeric {0} cell".format(name)) from exc
            if not math.isfinite(value):
                raise TrilegalUnavailableError("TRILEGAL output has a non-finite {0} cell".format(name))
        rows.append(",".join(cells))
    if not terminated:
        raise TrilegalUnavailableError("TRILEGAL output lacks its normal termination footer")
    star_count = len(rows) - 1
    if star_count == 0:
        raise TrilegalUnavailableError("TRILEGAL output contains no simulated stars")
    return "\n".join(rows) + "\n", star_count
