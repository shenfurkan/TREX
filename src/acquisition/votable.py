"""Parse complete TAP responses without losing native catalogue identifiers."""

from io import BytesIO

import numpy as np
from astropy.io.votable import parse


def parse_votable_bytes(raw_bytes: bytes, *, require_query_status: bool = True) -> dict:
    """Return the single complete result table of a VOTable response.

    Raises ``ValueError`` for a non-``OK`` (or, when ``require_query_status``,
    missing) ``QUERY_STATUS``, any ``error``/``overflow``/``warning`` INFO, a
    VizieR ``matches`` count that differs from the returned rows, a response
    without exactly one table, and duplicated field names or COOSYS IDs.

    Returns a dict with ``data`` (rows as lists; masked cells become ``None``
    and NumPy scalars become Python values, so 64-bit IDs stay exact),
    ``table`` (an Astropy table), ``field_refs`` (field name -> ``ref``) and
    ``coordinate_systems`` (COOSYS ID -> system and epoch).
    """
    document = parse(BytesIO(raw_bytes), invalid="exception")
    infos = list(document.infos)
    coordinate_systems = list(document.coordinate_systems)
    resources = list(document.resources)
    # RESOURCE elements nest; status and coordinate INFO can sit at any level.
    while resources:
        resource = resources.pop()
        infos.extend(resource.infos)
        coordinate_systems.extend(resource.coordinate_systems)
        resources.extend(resource.resources)

    # IVOA DALI defines QUERY_STATUS values in upper case (OK, ERROR,
    # OVERFLOW); other INFO names are matched case-insensitively because
    # services differ in how they spell them.
    statuses = [info.value for info in infos if info.name == "QUERY_STATUS"]
    if (require_query_status and not statuses) or any(status != "OK" for status in statuses):
        detail = "; ".join(f"{info.value}: {info.content}" for info in infos
                           if info.name == "QUERY_STATUS" and info.content)
        raise ValueError(f"TAP query is not complete: QUERY_STATUS={statuses}" + (f" ({detail})" if detail else ""))
    problems = [f"{info.name}={info.value}" + (f" ({info.content})" if info.content else "")
                for info in infos if info.name.lower() in ("error", "overflow", "warning")]
    if problems:
        raise ValueError("archive returned an error, overflow or schema warning: " + "; ".join(problems))

    tables = list(document.iter_tables())
    if len(tables) != 1:
        raise ValueError(f"archive response must contain exactly one result table, got {len(tables)}")
    element = tables[0]
    rows = [
        # .item() turns NumPy scalars into Python ints/floats/str; int64 IDs stay exact.
        [None if np.ma.is_masked(value) else value.item() if isinstance(value, np.generic) else value
         for value in row]
        for row in element.array
    ]
    for info in infos:
        # VizieR reports its own match count; a smaller table means a truncated response.
        if info.name == "matches" and int(info.value) != len(rows):
            raise ValueError(f"VizieR reports {int(info.value)} matching rows but the response has {len(rows)}")

    names = [field.name for field in element.fields]
    if len(set(names)) != len(names):
        raise ValueError("archive response repeats a field name")
    systems = [system.ID for system in coordinate_systems if system.ID is not None]
    if len(set(systems)) != len(systems):
        raise ValueError("archive response repeats a COOSYS ID")
    return {
        "data": rows,
        "table": element.to_table(use_names_over_ids=True),
        "field_refs": {field.name: field.ref for field in element.fields},
        "coordinate_systems": {
            system.ID: {"system": system.system, "epoch": system.epoch}
            for system in coordinate_systems if system.ID is not None
        },
    }
