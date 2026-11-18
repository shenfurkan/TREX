# Packaged mission response tables

Raw bytes are copied unchanged from the sources below; the readers in
`trex.acquisition.passbands` validate them on load. These are pre-launch
released/as-designed parameters, not in-flight calibrations.

## Roman/WFI (`roman/`)

- Files: `Roman_effarea_v8_SCA01..18_20240301.ecsv` (per-SCA effective area,
  m^2, including filter, optics and detector QE; "parameters as of 2024 03 01")
  and `Roman_zeropoints_20240301.ecsv` (per-SCA Vega/AB/ST magnitudes).
- Source: https://github.com/RomanSpaceTelescope/roman-technical-information,
  commit 14a9b0c98b0da723d0f2e4c6277956183c1fdd06, paths
  `roman_technical_information/data/WideFieldInstrument/Imaging/{EffectiveAreas,ZeroPoints}/`.
- Retrieved: 2026-11-15.
- License: CC0 1.0 (`roman/ROMAN-TECHNICAL-INFORMATION-LICENSE`).
- The ECSV header does not declare the `Wave` unit. Its range 0.400-2.599 and
  the documented WFI 0.48-2.3 um coverage identify micrometres; the packaged
  loader states that verification instead of guessing from a filename.

## PLATO (`plato/`)

- Files: `optical_transmission_ncam_v2.csv`, `optical_transmission_fcam_v2.csv`
  (DLR-TN-0113 v2 optical transmissions; columns cite PTO-EST-PL-REP-0936/0952).
- Source: https://github.com/IvS-KULeuven/PlatoSim3, commit
  4793088fc6fbb47cd134de1fbce2030da37d1421, path `docs/technical/DLR-TN-0113/`.
- Retrieved: 2026-11-15.
- License: PlatoSim3 terms (`plato/PLATOSIM3-LICENSE`, Apache-2.0 text,
  Copyright 2022 KU Leuven).
- Limitation: these curves cover optics/filters only. They contain **no detector
  quantum efficiency**, so they are not a complete PLATO photon response; a
  consumer needing absolute signal or detector-weighted colours must supply the
  QE edition explicitly. Official PLATO limb-darkening tables are not packaged.
