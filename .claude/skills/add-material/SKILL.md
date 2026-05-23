---
name: add-material
description: Add a new composite material preset (e.g. AS4/3501-6, IM7/8552, T700/2510, T800/epoxy) to the BVID-FE material library. Use when the user wants to add a material, register a new ply, support a new prepreg, or extend MATERIAL_LIBRARY.
---

# Add a new material to MATERIAL_LIBRARY

Materials live in `src/bvidfe/core/material.py`. The library is a dict
named `MATERIAL_LIBRARY` (NOT `MATERIAL_PRESETS` — older docs use that
name, but the actual symbol is `MATERIAL_LIBRARY`; `_resolve_material`
in `analysis/bvid.py` looks up by string key).

## Pre-flight

Before adding the entry, the user must provide a **literature source**
for every constant. BVID-FE refuses to ship a material without one. If
they don't have a source, stop and ask — vendor datasheets, published
papers, or NASA/CMH-17 handbooks all qualify.

Confirm with the user:

1. Material name (the dict key, e.g. `"AS4/3501-6"`)
2. The source(s) — citation goes in the dataclass `name` field's
   surrounding comment or a docstring above the entry.

## Required constants and units

All constants are required and must be `> 0`. Units:

| Field | Unit | Notes |
|---|---|---|
| `E11`, `E22`, `G12`, `G13`, `G23` | MPa | Moduli |
| `nu12` | dimensionless | Must satisfy `-1.0 < nu12 < 0.5` |
| `Xt`, `Xc`, `Yt`, `Yc`, `S12`, `S23` | MPa | In-plane strengths |
| `G_Ic`, `G_IIc` | N/mm | Interlaminar fracture toughness |
| `rho` | **kg/mm³** | e.g. `1.58e-6` for CFRP. NOT g/cm³ — `__post_init__` rejects values outside `[1e-7, 1e-5]` because the old wrong docstring caused silent 1000x errors |

Optional (leave as `None` for transverse-isotropic fallback):

- `Zt`, `Zc` — through-thickness tensile/compressive strengths (fall
  back to `Yt`/`Yc`)
- `S13` — 1-3 plane shear (falls back to `S12`)

Optional impact-mapping calibration fields (`olsson_alpha`, `dent_beta`,
`dent_gamma`, `fiber_break_eta`, `fiber_break_E_threshold`, `soutis_k_s`,
`soutis_m`, `wn_d0_mm`, `puck_p_nt_plus`, `puck_p_nt_minus`) keep their
defaults unless the user has material-specific calibration data.

## Steps

### 1. Add the entry to `MATERIAL_LIBRARY`

Append the new entry to the dict in `src/bvidfe/core/material.py`,
matching the formatting of the existing entries (one field per line,
alphabetical-ish but really just match the existing pattern). The
`name` field on the dataclass must equal the dict key.

### 2. Add tests in `tests/core/test_material.py`

At minimum, assert that:

- The new key resolves: `MATERIAL_LIBRARY["NewName"]` returns an
  `OrthotropicMaterial`.
- `__post_init__` validation does not raise on the new entry.
- A sanity assertion on `get_compliance_matrix()` / `get_stiffness_matrix()`
  shape and the round-trip identity (`S @ C ≈ I`).

Mirror the structure of the existing tests in the file — don't invent
a new pattern.

### 3. Validate and format

```bash
ruff check src tests
black src tests
pytest tests/core/test_material.py -v
```

### 4. Mention downstream surfaces (do NOT edit unless the user asks)

The new key automatically flows through:

- `AnalysisConfig.material` (string union)
- The Streamlit material picker in `app.py`
- The JSON Schema for `AnalysisConfig` (regenerate with the
  `regenerate-schemas` skill)

If `AnalysisConfig` exposes a typed `Literal` of material names anywhere
(grep `MATERIAL_LIBRARY` to be sure), the Literal needs to be extended
too. As of this writing it does not — materials are looked up by string
at runtime.

## Things this skill must never do

- Never invent material constants. If the user can't cite a source,
  stop and ask.
- Never use g/cm³ for `rho` — kg/mm³ only (`1.58e-6`, not `1.58`).
- Never skip the test addition. CI runs `pytest` and an unguarded
  entry will silently break the next time someone tweaks the dataclass.
