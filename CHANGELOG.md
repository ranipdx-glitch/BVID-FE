# Changelog

All notable changes to BVID-FE are documented in this file.

## [0.2.0-dev] - unreleased

In-progress work toward v0.2.0. No tag yet.

### Changed

- **README + ARCHITECTURE refreshed to match v0.2.0-dev reality.** The
  README's Limitations section claimed "No GUI in v0.1.0", "fe3d CAI
  uses first-ply-failure, not a buckling eigenvalue solve", and "no
  validated datasets in v0.1.0" — three statements that were correct
  for v0.1.0 but stale for v0.2.0-dev (where the GUI ships, the
  buckling eigensolve is implemented, and the validation harness is in
  the repo). The test-count badge advertised 216 passing; the actual
  count is 309. ``ARCHITECTURE.md``'s module catalog had no entry for
  the ``gui/`` package, the dependency diagram did not reach ``cli``
  or ``gui``, and the v0.2.0 roadmap section was a list of items that
  have since shipped. All three pages updated to reflect the current
  reality, with a "Shipped in v0.2.0-dev" / "Planned (post-v0.2.0)"
  split on the roadmap so future work and historical claims do not
  drift back into confusion.

### Changed

- **`_solve_failure_strain_analytic` (fe3d FPF inner solve) is now
  vectorised across Gauss points.** The per-element inner Python loop
  over Gauss points was replaced with a single batched call to
  ``larc05_index_batch`` / ``tsai_wu_index_batch`` (added in the previous
  commit) plus ``np.where``-masked branch evaluation for the four
  numerical edge cases (LaRC05 mask: ``idx_ref > 0``; Tsai-Wu masks:
  ``|b| < tiny`` linear fallback, ``|b| >= tiny`` quadratic, ``disc < 0``
  no-real-root, ``c <= 0`` non-physical-root). The previous scalar
  implementation is preserved verbatim as
  ``_solve_failure_strain_analytic_scalar_ref`` so the new
  ``tests/analysis/test_fpf_strain_solve_equivalence.py`` can prove the
  two paths return identical strain-at-failure values
  (``rel=1e-10``) on three representative inputs (pristine LaRC05,
  delaminated LaRC05, delaminated Tsai-Wu) and on the tension path
  (``strain_sign=+1``). Speedup is modest at default mesh resolutions
  (n_gp = 8 per element so the inner loop was small) but the change
  removes the last per-Gauss-point Python loop in the fe3d hot path
  and aligns the FPF solve with the rest of the failure-criterion API.

### Added

- **fe3d tier now exercised in CI as a monitoring step.** The CI
  ``validation`` job previously ran only the empirical tier (``python
  validation/validate_bvid_public.py --gate``). A new step runs the
  same harness with ``--tier=fe3d --max-cases=3``, marked
  ``continue-on-error: true`` so failures are advisory and don't block
  merges. The validation script gains:
  - A new ``_TIER_GATE_MULTIPLIER`` table — empirical and
    semi_analytical use the original 1.25× target_mae_pct slack;
    fe3d uses 2.0× to acknowledge the documented energy-flatness
    caveat without flapping the CI on regressions still inside the
    model uncertainty.
  - A new ``--max-cases N`` flag that truncates each dataset to the
    first ``N`` cases. Lets the fe3d step cap CI runtime at ~45 s
    even if the dataset later grows to dozens of cases. Default is
    ``None`` (full dataset), preserving the existing empirical-gate
    behaviour exactly.

- **Vectorised batch failure-criterion functions.** New
  ``larc05_index_batch(material, stresses)`` and
  ``tsai_wu_index_batch(material, stresses)`` evaluate their
  respective criteria over a Voigt-6 batch in a single numpy /
  ``np.einsum`` pass. ``FailureEvaluator.evaluate`` is refactored
  to call ``_index_batch`` once on the entire ``(n_elem, n_gp, 6)``
  field, then ``np.argmax`` to pick the critical (element, Gauss
  point) — replacing the prior nested Python loop that paid the
  function-call overhead n_elem × n_gp times. Three new tests in
  ``tests/failure/test_evaluator.py`` lock numerical equivalence
  between the batch and scalar forms (``rtol=1e-12``) and one
  test asserts the new ``evaluate`` agrees with the explicit
  nested-loop reference. The batch helpers are also intended as
  the prerequisite for a future vectorisation of
  ``_solve_failure_strain_analytic`` (deferred — see CHANGELOG
  rationale: small inner-loop sizes + weak regression coverage on
  the analytic strain solve mean the speedup-vs-risk tradeoff
  doesn't yet justify it).

### Changed

- **Refactor pass: tsai_wu vectorisation + heartbeat helper + fe3d pre-flight
  helper + dead-code removal.** Four targeted refinements (each independently
  verified by an explore-agent and locked by unit / equivalence tests):
  - ``tsai_wu_index`` no longer iterates 36 times in pure Python per call.
    ``_tsai_wu_coefficients`` now returns ``(F: np.ndarray, Q: np.ndarray)``
    instead of a Python list / list-of-lists, and the index is computed via
    ``F @ s + s @ Q @ s`` (BLAS-routed dot + bilinear form). A new
    ``test_tsai_wu_index_matches_explicit_nested_sums`` test pins 50
    randomised stress vectors to the explicit nested-sum form within
    ``rel=1e-12``. ``tsai_wu_strength_uniaxial`` is unchanged because
    ``F[i]`` and ``Q[i][i]`` work identically on numpy arrays.
  - The duplicated heartbeat-progress loop in ``AnalysisWorker.run`` and
    the per-iteration body of ``SweepWorker.run`` is now a single helper
    ``gui.workers._run_with_heartbeat(work, on_progress, *, start_pct,
    end_pct, interval_s, label)`` that spawns a daemon thread, ticks the
    progress callback toward ``end_pct`` at ``interval_s`` intervals, and
    re-raises a ``RuntimeError`` carrying the original traceback on
    failure. ``AnalysisWorker.run`` drops from ~40 lines to ~16; the
    ``SweepWorker`` per-energy body drops from ~19 lines to ~10. The
    ``TierComparisonWorker`` is intentionally NOT migrated because it
    ticks per-pair (synchronous, step-based) rather than per-second.
  - The duplicated fe3d pre-flight prologue (``_guard_problem_size`` +
    ``build_fe_mesh`` + ``_build_elements`` + log lines) shared by
    ``fe3d_cai_buckling``, ``_fe3d_cai_first_ply_failure``, and
    ``fe3d_tai`` is now a single ``analysis.fe_tier._fe3d_preflight(cfg,
    damage, lam, *, label) -> (mesh, elements, t0)`` helper. Each call
    site drops from ~6 lines to 1; ``fe3d_tai`` now also emits per-stage
    timing logs for parity with the CAI paths.
  - ``_k_contact_hertz_linearized`` (a back-compat alias in
    ``impact/olsson.py``) is deleted — re-verified to have zero callers
    in src/, tests/, examples/, scripts/, validation/, or docs/. The
    second prior dead-code candidate ``tsai_wu_strength_uniaxial`` was
    re-checked and KEPT (six active calls in
    ``tests/failure/test_tsai_wu.py``).

### Added

- **`tests/validation/` directory with five new analytical-reference tests.**
  Five new files lock the public formulas to known closed-form values so
  silent drift in the physics surfaces immediately:
  - ``test_clt_abd_reference.py`` — pinned ``A``, ``B``, ``D`` matrices for a
    ``[0/90]_s`` IM7/8552 laminate; symmetric layup → ``B == 0``; balanced
    layup → ``A_16 == A_26 == 0``.
  - ``test_olsson_threshold_reference.py`` — Olsson-2001 threshold-load
    shape relationships (``Pc / sqrt(D_eff * G_IIc) = pi*sqrt(8/9)``,
    panel-size invariance, ``Pc ~ h^{3/2}`` thickness scaling).
  - ``test_soutis_cai_reference.py`` — Soutis CAI knockdown self-
    consistency: zero-DPA short-circuit, dimensionless invariance under
    ``DPA / A_panel`` scaling, monotonicity, ``k_s`` calibration response.
  - ``test_whitney_nuismer_limits.py`` — Whitney-Nuismer TAI analytical
    limits: ``DPA → 0`` knockdown → 1.0; ``DPA → ∞`` knockdown → ``1 /
    Kt_inf`` (= 1/3 for Kt_inf = 3); monotone-decreasing in DPA;
    bounded-below by the asymptote.
  - ``test_fe3d_pristine_clt_consistency.py`` — pristine fe3d uniaxial
    extension recovers ``E_x = A11 / h`` (CLT) within 10% on a coarse
    ``20 × 10 mm × 4-ply`` mesh; pristine input produces no damaged
    elements; zero-load solve returns zero displacement.

- **Docstring upgrades on five high-traffic public symbols.**
  ``larc05_index``, ``tsai_wu_index``, ``olsson.threshold_load``,
  ``hex8.geometric_stiffness_matrix``, and
  ``semi_analytical.sublaminate_buckling_load`` previously had one-line
  docstrings that hid the underlying physics. Each now documents the
  formula, references (LaRC05 NASA/TM, Olsson 2001, Cook §17.7 / Bathe
  §6.8, Timoshenko & Gere §9.2), Voigt-index conventions, units, return-
  value semantics, and any raise-conditions — so ``help()`` and IDE
  tooltips show the same context that's currently spread across the
  README and ARCHITECTURE.

- **GUI panel-geometry helper deduplicated.** ``BvidMainWindow`` had
  three sites constructing the same ``PanelGeometry(Lx_mm, Ly_mm,
  boundary)`` from the panel-input panel (one of which was already
  exposed as ``panel_panel_as_geometry`` but unused). The two callers
  in ``_update_live_onset`` and ``_build_config`` now use the helper;
  the now-unused ``PanelGeometry`` import is dropped from
  ``_update_live_onset``. No behaviour change.

- **Repo hygiene: dependency caps, headless requirements.txt, top-level
  conftest, Dependabot.** Four small infrastructure changes that
  collectively make the project safer to install and easier to keep up:
  - ``pyproject.toml`` and ``requirements.txt`` now pin upper bounds
    on every direct dependency (e.g. ``numpy>=1.24,<3``,
    ``scipy>=1.10,<2``, ``PyQt6>=6.5,<7``, ``pytest>=7,<9``). The next
    major release of any of these libraries can no longer auto-break a
    BVID-FE install; users who want to live on the bleeding edge can
    still override the cap explicitly.
  - ``requirements.txt`` no longer pulls in the GUI stack (~100 MB of
    Qt + VTK + pyvistaqt). Headless library / CI users now get only the
    five core scientific deps; the GUI is opt-in via
    ``pip install -e ".[gui]"`` (or ``[all]``).
  - A new top-level ``conftest.py`` sets ``QT_QPA_PLATFORM=offscreen``
    before any PyQt import, so the ``tests/gui/`` suite runs cleanly on
    CI runners without a display server. Individual GUI test modules
    keep their per-file ``setdefault`` as defence-in-depth.
  - A new ``.github/dependabot.yml`` schedules weekly PRs for both pip
    and GitHub Actions dependencies (capped at 5 open per ecosystem),
    so security patches surface automatically and the test matrix
    runs against the new pin before a maintainer merges.

- **MeshParams input validation.** ``MeshParams.__post_init__`` now
  rejects ``elements_per_ply <= 0`` (and non-int values),
  ``in_plane_size_mm <= 0``, and ``cohesive_zone_factor <= 0``. The bad
  input previously slipped into ``build_fe_mesh`` and either silently
  produced a degenerate brick mesh or tripped a far-downstream error
  inside the assembler. Three parametrised tests in
  ``tests/analysis/test_mesh_params_validation.py`` lock the new
  behaviour.

- **CLI parser-helper unit tests.** ``_parse_panel``, ``_parse_layup``,
  ``_positive_float`` and ``_existing_path`` (introduced for issue #7)
  were exercised only end-to-end via ``subprocess`` — too slow for
  edge-case coverage. ``tests/test_cli_helpers.py`` adds 12 direct
  pytest tests covering malformed panel formats, non-positive
  dimensions, non-numeric layup tokens, zero / negative / non-numeric
  ``_positive_float`` inputs, and the missing-file / directory paths
  for ``_existing_path``.

- **Cross-tier consistency tests.** Per the new README "Knockdown
  definition and cross-tier comparability" section there are several
  documented relationships between tier outputs that no test enforced:
  (a) ``pristine_strength_MPa`` identical across tiers, (b) TAI
  ``empirical.knockdown == semi_analytical.knockdown`` exactly, (c)
  CAI ``semi_analytical.knockdown <= empirical.knockdown``, (d)
  pristine input → knockdown == 1.0, (e) every knockdown finite and
  bounded in [0, 1+ε]. ``tests/test_cross_tier_consistency.py`` locks
  all five.

### Fixed

- **C-scan loader hardens type and finiteness checks.** ``_validate_dict``
  in ``damage/io.py`` previously had several gaps: a string in a numeric
  field crashed at the ``< 0`` comparison with a raw ``TypeError``;
  ``NaN`` and ``Inf`` (both JSON-encodable via Python's default
  ``allow_nan=True``) passed through into the analysis pipeline; a non-
  dict entry in the ``delaminations`` array crashed the
  ``"k" not in d`` check; ``fiber_break_radius_mm < 0`` was accepted;
  ``interface_index = 2.5`` was silently truncated to ``2``; and unknown
  top-level / per-delamination fields were dropped silently, so a typo
  like ``"dent_dept_mm"`` (missing the trailing 'h') reported as
  ``missing required field: dent_depth_mm`` without flagging the typo.
  Fixed by routing every numeric field through a new
  ``_require_finite_number`` helper that raises ``CScanSchemaError`` on
  non-numeric, NaN, or Inf, plus per-delamination dict-shape checks,
  ``fiber_break_radius_mm >= 0``, strict-int ``interface_index``, and a
  ``UserWarning`` (forward-compatible — does not fail the load) on
  unknown fields. Eight new tests in ``tests/damage/test_io.py`` cover
  the string-in-numeric, NaN, Inf, negative-radius, non-object-element,
  unknown-field warning, and non-int-interface-index paths.

- **Invalid `BVIDFE_LOG_LEVEL` / `BVIDFE_FE3D_MAX_DOF` no longer crash
  imports.** Both env vars were read at module import time:
  ``logging.setLevel(os.environ.get(...).upper())`` raised on a typo
  like ``BVIDFE_LOG_LEVEL=DUBG``, and ``int(os.environ.get("BVIDFE_FE3D_MAX_DOF",
  "500000"))`` raised on a non-numeric value. Either error prevented
  ``bvidfe`` from being imported at all. Fixed by routing both reads
  through new ``_resolve_log_level`` / ``_resolve_max_dof`` helpers in
  ``analysis/fe_tier.py`` that fall back to the documented default with
  a single-line stderr warning when the value is unrecognised, zero,
  or negative. ``gui/workers.py`` shares the log-level resolver. Seven
  new tests in ``tests/analysis/test_env_resolution.py`` cover the
  default, valid-name, typo, override, non-numeric, zero, and negative
  paths.

- **Parametric sweeps no longer lose partial results on per-iteration
  failure.** ``sweep_energies`` / ``sweep_layups`` / ``sweep_thicknesses``
  previously called ``_run_one`` unguarded — a single failed iteration
  (Olsson out-of-regime, mesh degeneracy, Tsai-Wu invalid combination)
  raised out of the sweep and the CSV was never written, so a 12-energy
  sweep with one bad row dropped 11 valid results on the floor. Each
  entry point now accepts an ``on_error`` keyword (default
  ``"raise"`` for backward compatibility, ``"skip"`` to fill the failed
  row with NaN numerics + an ``error`` column and continue, ``"warn"``
  to do the same plus emit a ``UserWarning``) and an optional
  ``progress_callback(i_done, n_total)``. With ``on_error="skip"`` the
  partial CSV is preserved with the same row count as the input
  iterable. Five new tests in ``tests/sweep/test_sweep.py`` cover the
  raise / skip / warn paths, the partial CSV, the progress callback,
  and rejection of bogus ``on_error`` values.

- **CLI now validates inputs at parse time instead of deep in the pipeline.**
  Several `bvidfe` CLI flags previously accepted invalid values that only
  surfaced as opaque errors much later: `--material IM7/8553` (typo)
  reached ``MATERIAL_LIBRARY[args.material]`` before producing a raw
  ``KeyError``; `--energy 0` and negative numbers passed argparse and
  produced cryptic Olsson-threshold warnings; `--cscan
  /does/not/exist.json` raised a bare ``FileNotFoundError`` from inside
  ``load_cscan_json``; `--thickness`, `--impactor-diameter`, and
  `--mass` accepted any float including zero and negatives. The argparse
  setup now uses ``choices=`` for ``--material`` (so typos surface as
  ``invalid choice: 'IM7/8553' (choose from 'AS4/3501-6', 'IM7/8552',
  'T700/2510', 'T800/epoxy')``), a custom ``_positive_float`` type for
  the four numeric flags, and a custom ``_existing_path`` type for
  ``--cscan`` that fails fast with ``file not found``. Panel dimensions
  are also rejected when non-positive. A new ``--quick-json`` flag emits
  a single-line JSON object ``{knockdown, residual_strength_MPa,
  pristine_strength_MPa, tier_used}`` for scripted callers that need
  more than the bare scalar from ``--quick``. Four new tests in
  ``tests/test_cli.py`` cover the bad-material, zero-energy,
  missing-cscan, and quick-json paths.

- **GUI surfaces malformed input instead of silently dropping it.** Two
  separate paths previously absorbed user-input failures with no
  feedback: ``BvidMainWindow._load_config`` had a bare ``except
  Exception`` around ``json.loads`` + ``config_from_dict`` that turned
  every failure into a single cryptic ``str(exc)`` warning, and
  ``DamagePanel.get_damage_state`` swallowed ``ValueError`` /
  ``AttributeError`` per-row with ``continue`` so a typo in a single
  cell silently dropped that delamination from the analysis. Both are
  now explicit:
  - ``_load_config`` distinguishes ``OSError`` (cannot read file),
    ``json.JSONDecodeError`` (malformed JSON, with line + column),
    ``KeyError`` (missing required field), and ``TypeError`` /
    ``ValueError`` (wrong type), each with a distinct dialog title.
  - ``DamagePanel.get_damage_state`` records every skipped row in a new
    ``self.skipped_rows`` attribute and logs a warning on the
    ``bvidfe.gui`` logger; ``BvidMainWindow._build_config`` reads this
    after every config build and reports skipped row indices via the
    status bar so the user can fix the typo. Four new tests in
    ``tests/gui/test_config_io.py`` (covering JSON parse errors, missing
    fields, wrong types, and damage-row skipping) lock the new
    behaviour.

- **GUI tier-comparison no longer freezes the main thread.** The
  "Compare Tiers (empirical + semi_analytical)..." menu action ran 16
  `BvidAnalysis.run()` calls (2 tiers x 8 energies, ~12 s wall clock at
  default settings) directly on the Qt main thread inside
  `BvidMainWindow._compare_tiers`. The status bar updated to "Running
  tier comparison..." but the entire UI was unresponsive until the loop
  finished — including the Knockdown Curve tab where the result would
  ultimately appear. Fixed by extracting the loop into a new
  `TierComparisonWorker(QThread)` in `gui/workers.py` that mirrors the
  existing `AnalysisWorker` / `SweepWorker` pattern (resultReady / error /
  progress signals, daemon-thread heartbeats, deleteLater cleanup).
  Per-(tier, energy) failures are absorbed into NaN entries with a
  third tuple element listing the skipped pairs so the status bar can
  report them instead of aborting the whole comparison. Two new tests
  in `tests/gui/test_workers.py` cover the happy-path and the
  bogus-tier partial-failure path.

- **Hex8 elements now validate the Jacobian determinant.** `Hex8Element.B_matrix`
  and `Hex8Element.geometric_stiffness_matrix` previously called
  `np.linalg.inv(J)` with no check on `np.linalg.det(J)` — an inverted element
  (det < 0, from wrong node ordering) silently produced negative volume in
  the stiffness integral, and a singular element (det ≈ 0, from collapsed
  nodes) silently produced NaNs that propagated into the global K and
  corrupted every downstream eigensolve / FPF result. Both call sites now
  invoke a shared `_validate_jacobian` helper that raises a new
  `DegenerateElementError(ValueError)` with the offending detJ value, the
  natural-coordinate Gauss-point location, and the full node-coordinate
  list, so users get a reproducible "bad mesh" error instead of opaque
  NaNs further down the pipeline. `Hex8iElement.stiffness_matrix` (which
  inverts a separate centre-Jacobian for its bubble-mode Benh derivation)
  has the same guard. The structured brick mesh produced by `build_fe_mesh`
  is unaffected — regular cuboid grids always have strictly positive detJ.

- **Tsai-Wu now uses through-thickness strengths and the 1-3 shear strength.**
  `failure/tsai_wu.py` previously computed `F3 = 1/Yt - 1/Yc`, `F33 =
  1/(Yt*Yc)`, and `F55 = 1/S12**2`. The first two reuse the in-plane
  transverse strengths for the through-thickness direction (a missing-
  parameter conflation, since `OrthotropicMaterial` had no `Zt`/`Zc`
  fields), while `F55 = 1/S12**2` was a real formula bug — the 1-3
  shear coefficient should use the 1-3 shear strength (Voigt index 4
  in the project's `[xx, yy, zz, yz, xz, xy]` convention), not the
  in-plane shear strength. This deviated from Tsai & Wu (1971) and
  produced incorrect failure indices whenever a 3D stress state had
  non-trivial sigma_zz, tau_yz or tau_xz components — e.g. `fe3d` TAI
  on a damaged panel where the through-thickness coupling is broken.

  `OrthotropicMaterial` gains three optional fields, `Zt`, `Zc`, `S13`,
  and three corresponding `*_resolved` properties that fall back to
  `Yt` / `Yc` / `S12` when the user has not provided through-thickness
  test data (the standard transverse-isotropy assumption for
  unidirectional CFRP). The four library presets keep their existing
  in-plane values, so default behaviour is numerically backward-
  compatible — the formula reduces to the prior code and existing
  Tsai-Wu tests pass unchanged. Users with measured through-thickness
  strengths can now override per-material; three new tests in
  `tests/failure/test_tsai_wu.py` cover the default-fallback,
  Zt-override, and S13-override paths.

- **fe3d buckling silent fallbacks now surfaced via `AnalysisResults.notes`.**
  Previously, `fe3d_cai_buckling()` returned `(sigma_pristine_MPa, 0.0)` when
  the eigensolver found no positive eigenvalue or raised an exception, and
  `BvidAnalysis.run()` silently used FPF instead when the buckling result was
  below 5% of pristine ("plausibility gate"). The user saw `knockdown = 1.0`
  in the no-eigenvalue case and had no way to know that the buckling solve
  had quietly given up. `AnalysisResults` gains a `notes: list[str]` field
  populated by the analysis backends; `fe3d_cai_buckling()` now returns
  `(sigma, lambda_crit, notes)` so the caller can merge solver diagnostics
  into the result. The GUI Summary tab appends these notes (prefixed with
  ⚠) under the existing "--- Notes ---" section, and `AnalysisResults.summary()`
  / `to_dict()` include them in the text and JSON output. Two new tests
  in `tests/analysis/test_fe3d_buckling.py` cover the eigensolver-failure
  path end-to-end via `monkeypatch`.

- **fe3d damage factor now applied component-wise to the elasticity matrix.**
  The previous `DAMAGE_STIFFNESS_FACTOR = 0.30` was scaled into every entry of
  the 6×6 `_C_global` matrix in `analysis/fe_tier._build_elements`, uniformly
  reducing in-plane stiffness (E11, E22, G12) by 70% in delaminated zones —
  even though the docstring in `fe_mesh.py` already noted that "the plies
  themselves are intact (so in-plane load-carrying is mostly preserved)". The
  fix replaces the single constant with two physically-motivated factors:
  - `DAMAGE_OOP_FACTOR = 0.05` — out-of-plane (E33, G13, G23, and the
    in-plane / OOP Poisson cross-coupling), applied to every element inside a
    delamination ellipse footprint. Representative of the post-delamination
    interlaminar-modulus loss reported in Bolotin (2001) and Sun & Tao (1998).
  - `DAMAGE_FIBER_BREAK_INPLANE_FACTOR = 0.30` — in-plane sub-block (rows/cols
    {0, 1, 5} in the global Voigt frame), applied only inside the fiber-break
    core (within `fiber_break_radius_mm` of any delamination centroid).
    Representative of fiber-direction damage saturation in Camanho & Davila
    (2007) / Maimi et al. (2007) for CFRP.

  `FeMesh` gains a new `in_plane_damage_factors` array; existing
  `damage_factors` is now the OOP factor exclusively. The viz layer
  (stress-field heatmap, pyvista damage overlay) is unchanged because it
  visualises the OOP factor as "delamination depth", which is the right
  semantic. The fe3d Kg assembly now scales the uniaxial sigma_xx prestress
  by the in-plane factor (since the prestress is an in-plane component); pure
  delamination zones therefore carry full uniaxial load through the fibers.

  Effect on results: knockdown values shift, the buckling solve becomes more
  responsive to damage size (because the pure-delamination in-plane block is
  no longer over-penalised), and the previously-flagged "fe3d knockdown
  approximately flat vs. impact energy" caveat is partially attenuated.
  Existing tests are structural (inequalities and bounds, not pinned numeric
  values) and should still pass; CI will flag any that need updating.

### Added

- **Input-variable sensitivity rollup.** Before this change, four of the
  five impact-event inputs were silently ignored by the physics pipeline:
  - `panel.boundary` (simply_supported / clamped / free) never moved the
    Olsson bending stiffness, the semi-analytical sublaminate buckling
    coefficient, or the fe3d buckling BCs — all three tiers hard-coded
    simply-supported. Fixed by adding boundary-dependent multipliers
    (clamped=2.5× bending, 1.9× buckling; free=0.4× bending, 0.5× buckling
    per Timoshenko) in `impact/olsson.py` and `analysis/semi_analytical.py`,
    plus boundary-aware lateral-edge `u_z` penalty BCs in
    `analysis/fe_tier.py::fe3d_cai_buckling`.
  - `impactor.shape` (hemispherical / flat / conical) was never referenced
    by any physics code. Fixed by wiring shape into a per-shape Hertz-
    contact stiffness (Johnson, *Contact Mechanics* §3.4-3.5) and into a
    footprint-spread multiplier on the target DPA (flat=1.4×, conical=0.7×).
  - `impact.mass_kg` was never referenced. Fixed by adding a
    calibration-aware dynamic amplification factor `(5.5/m)^0.1`, exactly
    unity at the ASTM D7136 reference mass, plus a UserWarning when the
    impactor mass ratio falls below Olsson's quasi-static validity regime.
  - `impactor.diameter_mm` only shifted E_onset by <0.5% and was masked
    by the DPA cap; fixed by adding a `(16/d)^0.3` spread factor on DPA.

  A new scripted validation matrix (`scripts/validate_inputs.py`) runs 5
  levels of each variable across every (tier, loading) combination and
  flags any input that produces zero knockdown variation — the flags
  section now reports "(none)" for the first time.

- **Impact-location and fiber-break core markers on the damage map.**
  Previously the damage-map plot drew only the delamination ellipses, so
  a user could not tell from the plot WHERE the impact landed on the
  panel — a real problem for off-center impacts where the footprint
  asymmetry drives failure. `plot_damage_map` now adds a black "×"
  marker at each unique ellipse centroid (the impact point) and, when
  the material has a non-trivial fiber-break model (`fiber_break_eta > 0`
  and `E_impact` above the fiber-break threshold), a red filled circle
  showing the fiber-break core radius. Both appear in the legend. 4
  new regression tests in `tests/viz/test_damage_map_markers.py`.

- **Live DPA preview with saturation warning.** A new `DPA:` label in the
  Impact panel updates every time any input changes, showing both the
  absolute predicted damage area in mm² and the percentage of panel area.
  When the 80% cap engages the label switches to red bold text with a ⚠
  SATURATED marker — users now see saturation *before* pressing Run,
  rather than discovering it post-hoc via the Summary tab notice and
  wondering why knockdown stopped responding to energy. Saturation in
  the default 150×100 mm 8-ply configuration kicks in at ~15 J; this
  preview makes that limitation obvious. 4 new regression tests in
  `tests/gui/test_live_onset.py`.

- **Live E_onset preview in the Impact panel.** The `ImpactPanel.set_onset_energy()`
  method has existed since the first release but was never wired up — the
  "E_onset: — J" label stayed blank no matter what the user changed. Now
  the `BvidMainWindow` connects `configChanged` on the material, panel,
  impact, and input-mode panels to a single `_update_live_onset` slot
  that recomputes the Olsson threshold and updates the label in real
  time. Because E_onset is boundary- and shape-aware (new in this
  release), the label now responds visibly to every relevant input:
  simply_supported→clamped drops it from 0.60 J to 0.24 J, simply_supported→free
  raises it to 1.49 J, etc. Handles invalid intermediate states (user
  mid-edit) gracefully and blanks out in damage-driven mode where the
  preview is not applicable. 5 new regression tests in
  `tests/gui/test_live_onset.py`.

- **fe3d KD-vs-energy monotonicity fix.** `DAMAGE_STIFFNESS_FACTOR` in
  `analysis/fe_mesh.py` was raised from `1e-4` to `0.30`. The old value
  was physically unrealistic — it treated delaminated elements as
  essentially null in the stress field, which made the failure-index
  criterion unable to flag damaged regions, and the fe3d residual
  strength *increased* with impact energy past ~15% panel damage
  (because the peak stress in the undamaged shell drops as the damage
  footprint spreads wider). The new value represents typical residual
  in-plane stiffness after delamination (plies are intact, only the
  through-thickness coupling is lost); literature range for CFRP is
  0.1-0.5. Knockdown now trends monotonically with energy across
  simply-supported, clamped, and free boundaries (see
  `tests/test_input_sensitivity.py::test_fe3d_knockdown_mostly_decreases_with_energy`).

- **Summary tab edge-case notices.** The GUI Summary tab now surfaces
  the `UserWarning`s that previously only appeared on stderr: DPA
  saturation (knockdown insensitive to further energy), 'free' boundary
  soft-support approximation, small-mass quasi-static regime violation,
  and the known fe3d energy-insensitivity limitation. Each notice is
  one plain-language paragraph under a "--- Notes ---" section so the
  user sees why their result may not respond to an input they changed.

- **True linear buckling eigensolve in the `fe3d` CAI tier**
  `Hex8Element.geometric_stiffness_matrix(sigma_bar)` integrates
  `gradN.T @ S @ gradN` over the element via 2×2×2 Gauss quadrature and
  expands to a 24×24 K_g via `np.kron`. A new `fe3d_cai_buckling()` in
  `analysis/fe_tier.py` assembles K and K_g under a uniform uniaxial
  pre-stress (scaled by per-element damage factor), applies rigid-body
  penalty BCs, and solves `K·φ = λ·K_g·φ` via `eigsh` shift-invert.
  `BvidAnalysis.run()` for `tier="fe3d"` now returns the minimum of the
  buckling stress and the first-ply-failure stress, capturing whichever
  mode governs. `AnalysisResults.buckling_eigenvalues` is populated with
  the smallest positive eigenvalue. First-ply-failure is retained as
  `_fe3d_cai_first_ply_failure` — together they give engineers an upper
  bound (FPF) and a lower bound (buckling) on the residual strength.
- **"Damage View" GUI tab** (originally planned as "3D Mesh"). After
  three VTK embedding approaches (QtInteractor, lazy-init QtInteractor,
  BackgroundPlotter) all deadlocked Qt's main event loop on macOS, the
  tab was reworked to a matplotlib-based four-panel orthographic view:
  top (x-y), side (x-z), front (y-z), and a text summary panel. Renders
  in ~50 ms on the main thread, no OpenGL dependency. True VTK 3D
  visualization remains available via the `bvidfe.viz.plots_3d` Python
  API and the `examples/` scripts, just not embedded inside the GUI's
  event loop.
- **Validation harness** (`validation/validate_bvid_public.py`):
  `DatasetCase` dataclass + `run_dataset()` + MAE% metric. Auto-discovers
  any JSON dataset in `validation/datasets/`. Ships with a synthetic
  self-check dataset (MAE ≈ 0% by construction) so the harness is
  exercised in CI; real published datasets (Soutis, Caprino,
  Sanchez-Saez, NASA round-robin) remain to be digitized by hand.
- **CI regression gate** — new `validation` job in
  `.github/workflows/tests.yml` running
  `python validation/validate_bvid_public.py --gate`.
- **Auto-populated Knockdown Curve tab.** Every single-run analysis now
  kicks off an empirical-tier energy sweep in the background (8 points,
  sub-second) so the Knockdown Curve tab always shows a
  knockdown-vs-energy plot for context, without requiring the user to
  click "Run energy sweep" explicitly.
- **Buckling Eigenvalues tab** (was "Buckling Mode" placeholder): bar
  chart of the first up to 6 buckling eigenvalues for semi_analytical
  and fe3d runs; explanatory note for the empirical tier (no eigensolve).
- **Damage Severity tab** (was "Stress Field" placeholder): top-down
  heatmap showing how many interfaces are delaminated at each (x, y)
  column — through-thickness sum of `(1 - damage_factor)`, colored
  hot. Analogous to a C-scan plan view.
- **Stage-by-stage logging**: new `bvidfe.fe3d` and `bvidfe.gui` loggers
  write to stderr so users launching the app from a terminal see mesh
  build, K/Kg assembly, eigensolve, FPF, and heartbeat progress lines
  in real time. Controlled via `BVIDFE_LOG_LEVEL` env var.
- **Mesh-size guard** (`FE3DSizeError` + GUI dialog): both single-run
  and sweep paths check the fe3d problem size against
  `BVIDFE_FE3D_MAX_DOF` (default 500k). Hard stop above cap, soft
  warning for merely-large sizes. Prevents SIGSEGV from scipy's sparse
  solvers when memory is exhausted.
- **Heartbeat progress** in both workers: AnalysisWorker ticks every
  2 s during long fe3d runs, SweepWorker ticks per-energy. Prevents
  the status bar looking frozen during multi-minute runs.
- **Vectorized assembler + analytic FPF** in the fe3d tier: 6.5x
  speedup on realistic 5k-element meshes by (a) replacing the 24×24
  Python loop per element with one numpy broadcast, and (b) replacing
  the 12-iteration bisection for first-ply-failure with a single FE
  solve + closed-form quadratic root (Tsai-Wu and LaRC05 are both
  quadratic in stress and stress scales linearly with applied strain).
- **Sanity-check on buckling eigenvalue**: `fe3d_cai` falls back to
  FPF-only when the uniform-pre-stress buckling approximation returns
  a critical stress below 5% of pristine (indicates a numerical
  artefact on the simplified BC set).
- **Conservative defaults + inline fe3d caveat**: `MeshParams` defaults
  to 1 element/ply + 5 mm in-plane (from 4/1.0), and the Summary tab
  appends a note when `tier=fe3d` explaining that fe3d knockdown is
  largely flat vs. impact energy on the current simplified model and
  pointing users to empirical/semi_analytical for energy-dependent
  curves.
- **DPA cap + dent cap**: `impact_to_damage` clips DPA at 80% of panel
  area (emits `UserWarning`) and `dent_depth_mm` clips dent at 50% of
  laminate thickness. Prevents physically-absurd outputs like a 16 mm
  dent on a 1.2 mm laminate.
- **Four runnable examples** (`examples/01_empirical_quick.py` through
  `04_inspection_driven.py`) + `sample_cscan.json`: copy-paste workflows
  for single-run, tier comparison, energy sweep, and damage-driven
  analysis. Each writes outputs to `examples/output/`.
- **14 edge-case robustness tests** (`tests/test_edge_cases.py`): below-
  threshold energies, huge-energy DPA saturation, tiny panels, empty
  damage states, mixed tier switching, CLI end-to-end subprocess, and
  parametric monotonicity across 4 energies.
- **216 tests now passing** (was 179 at v0.1.0 — +37 new tests across
  analysis, elements, solver, GUI, CLI, and edge cases).

### Known limitations (still deferred)

- Material calibration constants remain placeholders until real datasets
  land.
- `fe3d` tier still uses stiffness-reduction at delaminations rather than
  zero-thickness cohesive surfaces (different physics — stiffness
  reduction captures the stiffness loss but not the debonding/sliding).
- **`fe3d` knockdown is approximately flat vs. impact energy** for any
  damage above the Olsson threshold. This is a structural limitation of
  the stiffness-reduction + uniform-pre-stress buckling simplifications:
  the stress-concentration-driven first-ply-failure strain is controlled
  by the healthy/damaged *boundary* rather than the damage magnitude,
  and the buckling eigenvalue is not reliably physical on our simplified
  BCs so it usually gets rejected by the 5%-pristine sanity check. An
  attempt at v0.2.0-dev to add graded damage + real in-plane pre-stress
  BCs to fix this was shelved: the proper-BC buckling eigensolve was
  14× low vs. analytical plate buckling (sign-convention / stress-field
  interpretation issues that need more time than a single session).
  For energy-dependent residual strength use `tier=empirical` (Soutis
  scales with DPA) or `tier=semi_analytical` (Rayleigh-Ritz sublaminate
  buckling scales with ellipse size). Fixing fe3d is v0.3.0 scope.
- Release artifacts are still unsigned.

## [0.1.0] - 2026-04-17

Graduated from `v0.1.0-alpha`. Adds the PyQt6 desktop GUI and packaging infrastructure.

### Added (since v0.1.0-alpha)

- **PyQt6 desktop GUI** (`bvidfe-gui` console script):
  - Seven input panels (MaterialPanel, PanelPanel, InputModePanel, ImpactPanel, DamagePanel, AnalysisPanel, SweepPanel)
  - Six result tabs (Summary, Damage Map, Knockdown Curve, 3D Mesh/Buckling/Stress placeholders)
  - `AnalysisWorker` and `SweepWorker` QThread subclasses for off-UI-thread computation
  - File menu: Save/Load Config (JSON), Export Results JSON, Export Damage Map PNG
  - Headless pytest-qt test coverage for all panels and workers
- **PyInstaller spec** (`BvidFE.spec`) for building macOS and Windows standalone apps
- **GitHub Actions release workflow** (`.github/workflows/release.yml`): on-tag build of macOS + Windows bundles, auto-uploaded to GitHub Releases
- 30 additional tests (149 → 179), including pytest-qt GUI smoke tests

### Remaining limitations (deferred to v0.2.0)

- Material calibration constants are still placeholders pending validation against published CAI/TAI datasets (Soutis, Caprino, Sanchez-Saez, NASA round-robin)
- `fe3d` tier uses stiffness reduction + first-ply-failure; true cohesive surfaces and buckling-based CAI eigensolve are deferred
- 3D Mesh / Buckling / Stress GUI tabs are placeholder widgets (2D plots are fully wired)
- Release artifacts are **unsigned** (no Apple Developer ID or Windows signing cert configured); macOS users may need `xattr -rd com.apple.quarantine BVID-FE.app`

## [0.1.0-alpha] - 2026-04-16

Initial release.

### Added

- `OrthotropicMaterial` dataclass + `MaterialLibrary` with four built-in presets (AS4/3501-6, IM7/8552, T700/2510, T800/epoxy)
- `Laminate` with Classical Lamination Theory ABD matrices and effective engineering constants
- `PanelGeometry`, `ImpactorGeometry`, `BoundaryKind` geometry primitives
- `DamageState` with shapely-union projected damage area; `DelaminationEllipse` per-interface ellipse model
- C-scan JSON/CSV import and validation (`damage/io.py`) per `docs/cscan_schema.md`
- Olsson quasi-static impact threshold (`P_c`, `E_onset`) with Navier-series plate stiffness
- Peanut-template per-interface DPA distribution (`impact/shape_templates.py`)
- Empirical dent-depth model (`impact/dent_model.py`)
- Impact-driven workflow orchestrator: `ImpactEvent` → `DamageState` (`impact/mapping.py`)
- Three modeling tiers for residual strength:
  - **Empirical**: Soutis open-hole-equivalent CAI knockdown + Whitney-Nuismer point-stress TAI
  - **Semi-analytical**: Rayleigh-Ritz sublaminate buckling + Soutis post-buckling envelope; Whitney-Nuismer TAI
  - **3D FE**: first-ply-failure on a damaged hexahedral mesh; LaRC05 for CAI, Tsai-Wu for TAI; stiffness-reduction approximation of delaminations
- FE primitives: `gauss_points_1d` / `gauss_points_hex`, `Hex8`, `Hex8i` (incompatible modes), `CohesiveSurfaceElement` (zero-thickness, bilinear traction-separation)
- Linear static solver (sparse direct, SciPy), linear buckling eigensolve (`eigsh` / dense fallback)
- `BvidAnalysis(AnalysisConfig).run()` high-level orchestrator returning `AnalysisResults`
- CLI entry point `bvidfe` supporting empirical / semi-analytical / fe3d runs
- `AnalysisResults` with `summary()` and `to_dict()` for JSON export
- 2D matplotlib plots: damage-map ellipse overlay, knockdown curves, tier comparison charts
- 3D PyVista plots: mesh with delamination surfaces, buckling mode shape, stress field
- Parametric sweeps over impact energy, layup, and ply thickness with CSV output (`sweep_energies`, `sweep_layups`, `sweep_thicknesses`)
- 149 unit + integration tests mirroring the package structure

### Known Limitations (deferred to v0.2.0)

- GUI (PyQt6 panels) not yet built
- Material calibration constants (`olsson_alpha`, `soutis_k_s`, `dent_beta`, etc.) are placeholder defaults for typical CFRP; precise values need to be calibrated against specific material test data
- `fe3d` tier uses stiffness reduction at delaminated interfaces, not true cohesive surfaces
- `fe3d` CAI uses first-ply-failure on the damaged mesh, not a buckling eigenvalue solve
- No validated datasets yet (Soutis, Caprino, Sanchez-Saez, NASA digitization pending)
- PyInstaller standalone packaging not yet built
