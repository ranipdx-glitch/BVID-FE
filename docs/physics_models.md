# Physics Models

BVID-FE exposes three residual-strength modeling tiers that share a common
data contract (`DamageState` → tier engine → `AnalysisResults`) but differ in
fidelity, runtime, and the failure mechanisms they capture. This page
summarises what each tier solves; see the
[Python API reference](python_api.md) for usage.

## Empirical tier

Soutis open-hole-equivalent CAI model relates the delamination-affected stress
concentration around the projected damage area to a net-section failure.
Whitney-Nuismer point-stress and average-stress criteria are used for TAI.
Both models are closed-form and run in milliseconds.

## Semi-analytical tier

The damaged sublaminate above the largest delamination is treated as a plate
with reduced in-plane stiffness. A Rayleigh-Ritz energy method solves for the
sublaminate buckling load, and the Soutis post-buckling envelope predicts the
far-field CAI stress at overall failure. Whitney-Nuismer is retained for TAI.
Sublaminate eigenvalues are available in `AnalysisResults.buckling_eigenvalues`.

## 3D FE tier

A structured hexahedral mesh is built for the damaged laminate. Delaminated
interfaces are approximated by a **component-wise stiffness-reduction model**
(true cohesive surfaces deferred to a future release): each damaged element
carries an out-of-plane factor (`DAMAGE_OOP_FACTOR ≈ 0.05`) that scales the
through-thickness and transverse-shear stiffness, while in-plane stiffness is
preserved (the plies themselves remain intact). Inside the fiber-break core
under the impact site, in-plane stiffness is also reduced
(`DAMAGE_FIBER_BREAK_INPLANE_FACTOR ≈ 0.30`) to represent fiber bundle
fracture. First-ply-failure is evaluated at all Gauss points using LaRC05
(CAI) and Tsai-Wu (TAI). For CAI, a true linear buckling eigensolve runs
alongside FPF and the lower of the two governs.

## Knockdown definition and cross-tier comparability

`AnalysisResults.knockdown` is computed in exactly one place —
`BvidAnalysis.run()` (`src/bvidfe/analysis/bvid.py`):

```python
knockdown = residual_strength_MPa / pristine_strength_MPa
```

**The denominator is identical across all three tiers.**
`_pristine_strength()` is a thickness-weighted ply-average of the
lamina-level strengths from the material card:

- CAI: `Σ tᵢ (Xc·cos²θᵢ + Yc·sin²θᵢ) / Σ tᵢ`
- TAI: `Σ tᵢ (Xt·cos²θᵢ + Yt·sin²θᵢ) / Σ tᵢ`

**The numerator (residual strength) is what differs between tiers:**

| Tier | CAI residual stress | TAI residual stress |
| --- | --- | --- |
| `empirical` | Soutis: `σ₀ / (1 + k_s·(DPA/A_panel)^m)` | Whitney-Nuismer point-stress on equivalent hole |
| `semi_analytical` | `min(Soutis, σ_buckling_sublam)` | Delegates to Whitney-Nuismer (identical to `empirical`) |
| `fe3d` | `min(λ_crit·σ_ref, FPF_LaRC05)`, capped at σ₀ | FPF Tsai-Wu on damaged mesh, capped at σ₀ |

**What this means for users:**

- All three tiers report knockdown on the **same scale** (ratio relative to
  the same pristine baseline), so values are *qualitatively comparable*.
- They are **not** numerically interchangeable: each tier captures
  different failure mechanisms.
    - For **TAI**, `empirical` and `semi_analytical` are mathematically
      identical; `fe3d` differs.
    - For **CAI**, `semi_analytical ≤ empirical` always (the buckling floor
      only lowers the residual). `fe3d` is independent and dominated by
      stress concentration at the damage boundary rather than damage
      magnitude — see "Limitations" below.
- For **energy-scaling studies**, prefer `empirical` (Soutis scales with
  DPA) or `semi_analytical` (Rayleigh-Ritz scales with ellipse size).
  `fe3d` is intended for stress-field context and through-thickness damage
  visualization, not energy-dependent knockdown curves.

## Limitations

- Material calibration constants (`olsson_alpha`, `soutis_k_s`, `dent_beta`,
  and related parameters) are reasonable defaults for typical CFRP systems.
  Precise values need calibration against material-specific coupon test data
  before use in certification.
- LaRC05 is implemented as a minimal Hashin-3D reduction. Full plane-search
  fiber-kinking is deferred to a future release.
- The `fe3d` tier uses component-wise stiffness reduction at delaminated
  interfaces (in-plane preserved, out-of-plane reduced) instead of true
  cohesive surfaces with bilinear traction-separation laws. Cohesive surfaces
  are deferred to a future release.
- **The `fe3d` tier's knockdown is partially insensitive to impact energy**
  above the Olsson threshold. Linear buckling now responds to delamination
  size, but the FPF fallback strain is controlled by stress concentration
  at the healthy/damaged boundary rather than damage magnitude. For
  energy-dependent knockdown curves prefer `tier="empirical"` or
  `tier="semi_analytical"`. Full energy-monotonicity (in-plane pre-stress
  BCs + cohesive surfaces) is v0.3.0 scope.
- No validated datasets included; comparison against published Soutis,
  Caprino, and NASA datasets is on the roadmap.

## References

- Olsson, R. (2001). Analytical prediction of large mass impact damage in
  composite laminates. *Composites Part A*, 32(9), 1207-1215.
- Soutis, C. (1996). Compressive strength of unidirectional composites:
  measurement and prediction. *ASTM STP*, 1242, 168-176.
- Whitney, J.M. & Nuismer, R.J. (1974). Stress fracture criteria for
  laminated composites containing stress concentrations.
  *Journal of Composite Materials*, 8(3), 253-265.
- Tsai, S.W. & Wu, E.M. (1971). A general theory of strength for anisotropic
  materials. *Journal of Composite Materials*, 5(1), 58-80.
- Davila, C.G., Camanho, P.P., & Rose, C.A. (2005). Failure criteria for FRP
  laminates. NASA/TM-2005-213530 (LaRC05).
