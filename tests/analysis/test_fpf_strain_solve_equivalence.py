"""Numerical equivalence between the vectorised and scalar-reference
implementations of ``_solve_failure_strain_analytic``.

The vectorised production routine in ``bvidfe.analysis.fe_tier`` replaces
a per-Gauss-point Python loop with batched ``larc05_index_batch`` /
``tsai_wu_index_batch`` calls. ``_solve_failure_strain_analytic_scalar_ref``
is the unchanged pre-vectorisation implementation, kept solely so this
test can prove the two paths return identical strain-at-failure values
on representative inputs. Any future regression in either path will
trip this test.

Two cases are exercised:

  * LaRC05 path  — the pure-quadratic branch: ``c = 1/sqrt(idx_ref)``.
  * Tsai-Wu path — the affine-quadratic branch including the linear-
    fallback (``|b| < 1e-14``) and disc < 0 mask edges.

Both run on a small fe3d mesh (~100 elements) so the test stays under a
second.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from bvidfe.analysis import AnalysisConfig, MeshParams
from bvidfe.analysis.fe_mesh import build_fe_mesh
from bvidfe.analysis.fe_tier import (
    _build_elements,
    _solve_failure_strain_analytic,
    _solve_failure_strain_analytic_scalar_ref,
)
from bvidfe.core.geometry import ImpactorGeometry, PanelGeometry
from bvidfe.core.laminate import Laminate
from bvidfe.core.material import MATERIAL_LIBRARY
from bvidfe.damage.state import DamageState, DelaminationEllipse
from bvidfe.impact.mapping import ImpactEvent


def _build_setup(damage: DamageState):
    cfg = AnalysisConfig(
        material="IM7/8552",
        layup_deg=[0.0, 90.0, 0.0, 90.0],
        ply_thickness_mm=0.2,
        panel=PanelGeometry(20.0, 10.0),
        loading="compression",
        tier="fe3d",
        impact=ImpactEvent(5.0, ImpactorGeometry(), mass_kg=5.5),
        mesh=MeshParams(elements_per_ply=1, in_plane_size_mm=2.5),
    )
    lam = Laminate(
        material=MATERIAL_LIBRARY[cfg.material],
        layup_deg=cfg.layup_deg,
        ply_thickness_mm=cfg.ply_thickness_mm,
    )
    mesh = build_fe_mesh(cfg, damage)
    elements = _build_elements(mesh, lam)
    return cfg, mesh, elements


@pytest.mark.parametrize("criterion", ["larc05", "tsai_wu"])
@pytest.mark.parametrize(
    "damage",
    [
        DamageState(delaminations=[], dent_depth_mm=0.0),
        DamageState(
            delaminations=[DelaminationEllipse(1, (10, 5), 6, 3, 0)],
            dent_depth_mm=0.2,
        ),
    ],
    ids=["pristine", "delaminated"],
)
def test_vectorised_matches_scalar_reference(criterion, damage):
    """Compression FPF: vectorised result must equal scalar-ref result."""
    cfg, mesh, elements = _build_setup(damage)
    eps_vec = _solve_failure_strain_analytic(
        cfg, mesh, elements, strain_sign=-1, criterion=criterion
    )
    eps_ref = _solve_failure_strain_analytic_scalar_ref(
        cfg, mesh, elements, strain_sign=-1, criterion=criterion
    )
    assert np.isfinite(eps_vec)
    assert np.isfinite(eps_ref)
    # Both paths share the same FE solve and the same algebra; the only
    # numerical-evaluation difference is BLAS-routed batched dot products
    # vs Python multiplications. Drift should be at floating-point noise
    # level.
    assert eps_vec == pytest.approx(eps_ref, rel=1e-10, abs=1e-12)


def test_vectorised_tension_path_also_matches_scalar_reference():
    """The tension path uses uniaxial_x_bcs (positive strain) and Tsai-Wu by
    default; verify the vectorised form is still equivalent under the
    +strain_sign branch."""
    damage = DamageState(
        delaminations=[DelaminationEllipse(1, (10, 5), 6, 3, 0)],
        dent_depth_mm=0.0,
    )
    cfg, mesh, elements = _build_setup(damage)
    eps_vec = _solve_failure_strain_analytic(
        cfg, mesh, elements, strain_sign=+1, criterion="tsai_wu"
    )
    eps_ref = _solve_failure_strain_analytic_scalar_ref(
        cfg, mesh, elements, strain_sign=+1, criterion="tsai_wu"
    )
    assert np.isfinite(eps_vec) and np.isfinite(eps_ref)
    assert eps_vec == pytest.approx(eps_ref, rel=1e-10, abs=1e-12)


def test_panel_boundary_changes_fpf_strain():
    """Issue #32: the FPF/TAI BC builder must honour ``panel.boundary``.

    Previously uniaxial_x_bcs ignored it, so the fe3d FPF/TAI residual was
    identical for clamped/simply_supported/free (silent no-op). Now the same
    u_z edge restraint the buckling path uses is applied, so the three
    boundary conditions must give measurably different — and physically
    ordered — failure strains: more out-of-plane edge restraint (clamped)
    is stiffer and fails sooner than the unrestrained (free) case.
    """
    damage = DamageState(delaminations=[], dent_depth_mm=0.0)
    cfg, mesh, elements = _build_setup(damage)

    def eps_for(boundary: str) -> float:
        panel = dataclasses.replace(cfg.panel, boundary=boundary)
        cfg_b = dataclasses.replace(cfg, panel=panel)
        return _solve_failure_strain_analytic(
            cfg_b, mesh, elements, strain_sign=-1, criterion="larc05"
        )

    eps_ss = eps_for("simply_supported")
    eps_cl = eps_for("clamped")
    eps_fr = eps_for("free")

    assert eps_cl != eps_ss
    assert eps_fr != eps_ss
    # Monotone: clamped (most u_z edge restraint) < simply_supported
    #           < free (no extra restraint).
    assert eps_cl < eps_ss < eps_fr, (eps_cl, eps_ss, eps_fr)
