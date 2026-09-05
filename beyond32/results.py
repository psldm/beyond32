"""Collect every number of the paper into one JSON-serialisable dictionary (results.json).

``collect(fast=False)`` runs the whole pipeline (groups -> harmonics -> molien ->
restrictions -> shells -> double group -> Ginzburg-Landau -> D12) and returns plain Python
data: strings for exact numbers (sympy), floats for the numerical minimisations, lists for
vectors.  ``beyond32 all`` writes it to ``results.json``; ``latex.write_tables`` turns it
into the LaTeX fragments of ``tables/``.
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
import time
from fractions import Fraction
from typing import Any, Dict, Optional

import numpy as np
import sympy as sp

from . import __version__
from ._exact import phi_form


# --------------------------------------------------------------------------- JSON helpers
def jsonable(obj: Any) -> Any:
    """Recursively convert sympy / numpy / dataclass objects to JSON-serialisable data."""
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (complex, np.complexfloating)):
        return {"re": float(obj.real), "im": float(obj.imag)}
    if isinstance(obj, np.ndarray):
        return [jsonable(x) for x in obj.tolist()]
    if isinstance(obj, Fraction):
        return str(obj)
    if isinstance(obj, sp.Matrix):
        return [[jsonable(obj[i, j]) for j in range(obj.cols)] for i in range(obj.rows)]
    if isinstance(obj, sp.Basic):
        if obj.is_Integer:
            return int(obj)
        return str(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(x) for x in obj]
    return str(obj)


def rational_if_close(x: float, max_den: int = 5000, tol: float = 1e-9) -> Optional[str]:
    """'9/5' if x is (numerically) a fraction with a small denominator, else None."""
    fr = Fraction(x).limit_denominator(max_den)
    if abs(float(fr) - x) < tol:
        return str(fr)
    return None


def phi_str(e) -> str:
    """An element of Q(sqrt5) as 'p + q*varphi'."""
    return str(phi_form(e))


def poly_terms(poly) -> list:
    """A polynomial in x, y, z with coefficients in Q(sqrt5) as a list of
    [[a, b, c], p, q] with coefficient p + q*varphi (p, q rational strings), terms ordered by
    descending exponents of x, then y, then z."""
    from ._exact import ab as _ab
    from .harmonics import coeff_dict

    out = []
    for e, c in sorted(coeff_dict(poly).items(), reverse=True):
        a, b = _ab(c)                      # c = a + b sqrt5 = (a - b) + 2b varphi
        out.append([list(e), str(a - b), str(2 * b)])
    return out


# --------------------------------------------------------------------------- sections
def groups_section() -> Dict[str, Any]:
    from . import groups as g

    G = g.icosahedral_group()
    T1 = g.CHAR_I
    T2 = g.character_table_2I()
    Th = g.character_table_Ih()
    return {
        "I": {"order": G.order, "classes": list(T1.classes), "class_sizes": list(g.class_sizes_I()),
              "character_table": {n: [phi_str(c) for c in row] for n, row in T1.chars.items()},
              "character_table_sqrt5": {n: [str(c) for c in row] for n, row in T1.chars.items()},
              "dims": {n: T1.dim(n) for n in T1.irreps}, "orthonormal": T1.is_orthonormal(),
              "sum_of_squared_dims": sum(T1.dim(n) ** 2 for n in T1.irreps)},
        "2I": {"order": 120, "angles_deg": list(g.ANGLES_2I), "class_sizes": list(T2.sizes),
               "cos_half_angle": [str(c) for c in g.COS_HALF_2I],
               "character_table": {n: [phi_str(c) for c in row] for n, row in T2.chars.items()},
               "dims": {n: T2.dim(n) for n in T2.irreps}, "orthonormal": T2.is_orthonormal(),
               "sum_of_squared_dims": sum(T2.dim(n) ** 2 for n in T2.irreps),
               "closed": g.check_closure_2I()},
        "Ih": {"classes": list(Th.classes), "class_sizes": list(Th.sizes)},
        "axes": {"fivefold": [[phi_str(c) for c in a] for a in g.fivefold_axes()],
                 "threefold": [[str(c) for c in a] for a in g.threefold_axes()],
                 "twofold": [[phi_str(c) for c in a] for a in g.twofold_axes()]},
    }


def harmonics_section(lmax: int = 6, fast: bool = False) -> Dict[str, Any]:
    from . import harmonics as h

    lb = lmax if not fast else min(lmax, 4)
    out = {
        "branching": {l: h.branching(l) for l in range(lb + 1)},
        "branching_by_characters": {l: h.branching_by_characters(l) for l in range(lmax + 1)},
        "basis_functions": {f"l={l},{irr}": {name: str(h.phi_form_poly(f)) for name, f in funcs.items()}
                            for (l, irr), funcs in h.paper_basis_functions().items()},
        "basis_functions_terms": {f"l={l},{irr}": {name: poly_terms(f) for name, f in funcs.items()}
                                  for (l, irr), funcs in h.paper_basis_functions().items()},
        "P6_terms": poly_terms(h.invariant_P6()),
        "seed_functions": {f"l={l},{irr},seed={s}": str(h.phi_form_poly(f))
                           for (l, irr, s), f in h.seed_basis_functions().items()},
        "P6": str(h.phi_form_poly(h.invariant_P6())),
        "P6_harmonic": h.is_harmonic(h.invariant_P6()),
    }
    hx = h.hexad_identity()
    out["hexad"] = {"c": str(hx["c"]), "e": str(hx["e"]),
                    "identity": f"sum_i (a_i.n)^6 = {hx['c']} P6 + {hx['e']} r^6"}
    if not fast:
        out["isotypic_bases"] = {f"l={l},{irr}": [str(p) for p in h.isotypic_basis(l, irr)]
                                 for l in range(lmax + 1) for irr in h.branching(l)}
    return out


def gl_section(fast: bool = False) -> Dict[str, Any]:
    from . import gl

    out: Dict[str, Any] = {}
    out["basis"] = {n: [str(b) for b in gl.channel(n).basis] for n in gl.CHANNELS}
    out["norms2"] = {n: [str(x) for x in gl.channel(n).norms2] for n in gl.CHANNELS}
    out["sym2"] = gl.sym2_table()
    if not fast:
        out["sym2_projector_ranks"] = {n: gl.sym2_projector_ranks(n) for n in gl.CHANNELS}
    out["harmonic_content"] = {n: sorted(gl.quartic_invariants(n).components) for n in gl.CHANNELS}
    out["relations"] = {n: {k: {kk: str(vv) for kk, vv in v.items()} for k, v in gl.relations(n).items()}
                        for n in gl.CHANNELS}
    out["N_terms"] = {n: {f"N{L}": len(sp.Poly(gl.quartic_invariants(n).expr(f"N{L}"),
                                                *gl.quartic_invariants(n).gens).terms())
                          for L in sorted(gl.quartic_invariants(n).N)} for n in gl.CHANNELS}
    H = gl.h_channel()
    # lam is a perfect square in Q(sqrt5): denest sqrt(lam) and verify it exactly
    lam_sqrt = sp.sqrtdenest(sp.sqrt(H.lam))
    if sp.expand(lam_sqrt**2 - H.lam) != 0:
        lam_sqrt = None
    out["H"] = {"lam": str(H.lam), "lam_float": float(sp.N(H.lam)),
                "lam_sqrt": None if lam_sqrt is None else str(lam_sqrt),
                "six_independent": H.six_independent,
                "J_seed": "random.seed(3), Rational(randint(-4,4), randint(1,3)) 6x15 row-major",
                "J": [[str(x) for x in row] for row in H.J.tolist()],
                "CR_terms": len(sp.Poly(H.expr("CR"), *H.inv.gens).terms()),
                "CIi_terms": len(sp.Poly(H.expr("CIi"), *H.inv.gens).terms())}
    minima = {}
    for n in gl.CHANNELS:
        m = gl.minimise_ratio(n)
        minima[n] = {"R_min": m.value, "R_min_fraction": rational_if_close(m.value), "I2": m.I2,
                     "fraction_of_restarts_at_min": m.fraction_at_min,
                     "R_max_over_restarts": float(m.all_values[-1]), "eta": m.eta}
    # the extremal values of the three SO(3)-like channels (Eq. 22): R at the real state
    # (1, 0, ...) and at the null-cone state (1, i, 0, ...), exactly (closed forms, reduced
    # fractions: the paper's 5061/2145 is 1687/715) and numerically
    for n, d in ((3, "T1"), (3, "T2"), (5, "H")):
        minima[d]["real"] = str(gl.weak_coupling_ratio_exact(d, [1] + [0] * (n - 1)))
        minima[d]["null_cone"] = str(gl.weak_coupling_ratio_exact(d, [1, sp.I] + [0] * (n - 2)))
        e0 = np.zeros(n, dtype=complex)
        e0[0] = 1
        en = np.zeros(n, dtype=complex)
        en[0], en[1] = 1, 1j
        minima[d]["real_R"] = gl.weak_coupling_ratio(d, e0)
        minima[d]["null_cone_R"] = gl.weak_coupling_ratio(d, en / np.sqrt(2))
    out["weak_coupling_minima"] = minima
    gs = gl.g_ground_state()
    out["G_ground_state"] = gs
    st = gl.g_stratum()
    out["G_stratum"] = {**st, "R_expr": str(st["R_expr"]), "R_u": str(st["R_u"]),
                        "stationary_u": [str(u) for u in st["stationary_u"]],
                        "stationary_R": [str(r) for r in st["stationary_R"]],
                        "stationary_R_float": [float(r) for r in st["stationary_R"]]}
    nd = gl.g_nodes()
    out["G_nodes"] = nd
    rows = gl.isotropy_table()
    fixed = gl.symmetry_fixed_states()
    fixed_keys = {(r.channel, r.subgroup, r.character) for r in fixed}
    def r_exact(r):
        if r.dim_fixed != 1:
            return None
        val = gl.exact_ratio_of_fixed_state(r.channel, r.subgroup, r.character)
        return None if val is None else str(val)

    out["isotropy"] = [{**dataclasses.asdict(r), "R_fraction": rational_if_close(r.R) if r.R is not None else None,
                        "R_exact": r_exact(r),
                        "moduli": None if r.eta is None else np.abs(r.eta),
                        "table7": (r.channel, r.subgroup, r.character) in fixed_keys} for r in rows]
    out["subgroup_orders"] = {k: len(v) for k, v in gl.subgroups().items()}
    out["H_candidates"] = gl.h_candidates()
    return out


def collect(fast: bool = False, lmax: int = 6) -> Dict[str, Any]:
    """Run everything and return the results dictionary (JSON-serialisable)."""
    from . import d12, double_group, molien, restrictions, shells

    timings: Dict[str, float] = {}
    res: Dict[str, Any] = {"package": {"name": "beyond32", "version": __version__,
                                       "generated": _dt.datetime.now().isoformat(timespec="seconds"),
                                       "fast": fast, "lmax": lmax}}

    def timed(name, fn):
        t = time.time()
        res[name] = jsonable(fn())
        timings[name] = round(time.time() - t, 2)

    timed("groups", groups_section)
    timed("harmonics", lambda: harmonics_section(lmax, fast))
    timed("molien", lambda: {**molien.paper_dictionary(), "table": molien.format_molien(),
                             "closed_form_from_group": (None if fast else molien.check_molien_closed_form())})
    timed("restrictions", lambda: {"table": restrictions.restriction_table(),
                                   "rows": restrictions.restriction_table_rows(),
                                   "d2_labels": restrictions.restriction_table_data()["d2_labels"],
                                   "cyclic": restrictions.cyclic_table(),
                                   "cyclic_rows": restrictions.cyclic_table_rows()})
    timed("shells", lambda: {"table": shells.shells_table(), "rows": shells.format_shells_table()})
    timed("double_group", double_group.summary)
    timed("gl", lambda: gl_section(fast))
    timed("d12", d12.summary)
    res["package"]["runtime_seconds"] = timings
    res["package"]["runtime_total_seconds"] = round(sum(timings.values()), 2)
    return res
