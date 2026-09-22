"""Command-line interface.

    beyond32 all     [--fast] [--out DIR] [--cache]   everything: results.json + tables/
    beyond32 tables  [--results FILE] [--out DIR]     LaTeX fragments from an existing results.json
    beyond32 gl      [--fast]                         Ginzburg-Landau summary on stdout
    beyond32 check   [--fast]                         recompute the key numbers and compare them
                                                      with the values quoted in the paper
    beyond32 d6d                                      Appendix B: the point group D6d of the
                                                      Ta1.6Te quasicrystal and Table 10
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from fractions import Fraction

from . import __version__

CACHE_DIR = ".cache"

# keys every LaTeX emitter needs; a results.json from an older schema fails here with a
# clear message instead of a KeyError half-way through writing tables/
_REQUIRED_KEYS = {"package": ("version",), "groups": ("I", "2I"),
                  "harmonics": ("branching_by_characters", "basis_functions_terms", "P6_terms", "hexad"),
                  "molien": ("m_A", "spectrum"), "restrictions": ("table",), "shells": ("table",),
                  "double_group": ("eq12", "eq13"),
                  "gl": ("sym2", "relations", "weak_coupling_minima", "G_ground_state", "G_stratum",
                         "isotropy", "H_candidates"),
                  "d6d": ("table10", "asoc", "restriction_D2d", "field_orbit")}


def _load_results(path: str | None):
    for p in ([path] if path else []) + ["results.json", os.path.join(CACHE_DIR, "results.json")]:
        if p and os.path.exists(p):
            with open(p) as f:
                return json.load(f), p
    return None, None


def schema_problem(res) -> str | None:
    """None if ``res`` has every key the LaTeX emitters read, else a message naming the
    first missing key (and the version the file was written by)."""
    for section, keys in _REQUIRED_KEYS.items():
        if section not in res or not isinstance(res[section], dict):
            missing = section
        else:
            missing = next((f"{section}/{k}" for k in keys if k not in res[section]), None)
        if missing:
            ver = (res.get("package") or {}).get("version", "unknown")
            return (f"results.json lacks '{missing}' (written by beyond32 {ver}, this is {__version__}); "
                    f"regenerate it with 'beyond32 all'")
    return None


def cmd_all(args) -> int:
    from .results import collect
    from .latex import write_tables

    t0 = time.time()
    res = collect(fast=args.fast)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(res, f, indent=1)
    if args.cache:
        os.makedirs(os.path.join(args.out, CACHE_DIR), exist_ok=True)
        with open(os.path.join(args.out, CACHE_DIR, "results.json"), "w") as f:
            json.dump(res, f, indent=1)
    paths = write_tables(res, os.path.join(args.out, "tables"))
    print(f"[beyond32] results.json written; {len(paths)} LaTeX fragments in tables/ "
          f"({time.time() - t0:.0f} s; per section: {res['package']['runtime_seconds']})")
    return 0


def cmd_tables(args) -> int:
    from .latex import write_tables

    res, src = _load_results(args.results)
    if res is None:
        print("[beyond32] no results.json found; computing (use 'beyond32 all' to write it)")
        from .results import collect
        res = collect(fast=args.fast)
    else:
        print(f"[beyond32] using {src}")
        problem = schema_problem(res)
        if problem:
            print(f"[beyond32] {problem}")
            return 1
        if res["package"]["version"] != __version__:
            print(f"[beyond32] note: {src} was written by beyond32 {res['package']['version']}")
    paths = write_tables(res, args.out)
    print("\n".join(paths))
    return 0


def cmd_gl(args) -> int:
    from .results import gl_section

    res = gl_section(fast=args.fast)
    print("Sym^2 decompositions / Hermitian forms / quartic terms (Table 6):")
    for ch, d in res["sym2"].items():
        print(f"  {ch:3s} {d['sym2']}  {d['hermitian_forms']}  {d['quartic_terms']}")
    print("\nSphere norms of the basis functions (units of 4 pi):")
    for ch, n in res["norms2"].items():
        print(f"  {ch:3s} {n}")
    print("\nRelations among the invariants (Eqs. 18-22):")
    for ch, rel in res["relations"].items():
        for k, v in rel.items():
            print(f"  {ch:3s} {k:6s} = {v}")
    print(f"\nH channel: |J h|^2/|h|^2 = {res['H']['lam']}; six forms independent: {res['H']['six_independent']}")
    print("\nWeak-coupling minima of R = int|Delta|^4/(int|Delta|^2)^2:")
    for ch, m in res["weak_coupling_minima"].items():
        print(f"  {ch:3s} min R = {m['R_min']:.6f} ({m['R_min_fraction'] or 'not a small fraction'}), |eta.eta|^2 = {m['I2']:.2e}")
    gs, st, nd = res["G_ground_state"], res["G_stratum"], res["G_nodes"]
    print(f"\nG ground state: R = {gs['R']:.6f}, eta ~ (1,1,1,kappa e^{{i phi0}}) with kappa = {st['kappa']:.3f}, "
          f"phi0 = {st['phi0_deg']:.1f} deg, I2/I1 = {st['I2_over_I1']:.1e}; best null-cone state R = {gs['null_cone_R']:.6f}")
    print(f"  stabiliser {gs['stabiliser']}, time reversal x {gs['tr_stabiliser']}; "
          f"{nd['n_nodes']} point nodes, {nd['on_fivefold_axes']} on the five-fold axes")
    print("\nSymmetry-fixed states (Table 7):")
    print(f"  {'ch':3s} {'K':3s} {'chi':5s} {'R_wc':>8s} {'TR':>3s} {'min|D|':>7s}")
    for r in res["isotropy"]:
        if r["table7"]:
            print(f"  {r['channel']:3s} {r['subgroup']:3s} {r['character']:5s} {r['R']:8.4f} "
                  f"{'yes' if r['time_reversal'] else 'no':>3s} {r['min_gap']:7.3f}")
    print("\nNull-cone H candidates (Table 8): N2, N4G, N4H, Re C, Im C")
    for k, r in res["H_candidates"].items():
        print(f"  {k:30s} {r['N2']:7.4f} {r['N4G']:7.4f} {r['N4H']:7.4f} {r['ReC']:8.4f} {r['ImC']:8.4f}")
    return 0


def _fmt_function(f) -> str:
    """A basis function of results.json ('cos(2*phi)' or ['sin(5*phi)', 'cos(5*phi)', '0'])."""
    return "(" + ", ".join(f) + ")" if isinstance(f, list) else str(f)


def cmd_d6d(args) -> int:
    from . import d6d

    s = d6d.summary()
    f = s["facts"]
    print(f"D6d = -12m2, the point group of the Ta1.6Te dodecagonal quasicrystal (Yamamoto 2004): "
          f"order {f['order']}, {f['n_classes']} classes")
    print(f"  inversion: {f['inversion']}; horizontal mirror: {f['horizontal_mirror']}; true 12-fold rotation: "
          f"{f['true_twelvefold_rotation']} (S12 has order {f['S12_order']}, largest proper rotation order "
          f"{f['max_proper_rotation_order']})")
    print(f"  mirror planes contain the directions {f['mirror_directions_deg']} deg (framework edges); "
          f"two-fold axes at {f['twofold_axes_deg']} deg")
    ct = s["character_table"]
    print("\nCharacter table (classes: " + ", ".join(f"{n} {c}" if n > 1 else c for c, n in zip(ct["classes"], ct["sizes"])) + "):")
    for row in ct["rows"]:
        print(f"  {row['irrep']:3s} " + " ".join(f"{x:>8s}" for x in row["chars"]))
    print("\nTable 10: pair functions on the in-plane circle (singlet psi -> psi(g^-1 k); triplet d -> det(g) g d(g^-1 k))")
    for key, title in (("singlet", "singlet psi(phi)"), ("triplet_inplane", "triplet, in-plane d"),
                       ("triplet_z", "triplet, d_z")):
        print(f"  {title}:")
        for sec in s[key]:
            print(f"    m = {sec['m']}: {sec['irreps']}")
            for irrep, funcs in sec["basis"].items():
                print(f"      {irrep:3s} " + "; ".join(_fmt_function(fn) for fn in funcs))
    a = s["asoc"]
    print("\nAntisymmetric spin-orbit vector g(k) (odd, axial), A1 multiplicities for |m| <= 6: "
          + ", ".join(f"m = {m}: in-plane {v['inplane']}, d_z {v['z']}" for m, v in a["A1_content"].items()))
    print(f"  g ~ {_fmt_function(a['g'])}")
    print("\nRestriction D6d -> D2d (-42m; S4 = S12^3, two-fold axis at 15 deg):")
    for k, v in s["restriction_D2d"]["table"].items():
        print(f"  {k:3s} -> {v}")
    fo = s["field_orbit"]
    print(f"\nOrbit of an in-plane field direction (H -> -H included): {fo['angles_deg']} deg; "
          f"field-angle period {fo['period_deg']} deg")
    return 0


def _reference_checks(res):
    """(name, computed, expected) triples for the values quoted in the paper."""
    R = []
    g = res["groups"]
    R.append(("I class sizes", g["I"]["class_sizes"], [1, 12, 12, 20, 15]))
    R.append(("2I class sizes", g["2I"]["class_sizes"], [1, 12, 20, 12, 30, 12, 20, 12, 1]))
    R.append(("I character table orthonormal", g["I"]["orthonormal"], True))
    R.append(("2I character table orthonormal", g["2I"]["orthonormal"], True))
    R.append(("T1 characters", g["I"]["character_table"]["T1"], ["3", "varphi", "1 - varphi", "0", "-1"]))
    br = res["harmonics"]["branching_by_characters"]
    R.append(("branching l=3", br["3"], {"T2": 1, "G": 1}))
    R.append(("branching l=6", br["6"], {"A": 1, "T1": 1, "G": 1, "H": 1}))
    R.append(("m_A(l), l<=30", res["molien"]["m_A"],
              [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 2]))
    R.append(("Poincare (60, 122)", [60, 122] in res["molien"]["spectrum"], True))
    rt = res["restrictions"]["table"]
    R.append(("G -> T", rt["G"]["T"], {"A": 1, "T": 1}))
    R.append(("H -> D5", rt["H"]["D5"], {"A1": 1, "E1": 1, "E2": 1}))
    R.append(("H -> D2", rt["H"]["D2"], {"A": 2, "B1": 1, "B2": 1, "B3": 1}))
    sh = {r["orbit"]: r for r in res["shells"]["table"]}
    R.append(("12-orbit under I_h", sh["12"]["decomposition_Ih"], {"Ag": 1, "Hg": 1, "T1u": 1, "T2u": 1}))
    R.append(("60-orbit under I", sh["60"]["decomposition_I"], {"A": 1, "T1": 3, "T2": 3, "G": 4, "H": 5}))
    dg = {r["j"]: r["irreps"] for r in res["double_group"]["eq12"]}
    R.append(("j=15/2 -> 2I", dg["15/2"], {"4s": 1, "6": 2}))
    gl = res["gl"]
    R.append(("Sym^2 H", gl["sym2"]["H"]["sym2"], {"A": 1, "G": 1, "H": 2}))
    R.append(("G: N4 relation", gl["relations"]["G"]["N4"], {"I1": "112/121", "I2": "-28/121", "N2": "-135/121"}))
    R.append(("H: quartic", gl["relations"]["H"]["quartic"], {"I1": "10/7", "I2": "5/7"}))
    R.append(("H: six forms independent", gl["H"]["six_independent"], True))
    R.append(("H: lambda", abs(gl["H"]["lam_float"] - (-7 / 20 + 47 * 5 ** 0.5 / 30) ** 2) < 1e-9, True))
    R.append(("H: sqrt(lambda)", gl["H"]["lam_sqrt"], "-7/20 + 47*sqrt(5)/30"))
    R.append(("N_L terms (legacy gl_inv2)", gl["N_terms"],
              {"T1": {"N0": 9, "N2": 12}, "T2": {"N0": 9, "N2": 12, "N4": 12, "N6": 12},
               "G": {"N0": 16, "N2": 21, "N4": 28, "N6": 28}, "H": {"N0": 25, "N2": 52, "N4": 53}}))
    m = gl["weak_coupling_minima"]
    R.append(("R of real / null-cone states (exact)",
              [[str(Fraction(m[ch][k])) for k in ("real", "null_cone")] for ch in ("T1", "T2", "H")],
              [[str(Fraction(9, 5)), str(Fraction(6, 5))], [str(Fraction(5061, 2145)), str(Fraction(3374, 2145))],
               [str(Fraction(15, 7)), str(Fraction(10, 7))]]))
    R.append(("min R (T1)", round(m["T1"]["R_min"], 5), round(6 / 5, 5)))
    R.append(("min R (T2)", round(m["T2"]["R_min"], 5), round(3374 / 2145, 5)))
    R.append(("min R (H)", round(m["H"]["R_min"], 5), round(10 / 7, 5)))
    R.append(("min R (G)", round(m["G"]["R_min"], 5), 1.52545))
    R.append(("best null-cone G", round(gl["G_ground_state"]["null_cone_R"], 5), 1.52721))
    R.append(("kappa", round(gl["G_stratum"]["kappa"], 3), 1.752))
    R.append(("phi0 (deg)", round(gl["G_stratum"]["phi0_deg"], 1), 92.6))
    R.append(("G nodes", [gl["G_nodes"]["n_nodes"], gl["G_nodes"]["on_fivefold_axes"]], [18, 12]))
    hc = gl["H_candidates"]
    R.append(("Y21 about C5", [round(hc["Y21 about C5"][k], 4) for k in ("N2", "N4G", "N4H", "ReC")],
              [0.6122, 0.7619, 0.0544, -0.1825]))
    R.append(("cyclic Im C", round(hc["cyclic (T)"]["ImC"], 4), -0.6186))
    d = res["d6d"]
    R.append(("D6d: order, classes, inversion, sigma_h, true C12",
              [d["facts"][k] for k in ("order", "n_classes", "inversion", "horizontal_mirror", "true_twelvefold_rotation")],
              [24, 9, False, False, False]))
    R.append(("D6d singlet m = 0, 2, 4, 6", [r["irreps"] for r in d["singlet"]], ["A1", "E2", "E4", "B1 + B2"]))
    R.append(("D6d triplet in-plane m = 1, 3, 5", [r["irreps"] for r in d["triplet_inplane"]],
              ["B1 + B2 + E4", "E2 + E4", "A1 + A2 + E2"]))
    R.append(("D6d triplet d_z m = 1, 3, 5", [r["irreps"] for r in d["triplet_z"]], ["E1", "E3", "E5"]))
    R.append(("D6d spin-orbit vector", d["asoc"]["g"], ["sin(5*phi)", "cos(5*phi)", "0"]))
    R.append(("D6d E2, E4 -> D2d", [d["restriction_D2d"]["table"][k] for k in ("E2", "E4")], ["B1 + B2", "A1 + A2"]))
    R.append(("D6d field-angle period (deg)", d["field_orbit"]["period_deg"], 30))
    return R


def cmd_check(args) -> int:
    from .results import collect

    res = collect(fast=args.fast)
    res = json.loads(json.dumps(res))          # same view as results.json
    ok = True
    for name, got, exp in _reference_checks(res):
        good = got == exp
        ok &= good
        print(f"  [{'ok' if good else 'FAIL'}] {name}: {got}" + ("" if good else f"  (expected {exp})"))
    print("[beyond32] all reference values reproduced" if ok else "[beyond32] DISCREPANCY -- see above")
    return 0 if ok else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="beyond32", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("all", help="compute everything, write results.json and tables/")
    a.add_argument("--fast", action="store_true", help="skip the optional slow cross-checks")
    a.add_argument("--out", default=".", help="output directory (default: current directory)")
    a.add_argument("--cache", action="store_true", help="also keep a copy under .cache/")
    a.set_defaults(func=cmd_all)
    t = sub.add_parser("tables", help="write the LaTeX fragments from results.json")
    t.add_argument("--results", default=None,
                   help="path to results.json (default: ./results.json, then ./.cache/results.json)")
    t.add_argument("--out", default="tables", help="output directory for the .tex fragments (default: tables)")
    t.add_argument("--fast", action="store_true",
                   help="if no results.json is found, compute it without the slow cross-checks")
    t.set_defaults(func=cmd_tables)
    gcmd = sub.add_parser("gl", help="print the Ginzburg-Landau summary")
    gcmd.add_argument("--fast", action="store_true", help="skip the Sym^2 projector-rank cross-check")
    gcmd.set_defaults(func=cmd_gl)
    c = sub.add_parser("check", help="recompute and compare with the values quoted in the paper")
    c.add_argument("--fast", action="store_true", help="skip the optional slow cross-checks")
    c.set_defaults(func=cmd_check)
    dcmd = sub.add_parser("d6d", help="print Appendix B: the point group D6d, its character table and Table 10")
    dcmd.set_defaults(func=cmd_d6d)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
