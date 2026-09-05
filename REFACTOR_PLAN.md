# Refactoring plan: `legacy/` development chain -> `beyond32` package

Companion code of *Beyond the 32: Superconducting Pairing Channels of the Icosahedral
Group* (Eva Moss, 2026). The original scripts (kept unchanged under `legacy/` in the git history,
commit 215886b, and removed in the release commit) were a
development chain: one script `exec`s the head of another, intermediate results travel
through pickles, and every script recomputes the group from scratch. This document maps
each old script to the new modules and lists the reference values that the test-suite pins
literally.

## 1. Ground rules carried over from the task

1. No mathematical result changes. Exact values are compared exactly; the numerical
   minimisations to the stated precision. Any discrepancy is reported in `REPORT.md`,
   never "fixed".
2. Exact arithmetic stays exact. Everything group-theoretic is computed with sympy over
   Q(sqrt 5). Floating point only where the legacy scripts use it: BFGS / Nelder-Mead
   minimisation of the weak-coupling ratio, node search on a grid, SVD for the isotropy
   fixed spaces, numerical sphere quadrature for the H-candidate coordinates, and the D12
   least-squares character extraction.
3. No `exec`, no pickles, no module-level state. Plain functions, small frozen dataclasses,
   `functools.lru_cache` for in-run memoisation, optional on-disk cache under `.cache/`.
4. Deterministic: fixed seeds (`random.seed(3)` for the intertwiner seed matrix as in
   `gl_H.py`; `numpy` seeds 0 / 1 for the restarts as in `gl_min.py` / `gl_states.py`).
5. Only this folder is touched; git initialised here.

## 2. Speed-ups that do not change the mathematics

* The legacy code canonicalises every entry with `sympy.nsimplify(..., [sqrt(5)])`, which is
  slow. The package works in the sympy algebraic field `QQ<sqrt(5)>` (`QQ.algebraic_field`)
  and uses `DomainMatrix` for the big linear algebra (representation matrices on monomials,
  projectors, null spaces, rref). This is the same field and the same arithmetic, just
  without repeated numeric re-recognition of the same numbers.
* The action of a rotation on degree-l monomials is computed by polynomial multiplication
  over the field (`Poly` with `domain=QQ<sqrt5>`) instead of `subs` + `expand`.
* Everything derived from the group (rotations, classes, representation matrices per l,
  projectors) is computed once per run and memoised.

## 3. Map from legacy scripts to package modules

(Planning names; several were renamed during implementation, e.g. `rep_matrix` -> `rep_matrices`,
`P6` -> `invariant_P6`, `restrict_to` -> `restrict`, `sym2_decomposition` -> `sym2_projector_ranks`,
`irrep_matrices` -> `channel`, `gap_squared_components`/`invariants_NL` -> `quartic_invariants`,
`g_ground_state_exact` -> `g_stratum`. README.md and REPORT.md describe the final code.)

| legacy script | what it does | new home |
|---|---|---|
| `ih_basis.py` §"2I as quaternions", `rot`, classes by angle, `chi`, `inner`, five-fold axes | 120 quaternions, 60 rotations, class sizes [1,12,12,20,15], character table of I, orthonormality, axes | `groups.py` (`binary_icosahedral`, `icosahedral_rotations`, `IGroup`, `CHAR_I`, `fivefold_axes`, `threefold_axes`, `twofold_axes`) |
| `ih_basis.py` §"2I character table" (`qclasses`, `chi_j`, `galois`, `irr2`) | 9 classes of 2I by SU(2) angle, 9 irreps, orthonormality, sum d^2 = 120 | `groups.py` (`CHAR_2I`, `two_i_classes`) |
| `ih_basis.py` §"action on harmonic polynomials" (`monomials`, `coeff_vector`, `rep_matrix`, `harmonic_basis`, projectors, `branching`, `bases`) | rep. of I on degree-l polynomials, harmonic subspace, isotypic projectors, branching SO(3)->I, rref bases, dimension check | `harmonics.py` (`monomials`, `rep_matrix`, `harmonic_subspace`, `projector`, `branching`, `isotypic_basis`) |
| `ih_basis.py` `chi_l`, character-formula cross-check | m_Gamma(l) from the character formula | `harmonics.py` (`branching_by_characters`) |
| `ih_basis.py` §"Molien / Poincare dictionary" | m_A(l), l<=30; Molien series (1+t^15)/((1-t^6)(1-t^10)); S^3/2I spectrum | `molien.py` |
| `ih_basis.py` §"subgroup restrictions" (`restrict`) | I -> T, D5, D3 by characters; D2 marked TODO | `restrictions.py` (`restrict_to`, T/D5/D3 by characters; D2 by parities of the basis functions of `harmonics.py`, as in the paper's Table 5 caption) |
| `ih_basis.py` §"GL quartic invariants" (`chi_class_sq`) | Sym^2 Gamma by characters, counts | `gl.py` (`sym2_by_characters`) — cross-checked against the projector-based `sym2_decomposition` |
| `ih_basis.py` §"SU(2) -> 2I", pair decompositions | j=1/2..15/2 branching; antisym/sym squares of 2, 2', 4_s, 6 | `double_group.py` |
| `ih_seeds.py` (`to_phi`, `poly_phi`, `harm_proj`, `apply`, `tasks`, P6, hexad) | phi-form basis functions from seeds; l=6 invariant P6; hexad identity | `harmonics.py` (`phi_form`, `harmonic_part`, `basis_from_seed`, `paper_basis_functions`, `P6`, `hexad_identity`) |
| `gl_inv.py` (`mono_int`, `sph_inner`, `act`, `bases`, D(g), Sym^2 projectors) | sphere inner product; orthonormal irrep matrices T1,T2,G,H; homomorphism check; Sym^2 decomposition | `gl.py` (`sphere_inner`, `channel_basis`, `irrep_matrices`, `sym2_decomposition`) |
| `gl_inv2.py` (`harm_components`, `sph_norm2_complex`, N_L) | harmonic content of Delta^2, invariants N_L, sum rule | `gl.py` (`gap_squared_components`, `invariants_NL`) |
| `gl_inv3.py` (`lin_relation`) | N_L in terms of I1, I2 (and N2 for G); int|Delta|^4 | `gl.py` (`relations`) — exact linear solve instead of `sympy.solve` |
| `gl_H.py` | H: N4 = N4G + N4H split, seed-map intertwiner J (6x15 on monomials, `random.seed(3)`), isometry scale lambda, cross terms C_R, C_I, independence of the six | `gl.py` (`h_channel`) |
| `gl_inv3.py` §H (5x5 orthonormal-basis J, `random.seed(1)`) | earlier construction of J; superseded by `gl_H.py` | not ported (the paper's Table 8 uses the `gl_H.py` normalisation, see REPORT.md) |
| `gl_min.py` | BFGS minimisation of R per channel, 60 restarts, seed 0 | `gl.py` (`minimise_ratio`) |
| `gl_states.py` §G | 80 restarts seed 1, stabiliser, nodes, null-cone minimum | `gl.py` (`g_ground_state`, `stabiliser`, `node_fraction`) |
| `gl_states.py` §H candidates, `gl_final.py` §H | Y22/Y21 about C5, C3, C2 and cyclic state; N2, N4G, N4H, Re C, Im C | `gl.py` (`h_candidates`) |
| `gl_final.py` §G | exact R(kappa, phase) on the (1,1,1,kappa e^{i phi}) stratum, stationary kappa^2, node clusters (18, 12 on five-fold axes) | `gl.py` (`g_ground_state_exact`, `g_nodes`) |
| `isotropy.py` | subgroups C2, C3, C5, D2, D3, D5, T of I with 1-dim characters; fixed spaces by SVD; R_wc, TR, min|Delta| | `gl.py` (`isotropy_table`) |
| `shells.py` | permutation representations of I_h on the 12/20/30/60 orbits | `shells.py` (also the I-only decompositions) |
| `d12.py` | D12 character table, m mod 12 -> irrep, enforced nodes | `d12.py` (+ Sym^2 E_m and the circle weak-coupling ratios 3/2, 1 quoted in Appendix B) |
| (none) | LaTeX fragments, results.json, CLI | `latex.py`, `results.py`, `cli.py` |

## 4. Package layout (as requested)

```
beyond32/__init__.py  groups.py  harmonics.py  molien.py  restrictions.py  shells.py
         double_group.py  gl.py  d12.py  latex.py  cli.py  _exact.py (Q(sqrt5) helpers)
tests/test_<module>.py, conftest.py (slow marker)
tables/*.tex   results.json   README.md LICENSE CITATION.cff pyproject.toml requirements.txt
CHANGELOG.md REPORT.md
```

`_exact.py` is a private helper (field `QQ<sqrt5>`, `DomainMatrix` conversions, phi-form
printing); it is not part of the requested layout but keeps the exact arithmetic in one place.
`results.py` assembles `results.json` (plain data) from all modules; `latex.py` renders it.

## 5. Reference values pinned in the tests (literal)

### tests/test_groups.py
* |2I| = 120, |I| = 60, every R orthogonal with det 1.
* class sizes of I: [1, 12, 12, 20, 15] for [E, C5, C5^2, C3, C2].
* characters: A=(1,1,1,1,1); T1=(3, phi, 1-phi, 0, -1); T2=(3, 1-phi, phi, 0, -1);
  G=(4,-1,-1,1,0); H=(5,0,0,-1,1); orthonormal; sum d^2 = 60.
* 2I classes by SU(2) angle 0, 72, 120, 144, 180, 216, 240, 288, 360 deg with sizes
  [1,12,20,12,30,12,20,12,1]; irrep dims [1,2,2,3,3,4,4,5,6]; sum d^2 = 120;
  spinor irreps 2, 2', 4_s, 6 have character -dim on -1; 2 x 2' = G; orthonormal.
* five-fold axes = cyclic permutations of (+-phi, +-1, 0); three-fold (+-1,+-1,+-1);
  two-fold along x, y, z (and the other 12).

### tests/test_harmonics.py
* branching l=0..6: A; T1; H; T2+G; G+H; T1+T2+H; A+T1+G+H — by projectors and by characters.
* basis functions (phi-form): l=1 T1 x,y,z; l=2 H xy, yz, zx, x^2-y^2, 2z^2-x^2-y^2;
  l=3 T2 f_x = x^3 + 3 phi^-1 x y^2 - 3 phi x z^2; l=3 G g_x = x^3 - phi^2 x y^2 - phi^-2 x z^2, xyz;
  l=4 G u_0, u_x; l=4 H v_x, w_x (as in the task statement); all harmonic; all in the
  claimed isotypic component (fixed by the projector).
* P6 = x^6+y^6+z^6 + (3-21 phi)(x^4y^2+y^4z^2+z^4x^2) + (21 phi-18)(x^4z^2+y^4x^2+z^4y^2) + 90 x^2y^2z^2,
  harmonic and I-invariant; hexad identity sum_i (a_i.n)^6 - 6/7 = -(2/35) P6 on the sphere.

### tests/test_molien.py
* m_A(l), l=0..30 = [1,0,0,0,0,0,1,0,0,0,1,0,1,0,0,1,1,0,1,0,1,1,1,0,1,1,1,1,1,0,2]
  = coefficients of (1+t^15)/((1-t^6)(1-t^10)).
* S^3/2I spectrum (k, mult): (0,1),(12,13),(20,21),(24,25),(30,31),(32,33),(36,37),(40,41),
  (42,43),(44,45),(48,49),(50,51), ..., (60,122).

### tests/test_restrictions.py
* T: A->A; T1->T; T2->T; G->A+T; H->E+T.
* D5: A->A1; T1->A2+E1; T2->A2+E2; G->E1+E2; H->A1+E1+E2.
* D3: A->A1; T1->A2+E; T2->A2+E; G->A1+A2+E; H->A1+2E.
* D2: A->A; T1,T2->B1+B2+B3; G->A+B1+B2+B3; H->2A+B1+B2+B3.

### tests/test_shells.py
* under I: 12: A+T1+T2+H; 20: A+T1+T2+2G+H; 30: A+T1+T2+2G+3H; 60: A+3T1+3T2+4G+5H.
* under I_h: 12: Ag+Hg+T1u+T2u; 20: Ag+Gg+Hg+T1u+T2u+Gu; 30: Ag+Gg+2Hg+T1u+T2u+Gu+Hu;
  60: Ag+T1g+T2g+2Gg+3Hg+2T1u+2T2u+2Gu+2Hu.

### tests/test_double_group.py
* j=1/2: 2; 3/2: 4_s; 5/2: 6; 7/2: 2'+6; 9/2: 4_s+6; 11/2: 2+4_s+6; 13/2: 2+2'+4_s+6; 15/2: 4_s+2*6.
* 2x2 = A (a) + T1 (s); 2'x2' = A + T2; 4_s x 4_s = (A+H) + (T1+T2+G); 6x6 = (A+G+2H) + (2T1+2T2+G+H).

### tests/test_gl.py
* Sym^2: T1 = A+H; T2 = A+H; G = A+G+H; H = A+G+2H; Hermitian-form counts 2,2,3,6; TR-even 2,2,3,5.
* sphere norms (units of 4 pi): T1 1/3; H 1/15; T2 4/7; G 4/21, 4/21, 4/21, 1/105.
* D(g) orthogonal, trace = character, homomorphism.
* N_L relations: T1 N0 = I2, N2 = 6/5 I1 - 2/5 I2; T2 N0 = I2, N2 = 8/15 I1 - 8/45 I2;
  G N0 = I2, N4 = (112 I1 - 28 I2 - 135 N2)/121, N6 = (700 I1 + 100 I2 + 875 N2)/1573;
  H N0 = I2, N2 + N4 = (10 I1 - 2 I2)/7, N4 = N4G + N4H.
* int|Delta|^4: T1 3/5 (2I1+I2); T2 1687/2145 (2I1+I2); G (196 I1 + 119 I2 + 63 N2)/143; H 5/7 (2I1+I2).
* H intertwiner: |J h|^2/|h|^2 = (-7/20 + 47 sqrt5/30)^2; six forms linearly independent.
* ratios at |eta|=1: T1 6/5 (null cone), 9/5 (real); T2 3374/2145, 5061/2145; H 10/7, 15/7.
* G: global minimum 1.52545 (5 dp) at (1,1,1,kappa e^{i phi0}), kappa ~ 1.752, phi0 ~ 92.6 deg,
  I2/I1 ~ 2e-3; best null-cone 1.52721; fixed states (1,w,w^2,0) 1.5273; C5 chiral 1.6056;
  xyz 2.2028; g_x 2.4378; D3 A1 2.4476; D3 A2 2.2909.
* G ground state: 18 point nodes, 12 on the five-fold axes; G|C5 has no trivial character.
* H null-cone candidates (N2, N4G, N4H, Re C): Y22/C5 (0, .7619, .6667, 0); Y22/C3 (0, .5644, .8642, 0);
  Y22/C2 (0, .5952, .8333, 0); Y21/C5 (.6122, .7619, .0544, -.1825); Y21/C3 (.6122, .1411, .6752, .1014);
  Y21/C2 (.6122, .2381, .5782, .0570); cyclic (.8163, 0, .6122, .3423), Im C = -/+ 0.6186.
* isotropy subgroups with 1-dim fixed space (Table 7): T1: D5 A2, D3 A2, D2, C5 chi+-1, C3 chi+-1;
  T2: D5 A2, D3 A2, D2, C5 chi+-2, C3 chi+-1; G: T A, D2, D3 A1, D3 A2, C5 chi+-1,+-2, C3 chi+-1;
  H: D5 A1, D3 A1, D2, C5 chi+-2, C5 chi+-1, T 1E, T 2E.

### tests/test_d12.py
* 24 elements; irreps A1, A2, B1, B2, E1..E5; orthonormal; sum d^2 = 24.
* m mod 12: 0 -> A1 (+A2 from m=12); 6 -> B1+B2; 1..5 -> E_m; 7..11 -> E_{12-m}.
* enforced nodes: A2 on all 24 directions; B2 on C2'; B1 on C2''; E_m none; on the axis all but A1, A2.
* Sym^2 E_m = A1 + E_{2m} (m=1,2,4,5), A1 + B1 + B2 (m=3); quartic counts 2,2,3,2,2.
* circle weak coupling: R = 3/2 real, 1 chiral.

## 6. LaTeX fragments (`tables/`)

`tab_charI.tex` (Table 1), `tab_char2I.tex` (Table 2), `tab_branching.tex` (Table 3),
`tab_shells.tex` (Table 4), `tab_restrict.tex` (Table 5), `tab_GL.tex` (Table 6),
`tab_states.tex` (Table 7), `tab_Hcand.tex` (Table 8), `tab_D12.tex` (Eq. 24 / Appendix B
assignment), plus `eq_bases.tex` (Eqs. 4-9, 11), `eq_invariants.tex` (Eqs. 18-22),
`tab_molien.tex` (Eq. 15-16), `tab_double_group.tex` (Eqs. 12-13).

## 7. Speed budget

Full pipeline for l <= 6 in a few minutes. The only pieces that may exceed that are the
l = 5, 6 isotypic bases with rref (kept, but timed) and the exact independence check of the
six H forms; these are marked `slow` in pytest and skipped by `beyond32 all --fast`.
