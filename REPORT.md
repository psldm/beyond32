# Refactoring report

Companion code of *Beyond the 32: Superconducting Pairing Channels of the Icosahedral Group*
(Eva Moss, 2026).  The development chain (12 scripts plus a README, `exec` + pickles; kept
unchanged as `legacy/` in the git history from commit 215886b until the release commit) was
turned into the installable package `beyond32` (`pip install -e .`, `beyond32 all`).

**Result: every reference value of the paper is reproduced; there is no discrepancy.**

## 1. What was reproduced

Exact values are compared exactly (sympy over Q(√5), or Q(√3, √5) where the normalised
H basis brings in √3); numerical values to the precision quoted in the paper.  All of the
following are pinned literally in `tests/` (107 tests); `beyond32 check` recomputes a
selection of about 35 key items and compares them with the paper.

| item | value | test |
|---|---|---|
| I: 60 rotations, class sizes [1,12,12,20,15], characters of A, T1, T2, G, H, orthonormal, Σd² = 60 | Table 1 | `test_groups` |
| 2I: 120 quaternions, closed; classes by SU(2) angle with sizes [1,12,20,12,30,12,20,12,1]; dims [1,2,2,3,3,4,4,5,6]; spinor irreps −dim on −1; 2 ⊗ 2′ = G; Table 2 literally | Table 2 | `test_groups` |
| five-fold axes = cyclic permutations of (±φ, ±1, 0); three-fold (1,±1,±1); two-fold x, y, z | Eq. 1 | `test_groups` |
| branching ℓ = 0…6: A; T1; H; T2+G; G+H; T1+T2+H; A+T1+G+H (projectors and characters) | Table 3 | `test_harmonics` |
| basis functions f_x, g_x, xyz, u_0, u_x, v_x, w_x in φ-form; harmonic; in their isotypic components | Eqs. 4–9 | `test_harmonics` |
| P6 literally; harmonic and I-invariant; hexad identity Σ(a_i·n)⁶ = −(2/35) P6 + (6/7) r⁶ | Eqs. 10–11 | `test_harmonics` |
| m_A(ℓ), ℓ ≤ 30 = [1,0,0,0,0,0,1,0,0,0,1,0,1,0,0,1,1,0,1,0,1,1,1,0,1,1,1,1,1,0,2] = coefficients of (1+t¹⁵)/((1−t⁶)(1−t¹⁰)); spectrum (0,1),(12,13),…,(60,122) | Eqs. 15–16 | `test_molien` |
| restrictions to T, D5, D3 (characters) and D2 (parities), all 20 entries | Table 5 | `test_restrictions` |
| G ↓ C5 contains no trivial character; every other channel does | Section 7 | `test_restrictions` |
| 12/20/30/60-orbits under I and under I_h, all eight decompositions | Table 4 | `test_shells` |
| SU(2) → 2I for j = 1/2 … 15/2; pair decompositions of 2, 2′, 4_s, 6 | Eqs. 12–13 | `test_double_group` |
| Sym²: A+H, A+H, A+G+H, A+G+2H; counts 2,2,3,6 and 2,2,3,5 (characters and projector ranks) | Table 6 | `test_gl` |
| sphere norms 1/3; 1/15; 4/7; 4/21, 4/21, 4/21, 1/105 | Section 6 | `test_gl` |
| N_L relations of T1, T2, G, H and ∫\|Δ\|⁴ for all four channels | Eqs. 18–22 | `test_gl` |
| \|Jh\|²/\|h\|² = (−7/20 + 47√5/30)² with the legacy seed; six forms independent; N4 = N4G + N4H | Section 6.2 | `test_gl` |
| R: 6/5, 9/5 (T1); 3374/2145, 5061/2145 (T2); 10/7, 15/7 (H); G 1.52545, null cone 1.52721; κ = 1.752, φ₀ = 92.6°, I2/I1 ≈ 2·10⁻³ | Eqs. 22–23 | `test_gl` |
| 18 point nodes of the G ground state, 12 on the five-fold axes | Section 6.3 | `test_gl` |
| Table 7: all (K, χ) with one-dimensional fixed space per channel, their R (1.5273, 1.6056, 2.2028, 2.4378, 2.4476, 2.2909, …) and TR | Table 7 | `test_gl` |
| Table 8: all seven rows to four decimals; Im C = ∓0.6186 for the two cyclic chiralities | Table 8 | `test_gl` |
| D12: 24 elements, 9 irreps; m mod 12 assignment; enforced nodes; Sym² E_m; circle ratios 3/2 and 1 | Section 10 ("Appendix 10") | `test_d12` |

## 2. Comparison with the legacy scripts

The legacy chain was run once in a scratch directory (unchanged scripts, only the hard-coded
output path of `ih_basis.py` redirected) to obtain the baseline printed values.  Runtimes:

| legacy script | time | package equivalent | time |
|---|---|---|---|
| `ih_basis.py 6` | 104 s | `groups` + `harmonics` (ℓ ≤ 6, incl. isotypic bases) | 12 s |
| `ih_seeds.py` | 77 s | `harmonics.paper_basis_functions`, `invariant_P6`, `hexad_identity` | (included) |
| `gl_inv.py` | 130 s | `gl.channel`, `gl.sym2_*` | 3 s |
| `gl_inv2.py` | 123 s | `gl.quartic_invariants` | 2 s |
| `gl_inv3.py` (optional, slow) | stopped after 34 min | `gl.relations` | < 1 s |
| `gl_H.py` | 79 s | `gl.h_channel` | 5 s |
| `gl_min.py` | 3 s | `gl.minimise_ratio` | 1 s |
| `isotropy.py` | 6 s | `gl.isotropy_table` | 5 s |
| `gl_final.py` | 35 s | `gl.g_stratum`, `gl.g_nodes`, `gl.h_candidates` | 2 s |
| `shells.py` | 62 s | `shells` | 1 s |
| `d12.py` | 1 s | `d12` | < 1 s |
| `gl_states.py` | 11 s | `gl.g_ground_state`, `gl.h_candidates` | 2 s |

`beyond32 all` (everything, ℓ ≤ 6): **22–24 s** on an idle Apple-silicon laptop (Python 3.12,
sympy 1.14); `pytest` (107 tests, including the six slow end-to-end cross-checks): 60 s, and
`pytest -m "not slow"`: 31 s, on the same idle machine.  Under heavy load (several concurrent
runs) these figures grow several-fold; the timings recorded in `results.json`
(`package.runtime_*`) are those of the run that produced it.

Baseline values printed by the legacy scripts and reproduced by the package (same digits):
Table 1–3 and the bases (`ih_basis`, `ih_seeds`); the sphere norms and Sym² decompositions
(`gl_inv`); the harmonic content ℓ ∈ {0,2}, {0,2,4,6}, {0,2,4,6}, {0,2,4} (`gl_inv2`);
λ = (−7/20 + 47√5/30)² = 9.9425…, C_R 58 terms, C_I 46 terms, six invariants independent
(`gl_H`); min R = 1.200000, 1.572960, 1.525451, 1.428571 with |η·η|² = 2.163·10⁻³ at the G
minimum (`gl_min`); the complete isotropy table with R_wc, |η·η|², min|Δ| and moduli
(`isotropy`); R(κ, φ), κ = 1.7523, phase 1.6158 rad, R(u) = 21(75u² + 170u + 687)/(715(u+3)²),
u* = 108/35, R(u*) = 15505/10153, the 18 node centres (`gl_final`); the four I_h shell
decompositions (`shells`); the D12 assignment and node conditions (`d12`); the G ground
state, its stabiliser {E, 2 C3} and time-reversal × {3 C2} stabiliser, the null-cone
minimum 1.527212900 and the H candidate rows (`gl_states`).

## 3. Notes (no mathematical change)

* **Speed.** The legacy code re-recognised every number with `nsimplify(…, [sqrt(5)])`.  The
  package keeps the numbers in sympy's algebraic field QQ<√5> (`DomainMatrix`) and, for the
  quartic invariants, works with the exact coefficient matrices of Δ² instead of expanding
  symbolic products; the mathematics is the same, and a slow test (`-m slow`) checks the
  form matrices against the direct symbolic expansion used by `gl_inv2.py`.
* **Fields.** The orthonormal basis functions of T2 and G carry √7 and the fourth H function
  √3; products of two basis functions are in Q(√5) except in the H channel, where √15 survives
  in individual coefficients of the quartic forms (their linear relations are rational).  The
  H-channel forms are therefore stored over Q(√3, √5) — still exact.
* **Random seeds.** The intertwiner seed matrix uses the `random.seed(3)` draw sequence of
  `gl_H.py` (from a private `random.Random(3)`, so the global generator is not reseeded), so λ
  is reproduced digit by digit; the restarts use `numpy` seeds 0 and 1 as in `gl_min.py` /
  `gl_states.py`, with one `RandomState(0)` per channel (`gl_min.py` drew the four channels
  from a single continuing stream) and the analytic gradient of R (`gl_min.py`: finite
  differences).  The individual starting points therefore differ from the baseline; the four
  minima agree with it to 10⁻⁶ (|η·η|² = 2.163·10⁻³ identically), while the fraction of
  restarts that land on the G minimum differs slightly (0.68 vs 0.72).
* **G nodes.** `gl.g_nodes()` searches the nodes at η = (1, 1, 1, iκ) with κ = √(108/35) =
  1.7566, the exact stationary point of the φ = π/2 line (R = 15505/10153), exactly as
  `gl_final.py` did, not at the global minimum κ = 1.7523, φ₀ = 92.6°; both states have the
  same 18 point nodes with 12 on five-fold axes.
* **Not ported.** The older 5×5 construction of J in `gl_inv3.py` (`random.seed(1)`), which
  `gl_H.py` superseded; it does not enter the paper.
* **Table 7.** The legacy `isotropy.py` lists every (K, χ) with a one-dimensional fixed space,
  including non-maximal K (e.g. the polar T1 state appears under C5 χ0, C3 χ0 and D5 A2, D3
  A2).  `gl.isotropy_table()` reproduces that list; `gl.symmetry_fixed_states()` keeps the
  entries whose stabiliser is K itself, which is exactly Table 7.  The legacy "TRS?" column
  tests whether η* is a rotation image of η (true for the chiral axial states via the
  perpendicular C2); the paper's TR column is I2 = I1.  Both flags are kept
  (`time_reversal`, `tr_up_to_rotation`).
* **Table 7 ratios.** The symmetry-fixed G states have rational weak-coupling ratios
  (315/143 for xyz, 1743/715 for g_x, 350/143 for D3 A1, 126/55 for D3 A2; chiral 1148/715 for
  C5 and 84/55 for C3).  For every state fixed by a real (±1) character the fixed space is
  recomputed exactly (null space over Q(√3, √5) of the exact irrep matrices,
  `gl.fixed_space_exact`) and R is evaluated in exact arithmetic (`gl.weak_coupling_ratio_exact`,
  `results.json` key `R_exact`); `tab_states.tex` prints these fractions, and 1687/715 (the
  paper's 5061/2145) for the polar T2 states.  For the chiral C_n states (complex characters)
  the fraction is recognised from the numerical value (`Fraction.limit_denominator(5000)`,
  tolerance 10⁻⁹) and checked to double precision; 84/55 for (1, ω, ω², 0) is also verified
  exactly.  The G ground state of Eq. (23) is printed as 1.5255.
* **Fractions** are printed reduced: the paper's 5061/2145 (real T2 states) is 1687/715.
* **Table 8** is computed with the legacy quadrature for the coordinates of the candidate
  functions (300 × 600 grid), so the digits agree with the paper.
* **I_h class labels.** The improper classes −C5 and −C5² are the rotoreflections S10⁷ ~ S10³
  and S10⁹ ~ S10; an early draft of `groups.py` had the two labels swapped, corrected before
  release.  No number depends on the labels.
* **Extras beyond the legacy scripts** (all checked): the Molien closed form derived from the
  60 matrices (`molien.check_molien_closed_form`), Γ ↓ C_n multiplicities
  (`restrictions.restrict_to_cyclic`), the I-decompositions and Frobenius cross-check of the
  shells, the second cyclic chirality in Table 8, the D12 Sym² and circle ratios, and the
  closed form √λ = −7/20 + 47√5/30 obtained by denesting (`sqrtdenest`) and verified exactly.
  The numbers of terms of the invariants N_L printed by `gl_inv2.py` (T1 9/12; T2 9/12/12/12;
  G 16/21/28/28; H 25/52/53 for N0/N2/…) are reproduced (`results.json` `gl.N_terms`).
* **Naming.** The D12 material is Section 10 of the paper, cross-referenced there as
  "Appendix 10"; the legacy README calls it "Appendix B".  The planning document
  `REFACTOR_PLAN.md` (kept in the git history up to the release commit) was written before the
  port; several function names it proposes were changed during
  implementation (e.g. `harmonics.rep_matrix` → `rep_matrices`, `gl.invariants_NL` →
  `quartic_invariants`, `gl.g_ground_state_exact` → `g_stratum`), and the D12 character
  extraction ended up exact rather than a least-squares fit; README.md and this report describe
  the final code.
* **results.json** is reproducible except for `package.generated` and the `package.runtime_*`
  timings.

## 4. Discrepancies

None.  Every value listed in the task statement and every value printed by the legacy chain
is reproduced.
