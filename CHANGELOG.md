# Changelog

## v1.0.0 (2026-09-05)

First public release, accompanying the paper *Beyond the 32: Superconducting Pairing
Channels of the Icosahedral Group* (Eva Moss, 2026).

* Exact group theory over Q(sqrt5): the binary icosahedral group 2I (120 quaternions),
  I (60 rotation matrices), I_h, conjugacy classes, character tables (Tables 1-2).
* Action on harmonic polynomials, isotypic projectors, branching SO(3) -> I (Table 3),
  phi-form basis functions (Eqs. 4-9), the l = 6 invariant P6 and the hexad identity
  (Eqs. 10-11).
* Molien series, m_A(l) for l <= 30 and the S^3/2I spectrum (Eqs. 15-16).
* Restrictions to T, D5, D3, D2 (Table 5) and to the cyclic groups (Section 7).
* Permutation representations of I and I_h on the 12-, 20-, 30-, 60-orbits (Table 4).
* SU(2) -> 2I branching and pair decompositions (Eqs. 12-13).
* Ginzburg-Landau theory: orthonormal irrep matrices, Sym^2 decompositions (Table 6),
  quartic invariants and their relations (Eqs. 18-22), the H-channel intertwiner and cross
  terms, weak-coupling minima, the G ground state (Eq. 23), symmetry-fixed states (Table 7)
  and the null-cone H candidates (Table 8).
* The dodecagonal group D12 (Section 10 of the paper, its "Appendix 10").
* `beyond32 all` regenerates `results.json` and the LaTeX fragments in `tables/` (every
  numerical table, Tables 1-8 and Section 10, and every displayed result, Eqs. 4-24);
  `pytest` pins every number quoted in the paper.

The development scripts from which the package was refactored are kept unchanged under
`legacy/` (twelve scripts and their README; see `REPORT.md` for the baseline comparison).
