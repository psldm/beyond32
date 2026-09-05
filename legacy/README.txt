Beyond the 32 -- computational companion (exact sympy, Python 3.12)
Run order (each step writes a pickle used by the next):
  python3 ih_basis.py 6      # groups, character tables, branching, bases, Molien, restrictions, 2I
  python3 ih_seeds.py        # phi-form basis functions, hexad identity for the l=6 invariant
  python3 gl_inv.py          # orthogonal irrep matrices T1,T2,G,H; Sym^2 decompositions  -> gl_data.pkl
  python3 gl_inv2.py         # harmonic content of Delta^2, invariants N_L               -> gl_inv2.pkl
  python3 gl_inv3.py         # (optional, slow) relations among N_L for T1,T2,G
  python3 gl_H.py            # H channel: N4G/N4H split, intertwiner J, cross terms        -> gl_H.pkl
  python3 gl_min.py          # weak-coupling minimisation per channel
  python3 isotropy.py        # isotropy subgroups (K,chi), symmetry-fixed states table
  python3 gl_final.py        # G ground state (kappa, phase, nodes); H candidate invariants -> H_rows.pkl
  python3 shells.py          # permutation representations of I_h on the four orbit types (Table of shells)
  python3 d12.py             # dodecagonal group D12: harmonics -> irreps, enforced nodes (Appendix B)
Dependencies: sympy, numpy, scipy.
