# Reflection diagnostics

This folder is for diagnostics checking whether the boundary response is ordinary propagating reflection or near-roof spinor-ABC filtering/storage.

Relevant script:

```text
python-scripts/953Reflection_Diag.py
```

The diagnostic computes roof backflow, a windowed longitudinal `k_z` decomposition below the roof, tangential branch weights, and Duhamel relative-error quantities.

Suggested files:

```text
summary.json
kz_R.npy
kz_P_plus.npy
kz_P_minus.npy
near_roof_mass.npy
roof_Jnet.npy
roof_Jplus.npy
roof_Jminus.npy
Q_minus_roof.npy
E_full_depthSteps_*.npy
```

Suggested figures:

```text
kz_reflection_ratio_vs_time.png
near_roof_mass_vs_time.png
Q_minus_roof_vs_time.png
E_full_depths.png
```
