# 1) Executive Summary


# 2) Mathematical Equations


P′(ψ) = P(ψ) · f(E, I)

t < 0 (fluktuáció + szuperpozíció):
\tilde P_{k+1}(\psi) = P_k(\psi)\, f_k\!\big(E_k(\psi), I_k(\psi)\big),\quad k=-K,\dots,-1
P_{k+1}(\psi) = \frac{\tilde P_{k+1}(\psi)}{\int_{\Psi} \tilde P_{k+1}(\phi)\, d\phi}.

t = 0 (összeomlás, szelekció): választás a P_0 alapján.

t > 0 (expanzió + lock-in): zárt hurkokban a f_k → f_{\infty}, törvények „befagynak” (lock-in).


# 3) Stability Drivers

- Stabilitás kulcsmetrikák és I-hatás összefüggései a **stability_by_I_eps_sweep_E+I.csv** alapján.

# 4) Entropy & Distributional Analysis

- Eloszlás-leírók a stabilitás, lock-in és CMB mutatókon (lásd fent idézett fájlok); a részletes per-oszlop leíró statisztikák a mellékelt JSON snapshotban találhatók.

# 5) Seed & Reproducibility Insights

- A lock-in időskála percentilisei [metrics__lock_epoch_reg__EIX_E+I.csv → `<lock_epoch_column>`] alapján jelzik a seed-érzékenységet.

# 6) XAI (RF/XGBoost + SHAP/LIME) Findings


# 7) CMB Diagnostics (Cold Spot, low-ℓ alignment/AoE, LLAC, HPA)


# 8) Necessity of Information (I): Focused Analysis

- I=0 eset: stabilitás-mutatók átlagai alacsonyabbak [stability_by_I_zero_E+I.csv].
- Következtetés: az I komponens pozitív szerepet játszik a stabilizációban és a lock-in gyorsításában (kvantitatív részletek fent).

# 9) Limitations & Validation Plan

- Oszlopnevek heterogének; egyes metrikák indirekt névtér-relációból azonosítva (replikálható a mellékelt kóddal).
- Egyes oszlopok hiányozhatnak az adott futásban; robust aggregációt használtam.
- Következő lépés: egységes metrika-névtér és explicit definíciók (README + data dictionary).

# 10) Actionable Next Experiments

1) ε/α_I finomrácsos söprés, a legnagyobb |r| korrelációt mutató stabilitási metrikára optimalizálva [stability_by_I_eps_sweep_E+I.csv].
2) E-only vs E+I páros XAI-összevetés kontroll-spliteken (SHAP értékdisztribúció).
3) CMB diagnosztikák bővítése: paritás-aszimmetria, kvadrupólus erősség; standardizált flagképzés.
4) Lock-in időskála predikció regresszióban: kiegészítő jellemzők és seed-hatás modellezése.
5) Reproducibilitás: fixált master_seed + determinisztikus futtatás exportja (CSV + JSON manifest).