"""
Verify two figures the thesis reports against what the data actually holds.
Run from the same directory as app.py and KiranaPasal.xlsx:

    python check_numbers.py
"""

import kirana_analysis as ka

df = ka.build_targets(
    ka.load_raw("KiranaPasal.xlsx", ka.PREFERRED_SHEET),
    unresolved_as_failure=False,
)

# 1. Belief-target denominator.
#    Thesis: "answered for every record", 101 of 400 = 25.25%.
answered = int(df["target_belief"].notna().sum())
flagged = int(df["target_belief"].sum())
print(f"records                : {len(df)}")
print(f"belief answered        : {answered}  ({len(df) - answered} blank)")
print(f"belief flagged         : {flagged}")
print(f"  over answered        : {flagged / answered:.2%}")
print(f"  over all records     : {flagged / len(df):.2%}")
if answered != len(df):
    print("  >>> THESIS SAYS 'answered for every record' — THAT IS WRONG.")
    print(f"  >>> Correct figure is {flagged}/{answered} = {flagged / answered:.2%}")

print()

# 2. Agreement base.
#    Thesis: 69.8% agreement, phi = 0.041, 32 of 46 failures unanticipated.
ag = ka.agreement(df)
print(f"agreement base n       : {ag['n']}  ({ag['n_dropped']} dropped)")
print(f"agreement rate         : {ag['agreement_rate']:.1%}   (thesis: 69.8%)")
print(f"phi                    : {ag['phi']:.3f}       (thesis: 0.041)")
print(f"failures unanticipated : {ag['failed_unanticipated']} of {ag['n_failures']}"
      f"   (thesis: 32 of 46)")
if ag["n_dropped"]:
    print("  >>> State the base explicitly in the thesis: 'on the n shops with both")
    print("  >>> a resolved outcome and an answered expectation'.")

print()

# 3. Encoded width under the reported specification.
#    Thesis: 266 encoded columns against 270 training rows.
X, _, _ = ka.build_predictors(df)
out = ka.run_models(X, df["target_observed"], n_repeats=1, merge_rare=False)
print(f"predictor variables    : {X.shape[1]}      (thesis: 88)")
print(f"encoded columns        : {out['n_encoded_cols']}     (thesis: 266)")
print(f"training rows          : {out['n_train']}     (thesis: 270)")
print(f"test failures          : {out['test_events']} of {out['n_test']}"
      f"   (thesis: 14 of 117)")
for name, r in out["results"].items():
    print(f"  {name:<20} test AUC {r['test_auc']:.3f}")