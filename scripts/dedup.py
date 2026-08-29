"""
QGene - Assembly deduplication.

ClinVar's variant_summary carries one row per variant PER GENOME ASSEMBLY.
The original download kept GRCh37, GRCh38 and 'na' rows, so the same variant
appears up to three times. A stratified split therefore leaks variants from
train into test.

This script collapses to one row per (Name, GeneSymbol), preferring
GRCh38 > GRCh37 > na, and drops any variant whose duplicate rows disagree
on Label.

Usage:  python scripts/dedup.py
"""

import os
import sys
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(BASE, "data", "brca_mutations.csv")
DST = os.path.join(BASE, "data", "brca_mutations_dedup.csv")

KEY = ["Name", "GeneSymbol"]
ASSEMBLY_RANK = {"GRCh38": 0, "GRCh37": 1, "na": 2}


def rule(line=""):
    print(("-" * 62) if not line else f"\n--- {line} " + "-" * max(0, 57 - len(line)))


print("=" * 62)
print("QGene - deduplicate ClinVar assembly-duplicated variants")
print("=" * 62)

df = pd.read_csv(SRC, low_memory=False)

rule("BEFORE")
print(f"rows                        : {len(df)}")
print(f"distinct (Name, GeneSymbol) : {df.drop_duplicates(subset=KEY).shape[0]}")
print(f"duplicated on key           : {df.duplicated(subset=KEY).sum()}")
print("\nper-assembly counts:")
print(df["Assembly"].value_counts(dropna=False).to_string())
print("\nLabel value_counts:")
print(df["Label"].value_counts(dropna=False).to_string())

# ── conflicting labels ────────────────────────────────────────────────
# A variant is conflicting if its rows carry more than one distinct Label.
rule("CONFLICTING-LABEL CHECK")
nlab = df.groupby(KEY, dropna=False)["Label"].nunique()
conflicting = nlab[nlab > 1]
n_conflict_variants = len(conflicting)
print(f"variants with >1 distinct Label across their rows : {n_conflict_variants}")

if n_conflict_variants:
    conflict_keys = set(conflicting.index)
    mask = pd.MultiIndex.from_frame(df[KEY]).isin(conflict_keys)
    n_conflict_rows = int(mask.sum())
    print(f"rows belonging to those variants                  : {n_conflict_rows}")
    print("\nexamples (up to 10 rows):")
    print(df.loc[mask, KEY + ["Assembly", "ClinicalSignificance", "Label"]]
            .sort_values(KEY).head(10).to_string(index=False))
    print("\n-> dropping these variants entirely (not picking a winner).")
    df = df.loc[~mask].copy()
else:
    n_conflict_rows = 0
    print("none found - no variants dropped for label conflict.")

# ── deduplicate ───────────────────────────────────────────────────────
rule("DEDUPLICATION")
unknown = set(df["Assembly"].dropna().unique()) - set(ASSEMBLY_RANK)
if unknown:
    print(f"NOTE: unranked Assembly values {sorted(unknown)} -> ranked last.")

df["_rank"] = df["Assembly"].map(ASSEMBLY_RANK).fillna(len(ASSEMBLY_RANK))
# stable sort on rank, then keep the first row of each key
df = df.sort_values("_rank", kind="mergesort")
dedup = df.drop_duplicates(subset=KEY, keep="first").drop(columns="_rank")
dedup = dedup.sort_index()

print(f"kept one row per (Name, GeneSymbol), preferring GRCh38 > GRCh37 > na")

rule("AFTER")
print(f"rows                        : {len(dedup)}")
print(f"distinct (Name, GeneSymbol) : {dedup.drop_duplicates(subset=KEY).shape[0]}")
print(f"duplicated on key           : {dedup.duplicated(subset=KEY).sum()}")
print("\nper-assembly counts (source assembly of the surviving row):")
print(dedup["Assembly"].value_counts(dropna=False).to_string())
print("\nLabel value_counts:")
print(dedup["Label"].value_counts(dropna=False).to_string())
lab = dedup["Label"].value_counts()
print(f"\nclass balance: pathogenic(1)={int(lab.get(1,0))} "
      f"benign(0)={int(lab.get(0,0))} "
      f"pathogenic share={lab.get(1,0)/len(dedup)*100:.2f}%")

# ── assertions ────────────────────────────────────────────────────────
assert dedup.duplicated(subset=KEY).sum() == 0, "duplicates remain on (Name, GeneSymbol)"
assert dedup["Label"].isna().sum() == 0, "null labels present"
print("\nASSERT PASS: zero duplicates remain on (Name, GeneSymbol).")

dedup.to_csv(DST, index=False)
print(f"\nwritten -> {os.path.relpath(DST, BASE)}  ({len(dedup)} rows)")
print("=" * 62)
