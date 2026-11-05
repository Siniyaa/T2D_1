"""Corrected family-history merge for Cohorting-1.

The original notebook ran `study = study.merge(fam_hist)` with no key and no validation.
UK Biobank family-illness data has multiple rows per participant, so pandas performed a
many-to-many join and expanded 20,916 rows to 23,188,684. Use this instead.
"""
import pandas as pd

def attach_family_history(study: pd.DataFrame, fam_hist: pd.DataFrame,
                          key: str = "EID") -> pd.DataFrame:
    """Attach one family-history row per participant, failing loudly if that is impossible."""
    fh = fam_hist.drop_duplicates(subset=[key])
    if len(fh) != len(fam_hist):
        raise ValueError(
            f"fam_hist had {len(fam_hist):,} rows for {fh[key].nunique():,} participants; "
            "collapse it to one row per participant before merging."
        )
    out = study.merge(fh, on=key, how="left", validate="one_to_one")
    assert len(out) == len(study), f"merge changed row count: {len(study):,} -> {len(out):,}"
    return out
