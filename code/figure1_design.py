"""Figure 1: study design and participant flow."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
try:
    from common import OUT
except Exception:
    OUT = "."

BLUE, PEACH, GREEN, PURPLE, TEAL = "#dce6f5", "#fbe0d0", "#dcecd5", "#e6dff0", "#d6ecec"
EDGE = {"#dce6f5": "#2a4c7d", "#fbe0d0": "#b4543a", "#dcecd5": "#3f6b3a",
        "#e6dff0": "#5b4a7d", "#d6ecec": "#2a6b6b"}


def box(ax, x, y, w, h, title, body, fc, ts=9.4, bs=7.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.010,rounding_size=0.016",
                                fc=fc, ec=EDGE[fc], lw=1.2, zorder=2))
    n = title.count("\n") + 1
    ax.text(x + w / 2, y + h - 0.022, title, ha="center", va="top", fontsize=ts,
            fontweight="bold", zorder=3, linespacing=1.3)
    ax.text(x + w / 2, y + h - 0.030 - 0.028 * n, body, ha="center", va="top", fontsize=bs,
            zorder=3, linespacing=1.5)


def arrow(ax, p, q, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=13, lw=1.2,
                                 color="#333333", zorder=4,
                                 connectionstyle=f"arc3,rad={rad}"))


def main():
    fig, ax = plt.subplots(figsize=(11.0, 9.8))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.text(0.012, 0.985, "Genotype data", fontsize=13, fontweight="bold", color="#2a4c7d")
    g = [("UK Biobank array", "Directly genotyped calls\n(Field 22418)\nBritish participants"),
         ("Genotype QC", "Individual\nmissingness < 0.2\nHeterozygosity\noutlier removed"),
         ("Variant QC", "MAF $\\geq$ 1%,\nmissingness $\\leq$ 5%\nHWE $P \\geq 1\\times10^{-6}$\nin controls"),
         ("Genetic risk score", "GWAS Catalog,\n$P < 5\\times10^{-8}$\n76 on array $\\rightarrow$ 74\nafter LD pruning")]
    for i, (t, b) in enumerate(g):
        box(ax, 0.012 + i * 0.249, 0.795, 0.228, 0.155, t, b, BLUE)
        if i:
            arrow(ax, (0.012 + i * 0.249 - 0.021, 0.8725), (0.012 + i * 0.249, 0.8725))

    ax.text(0.012, 0.762, "Phenotype data (baseline visit, 2006-2010)", fontsize=13,
            fontweight="bold", color="#b4543a")
    p = [("UK Biobank phenotypes", "Anthropometry, BP,\nspirometry, biochemistry,\nhaematology, lifestyle,\nfamily history"),
         ("Predictor curation", "175 inputs per analysis\nExcluded: HbA1c,\nglucose, self-rated health"),
         ("Outcome", "T2D: ICD-10 E11\n(hospital inpatient records)\nDays from baseline\nto diagnosis")]
    for i, (t, b) in enumerate(p):
        box(ax, 0.012 + i * 0.249, 0.572, 0.228, 0.155, t, b, PEACH)
        if i:
            arrow(ax, (0.012 + i * 0.249 - 0.021, 0.6495), (0.012 + i * 0.249, 0.6495))
    box(ax, 0.759, 0.572, 0.228, 0.155, "Integrated dataset",
        "Phenotypes +\ngenetic risk score\nStrata: sex $\\times$ age\n(<50 to $\\geq$65 years)", GREEN)
    arrow(ax, (0.873, 0.795), (0.873, 0.727))
    arrow(ax, (0.717, 0.6495), (0.759, 0.6495))

    ax.text(0.012, 0.524, "Analyses", fontsize=13, fontweight="bold", color="#3f6b3a")
    box(ax, 0.012, 0.040, 0.306, 0.448, "1. T2D recorded by 2010\n(case-finding; primary)",
        "10,443 cases (E11 recorded $\\leq$2010)\n10,473 controls (stratified\ndownsampling by age, sex,\nfamily history)\n\n"
        "10 sex $\\times$ age strata\n5 algorithms (LightGBM, LR,\nRF, ET, DT)\nRepeated stratified 5-fold CV\n(5 $\\times$ 5)\n"
        "LightGBM with vs without\nthe genetic risk score", GREEN)
    box(ax, 0.347, 0.040, 0.306, 0.448, "2. External evaluation\n(NHANES; primary)",
        "NHANES 2017-March 2020\n4,757 participants aged 40-69\n\n"
        "Harmonised 36-input model\napplied without modification\n\n"
        "Two outcomes in one survey:\ndiagnosed diabetes, and\nundiagnosed diabetes in the\nfasting subsample\n\n"
        "Survey-weighted AUROC,\ncalibration, decision curves,\nper-predictor contrast", TEAL)
    box(ax, 0.682, 0.040, 0.306, 0.448, "3. Incident T2D\n(secondary)",
        "15,813 case records diagnosed\nafter baseline (median 7.5 y)\n9,857 control records after\nexcluding observed baseline\nHbA1c $\\geq$48 mmol/mol and\ncontrols matching a case\nsignature (Table S18)\n\n"
        "Sex-specific LightGBM; 5 $\\times$ 5-fold CV\nAUROC by time to diagnosis\nComparators: five-variable score,\nadapted FINDRISC, HbA1c + glucose\n\n"
        "Exploratory for group membership:\nno participant identifier in the extract", PURPLE, ts=9.4, bs=7.0)
    # distribute through a horizontal bus so that no connector crosses a panel title
    BUS = 0.530
    ax.plot([0.873, 0.873], [0.566, BUS], color="#333333", lw=1.2, zorder=4,
            solid_capstyle="butt")
    ax.plot([0.165, 0.873], [BUS, BUS], color="#333333", lw=1.2, zorder=4,
            solid_capstyle="butt")
    for xt in (0.165, 0.500, 0.835):
        arrow(ax, (xt, BUS), (xt, 0.492))
    p_ = os.path.join(OUT, "Figure1.png")
    fig.savefig(p_, dpi=400, bbox_inches="tight")
    print("wrote", p_)


if __name__ == "__main__":
    main()
