"""
Insight engine for the Uber CX dashboard.

1) `generate()` - a deterministic "analyst" that reads the filtered responses and
   writes highlights, lowlights, recommendations and a key-driver analysis.
   No API key needed; it always runs and updates with every filter change.

2) `ai_narrative()` - optional. Sends an aggregated summary (no personal data)
   to Claude and returns an executive narrative. Used only when an
   ANTHROPIC_API_KEY is configured.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

GOOD = ["Good", "Excellent"]
BAD = ["Very poor", "Poor"]


@dataclass
class Finding:
    title: str
    text: str
    evidence: str = ""
    weight: float = 0.0          # used to rank findings
    tag: str = ""                # short label shown as a chip
    value: float | None = None   # headline number visualised as a ring
    unit: str = "%"              # %, /5, /10, /100, pts, ρ


@dataclass
class Recommendation:
    action: str
    why: str
    priority: str = "Medium"     # High / Medium / Quick win
    owner: str = ""


@dataclass
class Report:
    headline: str
    confidence: str
    highlights: list[Finding] = field(default_factory=list)
    lowlights: list[Finding] = field(default_factory=list)
    watch: list[Finding] = field(default_factory=list)
    recommendations: list[Recommendation] = field(default_factory=list)
    drivers: pd.DataFrame | None = None


# ---------------------------------------------------------------------------
def _share(s: pd.Series, values) -> float:
    s = s.dropna().astype(object)
    return float(s.isin(values).mean() * 100) if len(s) else float("nan")


def _n(s: pd.Series) -> int:
    return int(s.dropna().shape[0])


ASPECT_ACTIONS = {
    "r_support": ("Set a first-response SLA for in-app support (e.g. under 2 hours) and give agents authority to "
                  "refund or credit on first contact.", "Customer Support Ops"),
    "r_driver": ("Refresh driver service-standards training and surface low courtesy ratings to drivers "
                 "weekly with coaching tips.", "Driver Operations"),
    "r_punctuality": ("Tighten ETA accuracy and route guidance; flag repeat late-arrival drivers for review.",
                      "Marketplace / Driver Ops"),
    "r_app": ("Prioritise app stability fixes and run a usability review of the help & refund flows.",
              "Product & Engineering"),
    "r_overall": ("Run a service-recovery programme that follows up every unresolved complaint within 48 h.",
                  "CX Leadership"),
}
PROBLEM_ACTIONS = {
    "Poor communication with the driver": "Add pre-set in-app messages / translation for rider-driver chat and "
                                          "coach drivers on pickup communication.",
    "Driver punctuality or route knowledge": "Audit ETA accuracy in Irish cities and enforce navigation use for "
                                             "drivers with repeat route complaints.",
    "Driver behaviour or professionalism": "Introduce a visible professionalism standard with consequences "
                                           "for repeat low ratings.",
    "Unsafe or uncomfortable driving": "Escalate safety-related reports to a dedicated fast-track queue.",
    "Difficulty contacting Uber customer support": "Make 'contact a person' reachable in two taps from the trip "
                                                   "screen and add a callback option.",
    "Slow or ineffective problem resolution": "Track first-contact resolution and reopen rate as core support KPIs.",
    "Payment, fare or refund issues": "Auto-refund clear fare anomalies and show an itemised fare breakdown.",
}
THEME_ACTIONS = {
    "Pricing & fares": "Communicate price transparency (upfront fare, surge explanation) and test a "
                       "lower-cost ride tier.",
    "Driver courtesy": "Add courtesy to the driver rating breakdown so riders can flag it specifically.",
    "Faster problem resolution": "Publish expected resolution times inside the help flow and meet them.",
    "App reliability": "Monitor crash-free sessions as a release gate.",
    "Availability & coverage": "Review supply in under-served areas (rural / off-peak) with targeted driver incentives.",
}


def _key_drivers(df: pd.DataFrame, aspects: dict[str, str]) -> pd.DataFrame:
    """Importance = Spearman correlation of each touchpoint with Overall experience
    and with Q7 loyalty; Performance = average rating. Classic CX key-driver matrix."""
    rows = []
    for col, label in aspects.items():
        if col == "r_overall":
            continue
        x = df[col + "_n"].astype(float)
        imp_overall = x.corr(df["r_overall_n"].astype(float), method="spearman")
        imp_loyal = x.corr(df["cx_loyalty_n"].astype(float), method="spearman")
        rows.append({
            "aspect_key": col, "aspect": label,
            "performance": x.mean(),
            "importance": 0.0 if pd.isna(imp_overall) else imp_overall,
            "loyalty_link": 0.0 if pd.isna(imp_loyal) else imp_loyal,
            "t2b": _share(df[col], GOOD), "b2b": _share(df[col], BAD), "n": _n(x),
        })
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    imp_mid, perf_mid = d["importance"].median(), d["performance"].median()

    def quad(r):
        hi_imp, hi_perf = r["importance"] >= imp_mid, r["performance"] >= perf_mid
        if hi_imp and not hi_perf:
            return "Fix first"
        if hi_imp and hi_perf:
            return "Protect"
        if not hi_imp and not hi_perf:
            return "Monitor"
        return "Maintain"

    d["quadrant"] = d.apply(quad, axis=1)
    return d


def _segment_extremes(df: pd.DataFrame, segments: dict[str, str], min_n: int = 3):
    overall = df["cx_index"].mean()
    out = []
    for label, col in segments.items():
        if col in ("had_problem", "competitor_better"):
            continue
        g = df.groupby(df[col].astype(object))["cx_index"].agg(["mean", "count"])
        g = g[g["count"] >= min_n]
        for name, r in g.iterrows():
            out.append((label, name, r["mean"], int(r["count"]), r["mean"] - overall))
    return sorted(out, key=lambda t: t[4])


# ---------------------------------------------------------------------------
def generate(df: pd.DataFrame, aspects: dict[str, str], segments: dict[str, str]) -> Report:
    n = len(df)
    conf = ("Directional — fewer than 30 responses; treat as early signals." if n < 30 else
            "Moderate — sample of 30–99 responses." if n < 100 else "Solid sample (100+ responses).")
    H, L, W, R = [], [], [], []

    cx = df["cx_index"].mean()
    drivers = _key_drivers(df, aspects)
    perf = pd.DataFrame({
        "key": list(aspects), "label": list(aspects.values()),
        "avg": [df[c + "_n"].mean() for c in aspects],
        "t2b": [_share(df[c], GOOD) for c in aspects],
        "b2b": [_share(df[c], BAD) for c in aspects],
    }).sort_values("avg")
    best, worst = perf.iloc[-1], perf.iloc[0]

    # --- Touchpoints -------------------------------------------------------
    H.append(Finding(
        f"{best['label']} is the strongest touchpoint",
        f"It averages {best['avg']:.2f}/5 and {best['t2b']:.0f}% rate it Good or Excellent — "
        f"the clearest asset in the rider experience.",
        f"Q5 · n={_n(df[best['key']])}", best["t2b"], "Strength", value=best["avg"], unit="/5"))
    L.append(Finding(
        f"{worst['label']} is the weakest touchpoint",
        f"It averages only {worst['avg']:.2f}/5; {worst['b2b']:.0f}% rate it Poor or Very poor "
        f"and just {worst['t2b']:.0f}% rate it Good+.",
        f"Q5 · n={_n(df[worst['key']])}", 60 + worst["b2b"], "Gap", value=worst["avg"], unit="/5"))

    # --- Loyalty link -----------------------------------------------------
    loyal = _share(df["cx_loyalty"], ["Likely", "Very likely"])
    if loyal >= 50:
        H.append(Finding(
            "Good service converts into loyalty",
            f"{loyal:.0f}% say a positive customer-service experience makes them more likely to ride again "
            f"or recommend Uber — service quality is a direct growth lever.",
            f"Q7 · n={_n(df['cx_loyalty'])}", loyal, "Loyalty", value=loyal))
    else:
        L.append(Finding(
            "Service quality isn't translating into loyalty",
            f"Only {loyal:.0f}% say good service makes them more likely to ride again or recommend.",
            f"Q7 · n={_n(df['cx_loyalty'])}", 70 - loyal / 2, "Loyalty", value=loyal))

    # --- Trust erosion ----------------------------------------------------
    trust = df["trust_drop"].mean()
    high_trust = _share(df["trust_band"], ["High (8–10)"])
    if trust >= 6:
        L.append(Finding(
            "Unresolved problems destroy trust",
            f"Average trust drop is {trust:.1f}/10 when an issue isn't fixed; {high_trust:.0f}% put it at 8–10. "
            f"Every failed resolution puts the relationship at risk.",
            f"Q9 · n={_n(df['trust_drop'])}", 50 + high_trust, "Trust", value=trust, unit="/10"))
    else:
        H.append(Finding("Trust is relatively resilient",
                         f"Average trust drop is a moderate {trust:.1f}/10 when issues go unresolved.",
                         f"Q9 · n={_n(df['trust_drop'])}", 40, "Trust", value=trust, unit="/10"))

    # --- Poor driver behaviour -------------------------------------------
    dis = _share(df["poor_driver_sat"], ["Very dissatisfied", "Dissatisfied"])
    if dis >= 40:
        L.append(Finding(
            "Arriving isn't enough — driver conduct defines the trip",
            f"{dis:.0f}% are dissatisfied when a driver behaves poorly even if they reach their destination. "
            f"Riders judge the human interaction, not just the ride.",
            f"Q8 · n={_n(df['poor_driver_sat'])}", 30 + dis / 2, "Drivers", value=dis))

    # --- Support responsiveness -------------------------------------------
    prompt = _share(df["support_prompt"], ["Often", "Always"])
    rarely = _share(df["support_prompt"], ["Never", "Rarely"])
    if rarely >= 30:
        L.append(Finding(
            "Support is seen as slow or unhelpful",
            f"{rarely:.0f}% say support Never/Rarely responds promptly and helpfully, vs {prompt:.0f}% Often/Always.",
            f"Q6 · n={_n(df['support_prompt'])}", 40 + rarely / 2, "Support", value=rarely))
    elif prompt >= 40:
        H.append(Finding("Support responsiveness holds up",
                         f"{prompt:.0f}% say support Often or Always responds promptly and helpfully.",
                         f"Q6 · n={_n(df['support_prompt'])}", prompt, "Support", value=prompt))

    # --- Problems ---------------------------------------------------------
    inc = _share(df["had_problem"], ["Yes"])
    affected = int((df["problem_count"] > 0).sum())
    probs = df["problem_list"].explode().dropna()
    if len(probs):
        top, top_n = probs.value_counts().index[0], int(probs.value_counts().iloc[0])
        L.append(Finding(
            f"Top pain point: {top.lower()}",
            f"{inc:.0f}% of riders have had a problem. Among the {affected} who listed issues, "
            f"{top_n} cite {top.lower()}.",
            "Q4 / Q4b", 35 + inc / 2, "Problems", value=inc))
    yes, no = df[df["had_problem"] == "Yes"], df[df["had_problem"] == "No"]
    if len(yes) >= 3 and len(no) >= 3:
        gap = no["cx_index"].mean() - yes["cx_index"].mean()
        loy_yes = _share(yes["cx_loyalty"], ["Likely", "Very likely"])
        if gap >= 5:
            L.append(Finding("A single problem drags the whole experience down",
                             f"Riders who hit a problem score a CX Index {gap:.0f} pts lower "
                             f"({yes['cx_index'].mean():.0f} vs {no['cx_index'].mean():.0f}).",
                             f"Q4 × Q5 · n={len(yes)} vs {len(no)}", 40 + gap, "Recovery", value=gap, unit="pts"))
        elif loy_yes >= 50:
            H.append(Finding("Riders with problems are still willing to come back",
                             f"{loy_yes:.0f}% of riders who experienced a problem would still reuse or recommend "
                             f"Uber after good service — recovery works when it happens.",
                             f"Q4 × Q7 · n={len(yes)}", loy_yes / 1.5, "Recovery", value=loy_yes))

    # --- Competition ------------------------------------------------------
    comp = _share(df["competitor_better"], ["Yes"])
    named = df["competitor_clean"].dropna()
    named = named[~named.str.startswith("Comment")]
    if comp >= 30:
        lead = f" {named.value_counts().index[0]} is named most often." if len(named) else ""
        L.append(Finding("Competitors are seen as offering better service",
                         f"{comp:.0f}% believe another taxi or ride-hailing service does customer service better.{lead}",
                         f"Q10 · n={_n(df['competitor_better'])}", 30 + comp, "Competition", value=comp))
    else:
        H.append(Finding("Uber holds its ground against competitors",
                         f"{100 - comp:.0f}% do not think a rival offers better customer service.",
                         f"Q10 · n={_n(df['competitor_better'])}", 100 - comp, "Competition", value=100 - comp))

    # --- Voice of customer --------------------------------------------------
    th = df["themes"].explode().dropna()
    th_real = th[~th.isin(["Already satisfied", "Other"])]
    if len(th_real):
        top_theme, t_n = th_real.value_counts().index[0], int(th_real.value_counts().iloc[0])
        W.append(Finding(f"Riders' #1 ask: {top_theme.lower()}",
                         f"{t_n} of {int(df['improvement'].notna().sum())} open-ended suggestions mention it.",
                         "Q11", t_n, "Voice of customer",
                         value=t_n / max(int(df["improvement"].notna().sum()), 1) * 100))
    happy = int((th == "Already satisfied").sum())
    if happy:
        H.append(Finding("Some riders are advocates already",
                         f"{happy} rider(s) volunteered that Uber is already doing well.",
                         "Q11", happy * 5, "Advocacy",
                         value=happy / max(int(df["improvement"].notna().sum()), 1) * 100))

    # --- Segments ----------------------------------------------------------
    ext = _segment_extremes(df, segments)
    if ext:
        lo, hi = ext[0], ext[-1]
        if lo[4] <= -6:
            W.append(Finding(f"Least satisfied segment: {lo[1]} ({lo[0].lower()})",
                             f"CX Index {lo[2]:.0f} vs {cx:.0f} overall ({lo[4]:+.0f} pts, n={lo[3]}).",
                             "Segment analysis", abs(lo[4]), "Segment", value=lo[2], unit="/100"))
        if hi[4] >= 6:
            H.append(Finding(f"Happiest segment: {hi[1]} ({hi[0].lower()})",
                             f"CX Index {hi[2]:.0f} vs {cx:.0f} overall ({hi[4]:+.0f} pts, n={hi[3]}).",
                             "Segment analysis", hi[4], "Segment", value=hi[2], unit="/100"))

    # --- Key-driver insight ------------------------------------------------
    if drivers is not None and not drivers.empty:
        fix = drivers[drivers["quadrant"] == "Fix first"].sort_values("importance", ascending=False)
        top_imp = drivers.sort_values("importance", ascending=False).iloc[0]
        W.append(Finding(f"Biggest lever on overall satisfaction: {top_imp['aspect'].lower()}",
                         f"It moves most closely with Overall experience (ρ = {top_imp['importance']:.2f}); "
                         f"improving it should lift overall scores most.",
                         "Key-driver analysis", abs(top_imp["importance"]) * 50, "Driver",
                         value=top_imp["importance"], unit="ρ"))

    # --- Recommendations --------------------------------------------------
    used = set()
    if drivers is not None and not drivers.empty:
        for _, r in drivers[drivers["quadrant"] == "Fix first"].iterrows():
            act, owner = ASPECT_ACTIONS[r["aspect_key"]]
            R.append(Recommendation(act, f"{r['aspect']}: high impact on overall satisfaction "
                                         f"(ρ={r['importance']:.2f}) but below-median rating ({r['performance']:.2f}/5).",
                                    "High", owner))
            used.add(r["aspect_key"])
    if worst["key"] not in used:
        act, owner = ASPECT_ACTIONS[worst["key"]]
        R.append(Recommendation(act, f"{worst['label']} is the lowest-rated touchpoint ({worst['avg']:.2f}/5).",
                                "High", owner))
    if trust >= 6:
        R.append(Recommendation(
            "Close the loop on every complaint: confirm the fix, apologise, and offer a goodwill credit where "
            "the issue was Uber's fault.",
            f"Trust drops {trust:.1f}/10 on average when problems go unresolved.", "High", "Customer Support Ops"))
    if len(probs):
        for p in probs.value_counts().index[:2]:
            if p in PROBLEM_ACTIONS:
                R.append(Recommendation(PROBLEM_ACTIONS[p],
                                        f"'{p}' is among the most-reported problems.", "Medium",
                                        "Driver Ops" if "driver" in p.lower() else "Customer Support Ops"))
    if len(th_real):
        for t in th_real.value_counts().index[:2]:
            if t in THEME_ACTIONS:
                R.append(Recommendation(THEME_ACTIONS[t], f"'{t}' is a recurring rider suggestion (Q11).",
                                        "Quick win" if t in ("App reliability", "Driver courtesy") else "Medium",
                                        "Pricing / Marketing" if "Pric" in t else "Product & Ops"))
    if comp >= 30 and len(named):
        R.append(Recommendation(
            f"Benchmark the support journey of {named.value_counts().index[0]} (mystery-shop complaint handling).",
            f"{comp:.0f}% think a competitor offers better service.", "Medium", "Strategy / CX"))
    # de-duplicate while keeping order
    seen, recs = set(), []
    for r in R:
        if r.action not in seen:
            recs.append(r)
            seen.add(r.action)

    # --- Headline ---------------------------------------------------------
    verdict = ("strong" if cx >= 70 else "solid" if cx >= 60 else "middling" if cx >= 45 else "weak")
    headline = (f"Riders rate Uber's service experience as {verdict} (CX Index {cx:.0f}/100). "
                f"{best['label']} is the standout, but {worst['label'].lower()} lags behind"
                + (f", and with trust dropping {trust:.1f}/10 after an unresolved problem, "
                   f"service recovery is the biggest risk." if trust >= 6 else ".")
                + (f" {comp:.0f}% already see a rival as better." if comp >= 30 else ""))

    return Report(
        headline=headline, confidence=conf,
        highlights=sorted(H, key=lambda f: -f.weight),
        lowlights=sorted(L, key=lambda f: -f.weight),
        watch=sorted(W, key=lambda f: -f.weight),
        recommendations=recs[:7],
        drivers=drivers,
    )


# ---------------------------------------------------------------------------
# Optional: Claude-written narrative
# ---------------------------------------------------------------------------
def build_payload(df: pd.DataFrame, aspects: dict[str, str], questions: dict[str, str]) -> dict:
    """Aggregated, anonymous summary of the filtered survey for the LLM."""
    def dist(col):
        return {str(k): int(v) for k, v in df[col].astype(object).value_counts().items()}

    payload = {
        "n_responses": len(df),
        "cx_index_0_100": round(float(df["cx_index"].mean()), 1),
        "profile": {q: dist(c) for c, q in [("age", "age"), ("residency", "residency"), ("usage", "usage")]},
        "q4_had_problem": dist("had_problem"),
        "q4b_problem_types": {str(k): int(v) for k, v in df["problem_list"].explode().dropna().value_counts().items()},
        "q5_ratings": {label: {"distribution": dist(c), "avg_1_5": round(float(df[c + "_n"].mean()), 2)}
                       for c, label in aspects.items()},
        "q6_support_prompt": dist("support_prompt"),
        "q7_cx_drives_loyalty": dist("cx_loyalty"),
        "q8_satisfaction_when_driver_poor": dist("poor_driver_sat"),
        "q9_trust_drop_1_10": {"mean": round(float(df["trust_drop"].mean()), 2),
                               "distribution": {str(int(k)): int(v) for k, v in
                                                df["trust_drop"].value_counts().sort_index().items()}},
        "q10_competitor_better": dist("competitor_better"),
        "q10b_competitors_named": {str(k): int(v) for k, v in df["competitor_clean"].dropna().value_counts().items()},
        "q11_comments": [c for c in df["improvement"].dropna().astype(str).str.strip()
                         if c.lower() not in ("n/a", "na", "-")],
        "question_wording": questions,
    }
    return payload


SYSTEM_PROMPT = (
    "You are a senior customer-experience analyst. You interpret survey results for an executive audience. "
    "Be specific and quantitative, cite the numbers you use, never invent data, and flag when the sample is "
    "too small to be conclusive. Write in British English."
)

USER_PROMPT = """Below is an aggregated summary of a rider survey on how customer service affects satisfaction with Uber in Ireland.

Write the analysis in Markdown with exactly these sections:
### Executive verdict
(2–3 sentences)
### Highlights
(3–5 bullets — what is working, each with the supporting number)
### Lowlights
(3–5 bullets — what is hurting the experience, each with the supporting number)
### What is driving satisfaction
(2–4 bullets connecting the questions, e.g. problems → trust → loyalty)
### Recommendations
(4–6 prioritised actions, each tagged [High] / [Medium] / [Quick win])
### Caveats
(1–2 bullets on sample size or survey design)

DATA (JSON):
{data}
"""


def ai_narrative(payload: dict, api_key: str, model: str) -> str:
    import anthropic  # imported lazily so the dashboard runs without it

    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=model,
        max_tokens=1800,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": USER_PROMPT.format(data=json.dumps(payload, ensure_ascii=False))}],
    )
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
