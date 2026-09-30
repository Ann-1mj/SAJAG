"""AI briefings + outreach: LLM-drafted branch field briefings and
per-borrower outreach messages for the T-0h inland results (== the final
scored portfolio - see tests/test_countdown.py for why those match exactly).

This module is the offline generator: run it once (`python -m src.briefings`,
with GROQ_API_KEY set - directly in the shell, or via a .env file in the
project root) to populate data/processed/briefings.json and outreach.json.
app.py reads only those cached files at runtime - it never calls the API -
and falls back to the plain templates defined here whenever a cache entry is
missing or was itself a fallback.
"""
import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")  # no-op if the file doesn't exist; never overrides a var already set

from src.formatting import format_inr_lakh_cr  # noqa: E402
from src.labels import OCCUPATION_LABELS  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
BRIEFINGS_PATH = PROCESSED / "briefings.json"
OUTREACH_PATH = PROCESSED / "outreach.json"
BACKTRANSLATIONS_PATH = PROCESSED / "outreach_backtranslations.json"

LANGUAGE_NAMES = {"hi": "Hindi", "or": "Odia"}

# Checks WORDING/meaning, not numbers - is_grounded() only ever validates
# numbers, so it can't catch a promise made in words. English-only by
# design: run against the back-translated English text (see
# run_backtranslation_audit), never against the raw Hindi/Odia, since a
# regex of English words can't meaningfully match non-Latin script text.
FORBIDDEN_PROMISE_RE = re.compile(r"relief|moratorium|restructur|waive|forgiv|write.?off|discount", re.IGNORECASE)

MODEL = "openai/gpt-oss-120b"  # Groq - strongest general-purpose model available on this account
MAX_ATTEMPTS = 2  # 1 initial try + 1 retry, then fall back to a template

# Literal token the LLM must use verbatim instead of inventing a name - the
# synthetic portfolio has no real borrower names.
NAME_PLACEHOLDER = "[Borrower Name]"

BRANCH_SYSTEM_PROMPT = """You are a field-operations assistant helping a rural lender's branch \
staff respond after a cyclone. Given a JSON object of facts about one branch's borrowers, \
write a concise field briefing of 4-5 lines for branch staff covering:
1. Who to call or visit first.
2. Which village(s) to prioritize.
3. What to verify when reaching a borrower (e.g. safety, livelihood damage, ability to travel).

Rules:
- Plain English, no markdown formatting, no headings, just 4-5 short lines.
- Use ONLY the facts, numbers, and place names present in the JSON. Do not invent \
any number, name, or detail that isn't there.
- When citing amounts of money, use Indian units exactly as given (lakh/crore, e.g. \
"59.71 lakh" or "0.6 crore") - never "million" or "billion".
"""

OUTREACH_SYSTEM_PROMPT = """You are a warm, respectful relationship officer at a rural lender's \
branch, reaching out to a borrower shortly after a cyclone. Given a JSON object of facts about \
one borrower, write a short outreach message (2-4 sentences) in three languages: English, Hindi, \
and Odia.

Rules (follow all of them):
- Check on the borrower's wellbeing and on their livelihood/occupation.
- Offer a visit or phone call at the borrower's convenience.
- Use the exact text given in "first_name_placeholder" wherever you would use the borrower's \
name - never invent a name.
- Do NOT mention or promise loan relief, a moratorium, restructuring, waivers, or any change to \
loan terms.
- Do NOT use any number that is not present in the JSON facts (no invented amounts, dates, \
percentages, or counts).
- If an amount of money is given, use Indian units exactly as given (lakh/crore) - never \
"million" or "billion".
- Keep the tone warm, respectful, and simple.

Respond with ONLY a JSON object with exactly these three keys: "en", "hi", "or" (English, Hindi, \
and Odia message text). No markdown code fences, no extra commentary, no other keys.
"""


def load_config():
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def load_final_portfolio():
    """T-0h inland results: identical to the final scored portfolio."""
    return pd.read_csv(PROCESSED / "scored_portfolio.csv")


def _parse_reasons(reasons_raw):
    if pd.isna(reasons_raw):
        return []
    return [r.strip() for r in str(reasons_raw).split(";") if r.strip()]


def _reason_category(reason_text):
    """Collapse a borrower-specific reason string (which embeds numbers that
    differ per borrower) into a shared category, so branch-level counting
    actually finds repeats instead of comparing unique strings."""
    lower = reason_text.lower()
    if "exposure:" in lower:
        return "High/medium exposure to the cyclone track"
    if "highly sensitive to" in lower:
        return "Livelihood highly sensitive to the hazard"
    if "missed" in lower and "emi" in lower:
        return "Already missed EMIs before the event"
    return reason_text


def build_branch_facts(df, branch_id):
    """Facts dict for one branch: urgent/watch counts, top 3 villages by
    urgent count, most common reason categories, and total exposed
    outstanding. Every number the LLM is allowed to use must come from here.

    exposed_outstanding_lakh / _cr and top_villages_urgent_total are
    pre-derived (the exposed total in lakh and crore, and the sum of the
    top-3 villages' urgent counts) because the model reliably tries to cite
    a rounded rupee figure and a village-total on its own, and without them
    present verbatim it was computing them itself and getting rejected by
    is_grounded() for using a number that wasn't technically "given". Adding
    the derived number closes that gap without loosening validation: the
    model still can't use any number besides what's explicitly here. Lakh/
    crore (not million) because that's the unit Indian financial reporting
    - and the BRANCH_SYSTEM_PROMPT - actually calls for.
    """
    branch_df = df[df["branch_id"] == branch_id]
    urgent = branch_df[branch_df["tier"] == "high"]
    watch = branch_df[branch_df["tier"] == "medium"]
    exposed = branch_df[branch_df["tier"] != "not_exposed"]

    top_villages_counts = urgent.groupby("village_id").size().sort_values(ascending=False).head(3)
    subdivision_by_village = urgent.drop_duplicates("village_id").set_index("village_id")["subdivision"]

    reason_counts = Counter()
    for reasons_raw in urgent["reasons"]:
        reason_counts.update(_reason_category(r) for r in _parse_reasons(reasons_raw))

    top_occupation_counts = urgent.groupby("occupation").size().sort_values(ascending=False).head(3)

    top_villages = [
        {
            "village_id": village_id,
            "subdivision": subdivision_by_village.get(village_id, ""),
            "urgent_count": int(count),
        }
        for village_id, count in top_villages_counts.items()
    ]
    exposed_outstanding_inr = float(exposed["outstanding"].sum())

    return {
        "branch_id": branch_id,
        "district": branch_df["district"].iloc[0] if not branch_df.empty else "",
        "urgent_count": int(len(urgent)),
        "watch_count": int(len(watch)),
        "top_villages": top_villages,
        "top_villages_urgent_total": int(sum(v["urgent_count"] for v in top_villages)),
        "top_reasons": [
            {"reason": reason, "borrower_count": int(count)}
            for reason, count in reason_counts.most_common(3)
        ],
        "top_occupations": [
            {"occupation": OCCUPATION_LABELS.get(occ, occ), "borrower_count": int(count)}
            for occ, count in top_occupation_counts.items()
        ],
        "exposed_outstanding_inr": exposed_outstanding_inr,
        "exposed_outstanding_lakh": round(exposed_outstanding_inr / 100_000, 2),
        "exposed_outstanding_cr": round(exposed_outstanding_inr / 10_000_000, 2),
    }


def build_borrower_facts(row):
    """Facts dict for one HIGH-tier borrower. No real name is available in
    the synthetic portfolio, so first_name_placeholder is a literal token
    the LLM must use verbatim rather than inventing one."""
    return {
        "borrower_id": row["borrower_id"],
        "first_name_placeholder": NAME_PLACEHOLDER,
        "occupation": OCCUPATION_LABELS.get(row["occupation"], row["occupation"]),
        "village_id": row["village_id"],
        "subdivision": row["subdivision"],
        "district": row["district"],
        "reasons": _parse_reasons(row["reasons"]),
        "missed_emis": int(row["missed_emis"]),
    }


# --- Plain-template fallbacks (no LLM, always number-grounded by construction) ---


def template_branch_briefing(facts):
    lines = [
        f"{facts['urgent_count']} urgent and {facts['watch_count']} watch borrowers "
        f"in branch {facts['branch_id']} ({facts['district']})."
    ]
    if facts["top_villages"]:
        villages = ", ".join(
            f"{v['subdivision'] or v['village_id']} ({v['urgent_count']} urgent)" for v in facts["top_villages"]
        )
        lines.append(f"Call or visit first: {villages}.")
    if facts["top_reasons"]:
        reasons = "; ".join(f"{r['reason']} ({r['borrower_count']})" for r in facts["top_reasons"])
        lines.append(f"Common reasons flagged: {reasons}.")
    if facts.get("top_occupations"):
        occupations = ", ".join(f"{o['occupation']} ({o['borrower_count']})" for o in facts["top_occupations"])
        lines.append(f"Most affected occupations: {occupations}.")
    lines.append(f"Exposed outstanding at this branch: {format_inr_lakh_cr(facts['exposed_outstanding_inr'])}.")
    lines.append("Verify borrower safety, livelihood damage, and ability to reach the branch or a field officer.")
    return "\n".join(lines)


def template_outreach_message(facts):
    name = facts["first_name_placeholder"]
    place = facts["subdivision"] or facts["village_id"]
    occupation = facts["occupation"].lower()

    en = (
        f"Dear {name}, this is your bank branch checking in after the cyclone. We hope you and "
        f"your family are safe. We'd like to know how your {occupation} work in {place} has been "
        f"affected. A field officer can visit or call at a time that suits you - just let us know. "
        f"We're here to help."
    )
    hi = (
        f"प्रिय {name}, यह चक्रवात के बाद हालचाल जानने के लिए आपकी बैंक शाखा की ओर से संपर्क है। "
        f"हमें आशा है कि आप और आपका परिवार सुरक्षित हैं। हम {place} में आपके {occupation} कार्य के बारे में "
        f"जानना चाहते हैं। आपकी सुविधानुसार एक फील्ड अधिकारी आपसे मिल सकता है या कॉल कर सकता है। "
        f"हम सहायता के लिए यहाँ हैं।"
    )
    or_ = (
        f"ପ୍ରିୟ {name}, ଏହା ଘୂର୍ଣ୍ଣିବାତ୍ୟା ପରେ ଆପଣଙ୍କ ଠାରୁ ଖବର ନେବା ପାଇଁ ଆପଣଙ୍କ ବ୍ୟାଙ୍କ ଶାଖାର ସମ୍ପର୍କ। "
        f"ଆମେ ଆଶା କରୁଛୁ ଆପଣ ଏବଂ ଆପଣଙ୍କ ପରିବାର ସୁରକ୍ଷିତ ଅଛନ୍ତି। {place} ରେ ଆପଣଙ୍କ {occupation} କାର୍ଯ୍ୟ "
        f"କେମିତି ପ୍ରଭାବିତ ହୋଇଛି ଜାଣିବାକୁ ଚାହୁଁଛୁ। ଆପଣଙ୍କ ସୁବିଧା ଅନୁଯାୟୀ ଏକ ଫିଲ୍ଡ ଅଧିକାରୀ ଆପଣଙ୍କୁ ଭେଟିପାରିବେ "
        f"କିମ୍ବା କଲ୍ କରିପାରିବେ। ଆମେ ସାହାଯ୍ୟ ପାଇଁ ଏଠାରେ ଅଛୁ।"
    )
    return {"en": en, "hi": hi, "or": or_}


# --- Number-grounding validation ---

_NUMBER_RE = re.compile(r"\d[\d,]*\.?\d*")


def _extract_numbers(text):
    return _NUMBER_RE.findall(str(text))


def _normalize_number(token):
    token = token.replace(",", "")
    try:
        value = float(token)
        return str(int(value)) if value.is_integer() else str(value)
    except ValueError:
        return token


def _facts_number_pool(facts):
    pool = set()

    def walk(value):
        if isinstance(value, dict):
            for v in value.values():
                walk(v)
        elif isinstance(value, (list, tuple, set)):
            for v in value:
                walk(v)
        else:
            for token in _extract_numbers(value):
                pool.add(_normalize_number(token))

    walk(facts)
    return pool


def _flatten_text(parsed):
    if isinstance(parsed, dict):
        return " ".join(str(v) for v in parsed.values())
    return str(parsed)


def is_grounded(parsed, facts):
    """True iff every number appearing in `parsed` also appears somewhere in
    `facts` (after normalising commas/float formatting)."""
    allowed = _facts_number_pool(facts)
    used = {_normalize_number(t) for t in _extract_numbers(_flatten_text(parsed))}
    return used.issubset(allowed)


# --- LLM plumbing ---


def _get_client():
    import groq

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Put it in a .env file in the project root "
            "(GROQ_API_KEY=gsk_...) or export it in your shell before running "
            "`python -m src.briefings`. Either way it's never written to a cache file - "
            "see .gitignore, .env is already excluded from version control."
        )
    return groq.Groq(api_key=api_key)


def _call_llm(client, system, user_prompt, max_tokens):
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=max_tokens,
        # openai/gpt-oss-* models spend part of max_tokens on a hidden reasoning
        # pass before writing the visible answer; "low" keeps that pass short
        # enough that max_tokens isn't exhausted before any real text comes out.
        reasoning_effort="low",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
    )
    text = response.choices[0].message.content.strip()
    if not text:
        raise RuntimeError(f"empty completion (finish_reason={response.choices[0].finish_reason})")
    return text


def _parse_outreach_json(raw):
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    data = json.loads(cleaned)
    return {"en": str(data["en"]).strip(), "hi": str(data["hi"]).strip(), "or": str(data["or"]).strip()}


def _generate_with_validation(facts, system_prompt, max_tokens, parse_response, fallback, client):
    """Ask the LLM, validate every number is grounded in facts, retry once on
    failure, else fall back to a plain (always-grounded) template."""
    user_prompt = json.dumps(facts, ensure_ascii=False, indent=2)

    for attempt in range(MAX_ATTEMPTS):
        try:
            raw = _call_llm(client, system_prompt, user_prompt, max_tokens)
            parsed = parse_response(raw)
        except Exception as exc:
            print(f"  [warn] attempt {attempt + 1} failed: {exc}")
            continue
        if is_grounded(parsed, facts):
            return parsed, "llm"
        print(f"  [warn] attempt {attempt + 1} rejected: ungrounded number in output")

    return fallback(facts), "template"


def generate_branch_briefing(facts, client=None):
    client = client or _get_client()
    return _generate_with_validation(
        facts,
        system_prompt=BRANCH_SYSTEM_PROMPT,
        max_tokens=1024,  # generous headroom for gpt-oss's hidden reasoning pass, see _call_llm
        parse_response=lambda raw: raw.strip(),
        fallback=template_branch_briefing,
        client=client,
    )


def generate_outreach_message(facts, client=None):
    client = client or _get_client()
    return _generate_with_validation(
        facts,
        system_prompt=OUTREACH_SYSTEM_PROMPT,
        max_tokens=1500,  # 3 languages + JSON structure + reasoning pass
        parse_response=_parse_outreach_json,
        fallback=template_outreach_message,
        client=client,
    )


def backtranslate_message(text, language_code, client=None):
    """English back-translation of one Hindi/Odia message - an audit tool,
    not borrower-facing content, so no number-grounding check applies here."""
    client = client or _get_client()
    system = (
        f"You are a professional translator. Translate the following {LANGUAGE_NAMES[language_code]} "
        "text into natural, faithful English. Respond with ONLY the English translation - no "
        "commentary, no quotation marks, no additional notes."
    )
    return _call_llm(client, system, text, max_tokens=600)


def run_backtranslation_audit(client=None):
    """Back-translate every cached Hindi/Odia outreach message to English,
    resumably cached to outreach_backtranslations.json (already-translated
    borrower/language pairs are skipped on rerun, so this is safe to re-run
    after generating more outreach messages).

    This exists because is_grounded() only ever checks numbers - it cannot
    tell whether a Hindi or Odia message quietly promises loan relief in
    words. Back-translating to English makes that checkable by the same
    English FORBIDDEN_PROMISE_RE regex used elsewhere.
    """
    outreach = json.loads(OUTREACH_PATH.read_text(encoding="utf-8"))
    cache = json.loads(BACKTRANSLATIONS_PATH.read_text(encoding="utf-8")) if BACKTRANSLATIONS_PATH.exists() else {}

    client = client or _get_client()
    for borrower_id, entry in outreach.items():
        cache.setdefault(borrower_id, {})
        for lang in ("hi", "or"):
            if lang in cache[borrower_id]:
                continue
            try:
                cache[borrower_id][lang] = backtranslate_message(entry["messages"][lang], lang, client=client)
                print(f"[backtranslate] {borrower_id}/{lang}")
            except Exception as exc:
                print(f"  [warn] back-translation failed for {borrower_id}/{lang}: {exc}")

    _write_json(BACKTRANSLATIONS_PATH, cache)
    print(f"\n[write] back-translations for {len(cache)} borrowers -> {BACKTRANSLATIONS_PATH}")
    return cache


def check_backtranslation_violations(cache):
    """Per language: list of (borrower_id, translated_text) whose English
    back-translation mentions a forbidden promise."""
    violations = {"hi": [], "or": []}
    for borrower_id, langs in cache.items():
        for lang, text in langs.items():
            if lang in violations and FORBIDDEN_PROMISE_RE.search(text):
                violations[lang].append((borrower_id, text))
    return violations


def print_violation_report(violations):
    for lang, label in (("hi", "Hindi"), ("or", "Odia")):
        hits = violations[lang]
        print(f"{label}: {len(hits)} violation(s)")
        for borrower_id, text in hits:
            print(f"  {borrower_id}: {text}")


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit", type=int, default=None, help="Only process the first N high-tier borrowers (smoke test)."
    )
    parser.add_argument(
        "--audit-translations",
        action="store_true",
        help="Back-translate cached Hindi/Odia outreach messages to English, check for forbidden "
        "promise language, print a per-language report, and exit (skips normal generation).",
    )
    parser.add_argument(
        "--only-branches",
        type=str,
        default=None,
        help="Comma-separated branch_ids (e.g. BR003,BR004,BR015) to regenerate via the LLM. "
        "Facts are refreshed for every branch either way (cheap, local, no API calls) so the "
        "cache stays on one schema; outreach.json is left untouched. Merges into the existing "
        "briefings.json rather than overwriting it.",
    )
    args = parser.parse_args()

    if args.audit_translations:
        cache = run_backtranslation_audit()
        print()
        print_violation_report(check_backtranslation_violations(cache))
        return

    if args.only_branches:
        target_ids = {b.strip() for b in args.only_branches.split(",") if b.strip()}
        df = load_final_portfolio()
        client = _get_client()
        briefings = json.loads(BRIEFINGS_PATH.read_text(encoding="utf-8")) if BRIEFINGS_PATH.exists() else {}

        for branch_id in sorted(df["branch_id"].unique()):
            facts = build_branch_facts(df, branch_id)
            if branch_id in target_ids:
                text, source = generate_branch_briefing(facts, client=client)
                briefings[branch_id] = {"facts": facts, "text": text, "source": source}
                print(f"[branch] {branch_id}: {source}")
            else:
                briefings.setdefault(branch_id, {})["facts"] = facts

        _write_json(BRIEFINGS_PATH, briefings)
        print(f"\n[write] regenerated {len(target_ids)} branch(es) -> {BRIEFINGS_PATH}")
        return

    df = load_final_portfolio()
    client = _get_client()

    branch_ids = sorted(df["branch_id"].unique())
    briefings = {}
    for branch_id in branch_ids:
        facts = build_branch_facts(df, branch_id)
        text, source = generate_branch_briefing(facts, client=client)
        briefings[branch_id] = {"facts": facts, "text": text, "source": source}
        print(f"[branch] {branch_id}: {source}")

    high_df = df[df["tier"] == "high"].reset_index(drop=True)
    if args.limit is not None:
        high_df = high_df.head(args.limit)

    outreach = {}
    for _, row in high_df.iterrows():
        facts = build_borrower_facts(row)
        messages, source = generate_outreach_message(facts, client=client)
        outreach[row["borrower_id"]] = {"facts": facts, "messages": messages, "source": source}
        print(f"[outreach] {row['borrower_id']}: {source}")

    _write_json(BRIEFINGS_PATH, briefings)
    _write_json(OUTREACH_PATH, outreach)
    print(f"\n[write] {len(briefings)} branch briefings -> {BRIEFINGS_PATH}")
    print(f"[write] {len(outreach)} outreach messages -> {OUTREACH_PATH}")


if __name__ == "__main__":
    main()
