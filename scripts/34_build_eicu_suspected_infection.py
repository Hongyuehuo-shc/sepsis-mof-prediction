"""Build auditable eICU suspected-infection events for external validation.

Primary definition adapts the MIT-LCP MIMIC-IV suspected-infection timing rule:
culture precedes antibiotic by <=72 h, or antibiotic precedes culture by <=24 h.
The eICU microLab table contains organism/susceptibility records and therefore
does not capture every culture order; diagnosis-derived flags are exported only
as sensitivity variables and are never labelled as Sepsis-3 by this script.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.paths import INTERMEDIATE_DIR, RESULTS_DIR  # noqa: E402

MIMIC_ABX_SQL = ROOT / "vendor/mimic-code/mimic-iv/concepts_duckdb/medication/antibiotic.sql"


def quoted(values: list[str]) -> str:
    return ",".join("'" + x.replace("'", "''") + "'" for x in values)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--database", default=INTERMEDIATE_DIR / "eicu_v20.duckdb")
    p.add_argument("--output", default=RESULTS_DIR / "eicu_external_validation/suspected_infection")
    a = p.parse_args(); database = Path(a.database); out = Path(a.output)
    if not database.exists():
        raise FileNotFoundError(f"Run eicu-prepare-db first: {database}")
    out.mkdir(parents=True, exist_ok=True)

    source_sql = MIMIC_ABX_SQL.read_text(encoding="utf-8")
    terms = sorted(set(re.findall(r"LOWER\(drug\) LIKE '%([^']+)%'", source_sql, flags=re.I)))
    if len(terms) < 100:
        raise RuntimeError("Could not recover the official MIMIC-IV antibiotic dictionary")
    term_values = quoted(terms)

    con = duckdb.connect(str(database))
    con.execute("CREATE SCHEMA IF NOT EXISTS eicu_derived")
    # Match terms through unnest so the exact matching term remains auditable.
    con.execute(f"""
    CREATE OR REPLACE TABLE eicu_derived.antibiotic_events AS
    WITH terms(term) AS (SELECT * FROM unnest([{term_values}])),
    med AS (
      SELECT m.patientunitstayid,
             coalesce(m.drugstartoffset, m.drugorderoffset) AS antibiotic_offset,
             m.drugname, m.routeadmin, 'medication' AS source_table,
             min(t.term) AS matched_term
      FROM eicu.medication m JOIN terms t ON lower(m.drugname) LIKE '%' || t.term || '%'
      WHERE coalesce(m.drugordercancelled, false)=false
        AND coalesce(m.drugstartoffset, m.drugorderoffset) IS NOT NULL
        AND lower(coalesce(m.routeadmin,'')) NOT LIKE '%eye%'
        AND lower(coalesce(m.routeadmin,'')) NOT LIKE '%ear%'
        AND upper(trim(coalesce(m.routeadmin,''))) NOT IN ('OU','OS','OD','AU','AS','AD','TP','TOP','TOPICAL','EACH NARE')
        AND lower(m.drugname) NOT LIKE '%cream%'
        AND lower(m.drugname) NOT LIKE '%desensitization%'
        AND lower(m.drugname) NOT LIKE '%ophth oint%'
        AND lower(m.drugname) NOT LIKE '%gel%'
        AND lower(m.drugname) NOT LIKE '%ointment%'
        AND lower(m.drugname) NOT LIKE '%mupirocin%'
        AND lower(m.drugname) NOT LIKE '%bactroban%'
      GROUP BY 1,2,3,4,5
    ), inf AS (
      SELECT i.patientunitstayid, i.infusionoffset AS antibiotic_offset,
             i.drugname, 'continuous infusion' AS routeadmin, 'infusionDrug' AS source_table,
             min(t.term) AS matched_term
      FROM eicu.infusionDrug i JOIN terms t ON lower(i.drugname) LIKE '%' || t.term || '%'
      WHERE i.infusionoffset IS NOT NULL GROUP BY 1,2,3,4,5
    )
    SELECT DISTINCT * FROM (SELECT * FROM med UNION ALL SELECT * FROM inf)
    """)

    # microLab may contain one row per organism-antibiotic susceptibility result;
    # collapse to a stay/time/site specimen proxy before time pairing.
    con.execute("""
    CREATE OR REPLACE TABLE eicu_derived.culture_events AS
    SELECT patientunitstayid, culturetakenoffset AS culture_offset,
           coalesce(nullif(trim(culturesite),''),'Unknown') AS culture_site,
           1 AS positive_culture, count(*) AS source_rows,
           count(DISTINCT organism) AS organism_count,
           string_agg(DISTINCT organism, ' | ' ORDER BY organism) AS organisms
    FROM eicu.microLab
    WHERE culturetakenoffset IS NOT NULL
    GROUP BY 1,2,3
    """)

    # Reproduce the MIMIC ordering preference: for each antibiotic, use an
    # eligible preceding culture first; otherwise use an eligible later culture.
    con.execute("""
    CREATE OR REPLACE TABLE eicu_derived.suspected_infection_events AS
    WITH ab AS (
      SELECT *, row_number() OVER (ORDER BY patientunitstayid, antibiotic_offset, drugname, source_table) AS antibiotic_event_id
      FROM eicu_derived.antibiotic_events
    ), prior AS (
      SELECT ab.antibiotic_event_id, c.*,
             row_number() OVER (PARTITION BY ab.antibiotic_event_id ORDER BY c.culture_offset, c.culture_site) AS seq
      FROM ab JOIN eicu_derived.culture_events c USING (patientunitstayid)
      WHERE ab.antibiotic_offset > c.culture_offset
        AND ab.antibiotic_offset <= c.culture_offset + 4320
    ), later AS (
      SELECT ab.antibiotic_event_id, c.*,
             row_number() OVER (PARTITION BY ab.antibiotic_event_id ORDER BY c.culture_offset, c.culture_site) AS seq
      FROM ab JOIN eicu_derived.culture_events c USING (patientunitstayid)
      WHERE ab.antibiotic_offset >= c.culture_offset - 1440
        AND ab.antibiotic_offset < c.culture_offset
    )
    SELECT ab.patientunitstayid, ab.antibiotic_event_id, ab.antibiotic_offset,
           ab.drugname AS antibiotic, ab.routeadmin, ab.source_table, ab.matched_term,
           coalesce(prior.culture_offset, later.culture_offset) AS culture_offset,
           coalesce(prior.culture_site, later.culture_site) AS culture_site,
           coalesce(prior.positive_culture, later.positive_culture) AS positive_culture,
           CASE WHEN prior.antibiotic_event_id IS NOT NULL THEN prior.culture_offset
                WHEN later.antibiotic_event_id IS NOT NULL THEN ab.antibiotic_offset END AS suspected_infection_offset,
           CASE WHEN prior.antibiotic_event_id IS NOT NULL THEN 'culture_before_antibiotic_72h'
                WHEN later.antibiotic_event_id IS NOT NULL THEN 'antibiotic_before_culture_24h' END AS pairing_rule
    FROM ab
    LEFT JOIN prior ON ab.antibiotic_event_id=prior.antibiotic_event_id AND prior.seq=1
    LEFT JOIN later ON ab.antibiotic_event_id=later.antibiotic_event_id AND later.seq=1
    WHERE prior.antibiotic_event_id IS NOT NULL OR later.antibiotic_event_id IS NOT NULL
    """)
    con.execute("""
    CREATE OR REPLACE TABLE eicu_derived.suspected_infection_stays AS
    SELECT patientunitstayid, min(suspected_infection_offset) AS suspected_infection_offset,
           arg_min(antibiotic, suspected_infection_offset) AS index_antibiotic,
           arg_min(culture_site, suspected_infection_offset) AS index_culture_site,
           count(*) AS qualifying_pairs
    FROM eicu_derived.suspected_infection_events GROUP BY 1
    """)

    # Text flags provide sensitivity checks only, without asserting infection onset.
    con.execute("""
    CREATE OR REPLACE TABLE eicu_derived.diagnosis_infection_flags AS
    WITH text_rows AS (
      SELECT patientunitstayid, lower(coalesce(apacheadmissiondx,'')) txt FROM eicu.patient
      UNION ALL SELECT patientunitstayid, lower(coalesce(admitdxpath,'') || ' ' || coalesce(admitdxname,'') || ' ' || coalesce(admitdxtext,'')) FROM eicu.admissionDx
      UNION ALL SELECT patientunitstayid, lower(coalesce(diagnosisstring,'')) FROM eicu.diagnosis
    )
    SELECT patientunitstayid,
      max(CASE WHEN regexp_matches(txt, '(^|[^a-z])(sepsis|septic shock|septicemia)([^a-z]|$)') THEN 1 ELSE 0 END) AS sepsis_text_flag,
      max(CASE WHEN regexp_matches(txt, '(sepsis|septic|infection|infectious|pneumonia|pyelonephritis|urinary tract infection|meningitis|endocarditis|cholangitis|peritonitis|abscess|cellulitis)') THEN 1 ELSE 0 END) AS broad_infection_text_flag
    FROM text_rows GROUP BY 1
    """)

    abx = con.execute("SELECT * FROM eicu_derived.antibiotic_events").fetchdf()
    cultures = con.execute("SELECT * FROM eicu_derived.culture_events").fetchdf()
    events = con.execute("SELECT * FROM eicu_derived.suspected_infection_events").fetchdf()
    stays = con.execute("""
      SELECT p.patientunitstayid, p.uniquepid, p.hospitalid, p.wardid,
             s.suspected_infection_offset, s.index_antibiotic, s.index_culture_site, s.qualifying_pairs,
             coalesce(d.sepsis_text_flag,0) sepsis_text_flag,
             coalesce(d.broad_infection_text_flag,0) broad_infection_text_flag
      FROM eicu.patient p JOIN eicu_derived.suspected_infection_stays s USING(patientunitstayid)
      LEFT JOIN eicu_derived.diagnosis_infection_flags d USING(patientunitstayid)
    """).fetchdf()
    overlap = con.execute("""
      SELECT count(*) total_stays,
       count(*) FILTER (WHERE s.patientunitstayid IS NOT NULL) culture_antibiotic_stays,
       sum(coalesce(d.sepsis_text_flag,0)) sepsis_text_stays,
       sum(coalesce(d.broad_infection_text_flag,0)) broad_infection_text_stays,
       count(*) FILTER (WHERE s.patientunitstayid IS NOT NULL AND d.sepsis_text_flag=1) strict_and_sepsis_text,
       count(*) FILTER (WHERE s.patientunitstayid IS NOT NULL AND d.broad_infection_text_flag=1) strict_and_broad_text
      FROM eicu.patient p LEFT JOIN eicu_derived.suspected_infection_stays s USING(patientunitstayid)
      LEFT JOIN eicu_derived.diagnosis_infection_flags d USING(patientunitstayid)
    """).fetchdf().iloc[0].to_dict()
    con.close()

    abx.to_csv(out / "antibiotic_events.csv.gz", index=False, compression="gzip")
    cultures.to_csv(out / "culture_events.csv", index=False)
    events.to_csv(out / "suspected_infection_events.csv", index=False)
    stays.to_csv(out / "suspected_infection_stays.csv", index=False)
    pd.DataFrame({"match_term": terms, "source": str(MIMIC_ABX_SQL), "adaptation": "matched against eICU medication.drugname and infusionDrug.drugname"}).to_csv(out / "antibiotic_dictionary.csv", index=False)
    route_audit = abx.groupby(["source_table", "routeadmin"], dropna=False).agg(events=("patientunitstayid","size"), stays=("patientunitstayid","nunique")).reset_index().sort_values("events", ascending=False)
    route_audit.to_csv(out / "antibiotic_route_audit.csv", index=False)
    summary = {
      "definition_status": "eICU adaptation of MIMIC-IV suspected-infection timing logic; not itself Sepsis-3",
      "antibiotic_dictionary_terms": len(terms), "antibiotic_events": len(abx),
      "antibiotic_stays": int(abx.patientunitstayid.nunique()), "culture_specimen_proxies": len(cultures),
      "culture_stays": int(cultures.patientunitstayid.nunique()), "qualifying_pairs": len(events),
      "suspected_infection_stays": len(stays), "overlap": {k: int(v) for k,v in overlap.items()},
      "timing_rule_minutes": {"culture_before_antibiotic": 4320, "antibiotic_before_culture": 1440},
      "primary_limitation": "eICU microLab is an organism/susceptibility table and does not enumerate all negative culture orders; the primary paired definition is expected to be high-specificity and under-sensitive.",
      "diagnosis_flags": "Sensitivity variables only; they do not define Sepsis-3 and provide no reliable onset time.",
    }
    (out / "suspected_infection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("eICU suspected-infection event construction complete:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
