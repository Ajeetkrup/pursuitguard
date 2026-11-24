"""Deterministic synthetic Deltek-style CRM + FTS5 narrative seeder for PursuitGuard."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from datetime import date
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = Path("data/crm.db")
DEFAULT_NARRATIVES_DIR = Path("data/narratives")
REFERENCE_DATE = date(2026, 11, 24)

LEED_RANK = {"None": 0, "Silver": 1, "Gold": 2, "Platinum": 3}

PROJECTS: list[tuple[Any, ...]] = [
    # Acute care TX (3 qualifying prime >= 50k sq ft >= 2021 + clashes/ties + traps)
    ("PRJ-21-084", "commercial_aec", "PRJ-21-084 Methodist Hlth Ph2 (TX)", "acute_care", "TX", "prime", 58000, 18000000.0, 2024, "Gold", "2026-11-13"),
    ("PRJ-19-012", "commercial_aec", "PRJ-19-012 Memorial Hermann Med Ctr Ph1 (TX)", "acute_care", "TX", "prime", 69000, 17500000.0, 2022, "Silver", "2026-11-18"),
    ("PRJ-22-091", "commercial_aec", "PRJ-22-091 Memorial Hermann Med Ctr Ph2 (TX)", "acute_care", "TX", "prime", 71000, 19500000.0, 2023, "Gold", "2026-11-19"),
    ("PRJ-24-007", "commercial_aec", "PRJ-24-007 Ctrl TX VA ICU Tower (TX)", "acute_care", "TX", "prime", 78000, 26200000.0, 2024, "Gold", "2026-11-21"),
    ("PRJ-23-055", "commercial_aec", "PRJ-23-055 Dell Seton Surg Wing (TX)", "acute_care", "TX", "prime", 62000, 19200000.0, 2023, "Silver", "2026-11-20"),
    ("PRJ-22-019", "commercial_aec", "PRJ-22-019 TX Childrens Crit Care (TX)", "acute_care", "TX", "prime", 95000, 34000000.0, 2022, "Gold", "2026-11-16"),
    ("PRJ-23-102", "commercial_aec", "PRJ-23-102 Parkland Crit Care Mech (TX)", "acute_care", "TX", "subconsultant", 42000, 8400000.0, 2023, "None", "2026-11-20"),
    ("PRJ-18-044", "commercial_aec", "PRJ-18-044 Baylor St Lukes Bed Expan (TX)", "acute_care", "TX", "prime", 55000, 16500000.0, 2019, "Silver", "2026-11-19"),
    # Ambulatory (non-acute-care distractors)
    ("PRJ-23-088", "commercial_aec", "PRJ-23-088 Banner Phx Outpat Clinic (AZ)", "ambulatory", "AZ", "prime", 44000, 15600000.0, 2023, "Silver", "2026-11-20"),
    ("PRJ-24-031", "commercial_aec", "PRJ-24-031 Scottsdale Amb Care (AZ)", "ambulatory", "AZ", "prime", 31000, 11000000.0, 2024, "Gold", "2026-11-22"),
    ("PRJ-20-051", "commercial_aec", "PRJ-20-051 Dallas Children Ambulatory (TX)", "ambulatory", "TX", "prime", 48000, 14900000.0, 2020, "Silver", "2026-11-19"),
    # Water conveyance TX (CRM sum >= 2020 is $24.3M; Barton Creek closeout flips it to $26.9M)
    ("PRJ-22-119", "commercial_aec", "PRJ-22-119 Barton Crk Pump Stn TX", "water_conveyance", "TX", "prime", 32000, 14200000.0, 2024, "None", "2026-11-12"),
    ("PRJ-21-063", "commercial_aec", "PRJ-21-063 Trinity 72in PCCP Rehab TX", "water_conveyance", "TX", "prime", 18000, 10100000.0, 2021, "None", "2026-11-21"),
    ("PRJ-19-077", "commercial_aec", "PRJ-19-077 San Antonio Lift Stn TX", "water_conveyance", "TX", "prime", 24000, 9200000.0, 2019, "None", "2026-11-20"),
    ("PRJ-22-048", "commercial_aec", "PRJ-22-048 Houston Bayou Force Main (TX)", "water_conveyance", "TX", "subconsultant", 12000, 4500000.0, 2022, "None", "2026-11-19"),
    ("PRJ-21-033", "commercial_aec", "PRJ-21-033 Seattle Harbor Pump (WA)", "water_conveyance", "WA", "prime", 29000, 15400000.0, 2021, "None", "2026-11-18"),
    ("PRJ-23-118", "commercial_aec", "PRJ-23-118 San Diego Water Tunnel (CA)", "water_conveyance", "CA", "prime", 45000, 27500000.0, 2023, "Gold", "2026-11-21"),
    # Civic seismic CA (3 qualifying >= 40k LEED Gold >= 2020 + 1 sub-threshold + CO)
    ("PRJ-23-041", "commercial_aec", "PRJ-23-041 SF Fed Annex Seismic CA", "civic_seismic", "CA", "prime", 82000, 28000000.0, 2025, "Gold", "2026-11-14"),
    ("PRJ-22-015", "commercial_aec", "PRJ-22-015 Oakland Hist Courthouse (CA)", "civic_seismic", "CA", "prime", 96000, 39500000.0, 2023, "Platinum", "2026-11-21"),
    ("PRJ-21-098", "commercial_aec", "PRJ-21-098 Sacramento Justice Ctr (CA)", "civic_seismic", "CA", "prime", 65000, 22400000.0, 2022, "Gold", "2026-11-20"),
    ("PRJ-23-074", "commercial_aec", "PRJ-23-074 Alameda Civic Bracing (CA)", "civic_seismic", "CA", "prime", 38000, 11500000.0, 2023, "Silver", "2026-11-22"),
    ("PRJ-22-104", "commercial_aec", "PRJ-22-104 Denver Federal Building (CO)", "civic_seismic", "CO", "prime", 74000, 24500000.0, 2022, "Gold", "2026-11-20"),
    # Other commercial AEC
    ("PRJ-24-011", "commercial_aec", "PRJ-24-011 Austin Regional Lab (TX)", "lab_research", "TX", "prime", 54000, 19800000.0, 2024, "Gold", "2026-11-21"),
    ("PRJ-21-125", "commercial_aec", "PRJ-21-125 Tampa Acute Tower (FL)", "acute_care", "FL", "prime", 88000, 31000000.0, 2021, "Gold", "2026-11-20"),
    ("PRJ-24-052", "commercial_aec", "PRJ-24-052 Austin Energy Substation (TX)", "industrial_power", "TX", "prime", 26000, 12200000.0, 2024, "None", "2026-11-22"),
    # Restricted defense workspace (isolated from commercial_aec)
    ("PRJ-25-901", "restricted_defense", "PRJ-25-901 Ft Belvoir Command Bunker (VA)", "defense_secure", "VA", "prime", 140000, 94000000.0, 2025, "Gold", "2026-11-23"),
    ("PRJ-24-902", "restricted_defense", "PRJ-24-902 Cheyenne Mtn Shielded Ops (CO)", "defense_secure", "CO", "prime", 85000, 68000000.0, 2024, "Gold", "2026-11-23"),
    ("PRJ-23-903", "restricted_defense", "PRJ-23-903 Norfolk Naval Cyber Wing (VA)", "defense_secure", "VA", "prime", 62000, 47500000.0, 2023, "Silver", "2026-11-22"),
]

STAFF_CREDENTIALS: list[tuple[Any, ...]] = [
    ("CRED-01", "commercial_aec", "Elena Vasquez", "PE", "TX", "TX-PE-91842", "2028-06-30", "2026-11-21"),
    ("CRED-02", "commercial_aec", "Marcus Chen", "PE", "TX", "TX-PE-84410", "2027-12-31", "2026-11-20"),
    ("CRED-03", "commercial_aec", "Priya Nair", "AIA", "TX", "TX-AIA-55210", "2027-09-30", "2026-11-19"),
    ("CRED-04", "commercial_aec", "David Kowalski", "SE", "TX", "TX-SE-33019", "2027-05-15", "2026-11-18"),
    ("CRED-05", "commercial_aec", "Hannah Brooks", "SE", "CA", "CA-SE-61022", "2026-08-31", "2026-11-20"),  # EXPIRED
    ("CRED-06", "commercial_aec", "Victor Alvarez", "SE", "CA", "CA-SE-59814", "2025-12-31", "2026-11-15"),  # EXPIRED
    ("CRED-07", "commercial_aec", "Hannah Brooks", "PE", "CA", "CA-PE-77412", "2027-11-30", "2026-11-20"),
    ("CRED-08", "commercial_aec", "Rachel Lin", "AIA", "CA", "CA-AIA-44910", "2028-03-31", "2026-11-21"),
    ("CRED-09", "commercial_aec", "Owen Mercer", "PE", "CO", "CO-PE-49201", "2027-08-31", "2026-11-19"),
    ("CRED-10", "commercial_aec", "Carlos Mendez", "PE", "FL", "FL-PE-38811", "2026-09-30", "2026-11-10"),  # EXPIRED
    ("CRED-11", "commercial_aec", "Tessa Grant", "PE", "WA", "WA-PE-70215", "2027-10-31", "2026-11-22"),
    ("CRED-12", "commercial_aec", "Elena Vasquez", "LEED_AP", "TX", "GBCI-104492", "2028-01-31", "2026-11-21"),
    ("CRED-13", "commercial_aec", "Rachel Lin", "LEED_AP", "CA", "GBCI-109223", "2028-04-30", "2026-11-21"),
    ("CRED-14", "restricted_defense", "Col. James Sterling", "PE", "VA", "VA-PE-99011", "2028-09-30", "2026-11-23"),
    ("CRED-15", "restricted_defense", "Dr. Nadia Vance", "SE", "VA", "VA-SE-88201", "2028-02-28", "2026-11-23"),
]

CLIENT_PURSUITS: list[tuple[Any, ...]] = [
    ("CP-01", "commercial_aec", "VA VISN 17", "Temple VA Bed Tower Phase 1", 2024, "Won", 94.0, "Top technical score on ICRA Class IV phasing.", "2026-11-21"),
    ("CP-02", "commercial_aec", "VA VISN 17", "San Antonio VA Outpatient Lab", 2023, "Won", 90.5, "Strong past performance & TX PE bench.", "2026-11-21"),
    ("CP-03", "commercial_aec", "VA VISN 17", "Dallas VA Sterile Processing", 2022, "Lost", 83.0, "Scored well clinically; slightly higher fee than incumbent.", "2026-11-21"),
    ("CP-04", "commercial_aec", "VA VISN 17", "Houston VA Emergency Dept Expansion", 2021, "Won", 91.0, "Zero clinical disruption noted by COR.", "2026-11-21"),
    ("CP-05", "commercial_aec", "Texas Water Development Authority", "Barton Creek Pump Station Upgrade", 2024, "Won", 91.0, "Strong SCADA bypass strategy.", "2026-11-12"),
    ("CP-06", "commercial_aec", "Texas Water Development Authority", "North Texas PCCP Interconnect", 2022, "Won", 88.0, "Rated excellent on zero-shutdown relining.", "2026-11-21"),
    ("CP-07", "commercial_aec", "Texas Water Development Authority", "Guadalupe River Intake Expansion", 2021, "Lost", 80.5, "Lost on local JV participation points.", "2026-11-21"),
    ("CP-08", "commercial_aec", "GSA Region 9", "San Francisco Federal Annex Seismic", 2025, "Won", 92.0, "Delivered ahead of schedule with LEED Gold.", "2026-11-14"),
    ("CP-09", "commercial_aec", "GSA Region 9", "San Jose Federal Building Retrofit", 2025, "Lost", 74.0, "Debrief warning: proposed CA SE license renewal was flagged.", "2026-11-20"),
    ("CP-10", "commercial_aec", "GSA Region 9", "Fresno Courthouse Envelope Rehab", 2023, "Lost", 79.0, "Fee sensitivity on historic preservation.", "2026-11-20"),
    ("CP-11", "restricted_defense", "USACE Defense Cyber", "Fort Belvoir Command Bunker", 2025, "Won", 96.0, "Restricted clearance SCIF execution.", "2026-11-23"),
    ("CP-12", "restricted_defense", "USACE Defense Cyber", "Norfolk Naval Cyber Wing", 2023, "Won", 93.0, "Restricted defense workspace only.", "2026-11-22"),
]


def parse_narrative_markdown(path: Path) -> dict[str, Any]:
    """Parse YAML-style frontmatter and markdown body from a narrative file."""
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise ValueError(f"Narrative file missing frontmatter: {path}")
    _, fm_block, body = text.split("---", 2)
    meta: dict[str, Any] = {}
    for line in fm_block.strip().splitlines():
        if ":" not in line:
            continue
        key, raw_val = line.split(":", 1)
        key = key.strip()
        val = raw_val.strip()
        if key in {"claimed_fee_usd"}:
            meta[key] = float(val) if val else None
        elif key in {"claimed_sq_ft", "completion_year"}:
            meta[key] = int(val) if val else None
        else:
            meta[key] = val
    meta["body"] = body.strip()
    meta["source_file"] = path.name
    return meta


def compute_optional_embeddings(texts: list[str]) -> list[str | None]:
    """Optionally embed narratives using Gemini Agent Platform API key or ADC."""
    if not texts:
        return []
    from google import genai  # local import so offline seeding has zero API dependency

    api_key = os.getenv("GOOGLE_API_KEY", "").strip() or os.getenv("GEMINI_API_KEY", "").strip()
    project = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "global").strip() or "global"
    model = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-2")

    if api_key:
        if project:
            client = genai.Client(vertexai=True, api_key=api_key, project=project, location=location)
        else:
            client = genai.Client(vertexai=True, api_key=api_key)
    else:
        client = genai.Client(vertexai=True, project=project or None, location=location)

    response = client.models.embed_content(model=model, contents=texts)
    return [json.dumps(emb.values) for emb in response.embeddings]


def seed_database(
    db_path: Path = DEFAULT_DB_PATH,
    narratives_dir: Path = DEFAULT_NARRATIVES_DIR,
    reference_date: date = REFERENCE_DATE,
    with_embeddings: bool = False,
) -> dict[str, int]:
    """Create and populate the single SQLite CRM + FTS5 database."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE projects (
                project_id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                title TEXT NOT NULL,
                subsector TEXT NOT NULL,
                state TEXT NOT NULL,
                role TEXT NOT NULL,
                sq_ft INTEGER NOT NULL,
                contract_value_usd REAL NOT NULL,
                completion_year INTEGER NOT NULL,
                leed_tier TEXT NOT NULL,
                leed_rank INTEGER NOT NULL,
                last_synced_at TEXT NOT NULL
            );

            CREATE TABLE staff_credentials (
                credential_id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                staff_name TEXT NOT NULL,
                credential_type TEXT NOT NULL,
                state TEXT NOT NULL,
                license_number TEXT NOT NULL,
                expires_on TEXT NOT NULL,
                is_active INTEGER NOT NULL,
                last_synced_at TEXT NOT NULL
            );

            CREATE TABLE client_pursuits (
                pursuit_id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                client_name TEXT NOT NULL,
                opportunity_name TEXT NOT NULL,
                pursuit_year INTEGER NOT NULL,
                outcome TEXT NOT NULL,
                debrief_score REAL NOT NULL,
                debrief_notes TEXT NOT NULL,
                last_synced_at TEXT NOT NULL
            );

            CREATE VIEW v_projects AS SELECT * FROM projects;
            CREATE VIEW v_staff_credentials AS SELECT * FROM staff_credentials;
            CREATE VIEW v_client_pursuits AS SELECT * FROM client_pursuits;

            CREATE TABLE narratives_meta (
                doc_id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                doc_type TEXT NOT NULL,
                title TEXT NOT NULL,
                subsector TEXT NOT NULL,
                state TEXT NOT NULL,
                status TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                claimed_fee_usd REAL,
                claimed_sq_ft INTEGER,
                completion_year INTEGER,
                source_file TEXT NOT NULL,
                body TEXT NOT NULL,
                embedding_json TEXT
            );

            CREATE VIRTUAL TABLE narratives_fts USING fts5(
                doc_id UNINDEXED,
                title,
                body,
                tokenize = 'porter unicode61'
            );
            """
        )

        project_rows = [
            (
                pid,
                ws,
                title,
                subsector,
                state,
                role,
                sqft,
                fee,
                year,
                leed,
                LEED_RANK[leed],
                synced,
            )
            for (pid, ws, title, subsector, state, role, sqft, fee, year, leed, synced) in PROJECTS
        ]
        conn.executemany(
            "INSERT INTO projects VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            project_rows,
        )

        ref_iso = reference_date.isoformat()
        cred_rows = [
            (
                cid,
                ws,
                name,
                ctype,
                state,
                lic,
                exp,
                1 if exp >= ref_iso else 0,
                synced,
            )
            for (cid, ws, name, ctype, state, lic, exp, synced) in STAFF_CREDENTIALS
        ]
        conn.executemany(
            "INSERT INTO staff_credentials VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            cred_rows,
        )

        conn.executemany(
            "INSERT INTO client_pursuits VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            CLIENT_PURSUITS,
        )

        narrative_files = sorted(narratives_dir.glob("*.md"))
        parsed_docs = [parse_narrative_markdown(p) for p in narrative_files]
        embeddings: list[str | None] = (
            compute_optional_embeddings([f"{d['title']}\n{d['body']}" for d in parsed_docs])
            if with_embeddings
            else [None] * len(parsed_docs)
        )

        for doc, emb_json in zip(parsed_docs, embeddings, strict=True):
            conn.execute(
                """
                INSERT INTO narratives_meta (
                    doc_id, workspace_id, doc_type, title, subsector, state,
                    status, updated_at, claimed_fee_usd, claimed_sq_ft,
                    completion_year, source_file, body, embedding_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc["doc_id"],
                    doc["workspace_id"],
                    doc["doc_type"],
                    doc["title"],
                    doc["subsector"],
                    doc["state"],
                    doc["status"],
                    doc["updated_at"],
                    doc.get("claimed_fee_usd"),
                    doc.get("claimed_sq_ft"),
                    doc.get("completion_year"),
                    doc["source_file"],
                    doc["body"],
                    emb_json,
                ),
            )
            conn.execute(
                "INSERT INTO narratives_fts (doc_id, title, body) VALUES (?, ?, ?)",
                (doc["doc_id"], doc["title"], doc["body"]),
            )

        conn.commit()
        return {
            "projects": len(project_rows),
            "staff_credentials": len(cred_rows),
            "client_pursuits": len(CLIENT_PURSUITS),
            "narratives": len(parsed_docs),
        }
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed PursuitGuard SQLite CRM & FTS5 database.")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--narratives-dir", type=Path, default=DEFAULT_NARRATIVES_DIR)
    parser.add_argument("--with-embeddings", action="store_true")
    args = parser.parse_args()

    counts = seed_database(
        db_path=args.db_path,
        narratives_dir=args.narratives_dir,
        with_embeddings=args.with_embeddings,
    )
    print(f"Seeded {args.db_path}: {counts}")


if __name__ == "__main__":
    main()
