
import csv
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(override=True)

INPUT_FILE = Path("requiredsoftwares.csv")
OUTPUT_FILE = Path("software_versions.csv")

MODEL = "gemini-2.5-flash"


def find_latest_version(client, name, current_version, url):
    prompt = f"""
You are a software release research agent.

Find the latest stable, officially released version of this software.

Software: {name}
Reference version: {current_version or "Not provided"}
Provided URL: {url or "Not provided"}

Instructions:
1. Use Google Search to find current release information.
2. Prefer official vendor websites and release pages.
3. Do not confuse beta, alpha, nightly, or release candidates
   with the latest stable version.
4. Never invent version numbers or release dates.
5. If you cannot verify the version, return Unknown.
6. Return exactly one JSON object with these fields:
   latest_version, source_url, release_date, status, notes.
7. status must be verified, unverified, or not_found.
8. Mark verified only when reliable evidence supports the result.

Return JSON only, without Markdown fences.
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[
                types.Tool(
                    google_search=types.GoogleSearch()
                )
            ],
            temperature=0,
        ),
    )

    answer = response.text or ""

    # Extract the JSON object from the response.
    match = re.search(r"\{.*\}", answer, re.DOTALL)

    if not match:
        raise ValueError(
            "Gemini did not return a valid JSON object."
        )

    result = json.loads(match.group())

    latest_version = str(
        result.get("latest_version", "Unknown")
    )

    status = str(
        result.get("status", "unverified")
    ).lower()

    if status not in {"verified", "unverified", "not_found"}:
        status = "unverified"

    # Get source URLs from Google Search grounding evidence.
    sources = []
    candidates = getattr(response, "candidates", None) or []

    if candidates:
        metadata = getattr(
            candidates[0], "grounding_metadata", None
        )

        if metadata:
            for chunk in (
                getattr(metadata, "grounding_chunks", None) or []
            ):
                web_source = getattr(chunk, "web", None)

                if web_source and getattr(web_source, "uri", None):
                    sources.append(web_source.uri)

    source_url = str(result.get("source_url", "")).strip()

    if not source_url and sources:
        source_url = sources[0]

    if not source_url:
        source_url = url

    # Do not claim verification without search evidence.
    if not sources or latest_version.lower() == "unknown":
        status = "unverified"

    return {
        "latest_version": latest_version,
        "source_url": source_url,
        "release_date": str(
            result.get("release_date", "Unknown")
        ),
        "status": status,
        "notes": str(result.get("notes", "")),
    }


def main():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing from your .env file."
        )

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"CSV file not found: {INPUT_FILE.resolve()}"
        )

    client = genai.Client(api_key=api_key)

    with INPUT_FILE.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        reader = csv.DictReader(file)

        required = {"sno", "name"}

        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                "CSV must contain sno and name columns."
            )

        rows = list(reader)

    results = []

    for row in rows:
        name = (row.get("name") or "").strip()

        if not name:
            continue

        version = (row.get("version") or "").strip()
        url = (row.get("url") or "").strip()

        print(f"\nResearching {name}...")

        try:
            result = find_latest_version(
                client, name, version, url
            )

            print("Latest:", result["latest_version"])
            print("Status:", result["status"])

        except Exception as error:
            print(f"Could not verify {name}: {error}")

            result = {
                "latest_version": "Unknown",
                "source_url": url,
                "release_date": "Unknown",
                "status": "unverified",
                "notes": str(error)[:300],
            }

        results.append({
            "sno": row.get("sno", ""),
            "name": name,
            "current_version": version,
            **result,
        })

    columns = [
        "sno",
        "name",
        "current_version",
        "latest_version",
        "source_url",
        "release_date",
        "status",
        "notes",
    ]

    with OUTPUT_FILE.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
