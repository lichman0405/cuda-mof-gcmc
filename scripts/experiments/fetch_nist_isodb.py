"""Step 3 (Item 5): Fetch curated experimental CO2 298 K isotherms from NIST ISODB.

Queries:
  - https://adsorption.nist.gov/matdb/api/materials.json (hashkey -> name & synonyms)
  - https://adsorption.nist.gov/isodb/api/isotherms.json (experiment index)
  - https://adsorption.nist.gov/isodb/api/isotherm/<filename>.json (full tables)

Filters strictly for pure experimental ('exp'), single-component CO2, ~298 K,
unmodified pristine adsorbents (no water-loaded, amine-grafted, or composite
variants), gravimetric units, and physical pressure ranges (< 100 bar) covering
sub-atmospheric / 1-5 bar regimes.
"""

import csv
import json
import shutil
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "exp_isotherms"

MATDB_URL = "https://adsorption.nist.gov/matdb/api/materials.json"
ISODB_INDEX_URL = "https://adsorption.nist.gov/isodb/api/isotherms.json"
CO2_INCHIKEY = "CURLTUGMZLYLDI-UHFFFAOYSA-N"

EXCLUDE_ADS_SUBSTRINGS = [
    "water", "loaded", "doped", "modified", "pellet", "composite",
    "membrane", "amine", "mmen", "tepa", "pei", "en-", "ed-", "pmg-",
    "graphene", "go/", "cnt", "@", "hydr", "humid", "ball-milled",
    "defect", "aged", "irradiated", "acid", "alkali", "naoh",
]


def fetch_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "cuda-mof-gcmc/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def unit_to_mmol_g(val: float, units: str) -> float:
    u = (units or "").strip().lower()
    if u in ("mmol/g", "mol/kg"):
        return float(val)
    if u in ("mol/g",):
        return float(val) * 1000.0
    if u == "mg/g":
        return float(val) / 44.0095
    if u == "g/100g":
        return float(val) * 10.0 / 44.0095
    if u == "wt%":
        w = float(val)
        return (w / (100.0 - w)) * 1000.0 / 44.0095 if w < 99.0 else float(val)
    if u in ("cm3(stp)/g", "cm^3(stp)/g", "cc(stp)/g", "ml(stp)/g"):
        return float(val) / 22.41396954
    if u == "g/g":
        return float(val) * 1000.0 / 44.0095
    raise ValueError(f"Unsupported gravimetric NIST ISODB unit: {units!r}")


def pressure_to_bar(p: float, p_units: str) -> float:
    u = (p_units or "bar").strip().lower()
    if u == "bar":
        return float(p)
    if u == "kpa":
        return float(p) / 100.0
    if u == "pa":
        return float(p) / 1e5
    if u == "mpa":
        return float(p) * 10.0
    if u == "atm":
        return float(p) * 1.01325
    if u in ("torr", "mmhg"):
        return float(p) * 1.01325 / 760.0
    raise ValueError(f"Unsupported pressure unit: {p_units!r}")


def is_pristine_name(names_str: str, allowed_tokens) -> bool:
    if not any(tok in names_str for tok in allowed_tokens):
        return False
    return not any(bad in names_str for bad in EXCLUDE_ADS_SUBSTRINGS)


def pick_best_records(index_rows, mat_map):
    targets = {
        "Mg-MOF-74": {
            "adsorbents": ["mg-mof-74", "cpo-27-mg", "mg2(dobdc)", "mg-dobdc", "mg/dobdc", "mof-74-mg"],
            "preferred_dois": ["10.1039/c1ee01720a", "10.1039/c4sc01190b", "10.1021/ja8057953", "10.1021/acs.jpcc.7b08239"],
        },
        "Ni-MOF-74": {
            "adsorbents": ["ni-mof-74", "cpo-27-ni", "ni2(dobdc)", "ni-dobdc", "ni/dobdc", "mof-74-ni"],
            "preferred_dois": ["10.1039/c4sc01190b", "10.1021/ja901125z", "10.1021/acs.jpcc.5b03700"],
        },
        "HKUST-1": {
            "adsorbents": ["hkust-1", "cu-btc", "cubtc", "basolite c300", "basolite c 300"],
            "preferred_dois": ["10.1021/ja9057234", "10.1039/c1ee01720a", "10.1021/ja0570032", "10.1002/aic.15837"],
        },
        "IRMOF-1": {
            "adsorbents": ["irmof-1", "mof-5", "irmof1"],
            "preferred_dois": ["10.1021/ja0570032", "10.1021/ja076595g", "10.1021/acs.jctc.5b00093"],
        },
    }

    selected = {}
    for mof_key, spec in targets.items():
        cands = []
        for row in index_rows:
            if str(row.get("category", "")).lower() != "exp":
                continue
            ads = row.get("adsorbates", [])
            if len(ads) != 1 or ads[0].get("InChIKey") != CO2_INCHIKEY:
                continue
            temp = row.get("temperature")
            if temp is None or abs(float(temp) - 298.0) > 2.5:
                continue
            hk = row.get("adsorbent", {}).get("hashkey", "")
            names_str = mat_map.get(hk, "")
            if not is_pristine_name(names_str, spec["adsorbents"]):
                continue
            doi = str(row.get("DOI", "")).lower()
            score = 0
            for idx_d, p_doi in enumerate(spec["preferred_dois"]):
                if p_doi in doi:
                    score += (10 - idx_d) * 20
            if row.get("tabular_data", 0):
                score += 5
            if abs(float(temp) - 298.0) <= 0.5:
                score += 3
            cands.append((score, row))
        cands.sort(key=lambda x: -x[0])
        selected[mof_key] = [r for _, r in cands[:25]]
    return selected


def main():
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading NIST MATDB from {MATDB_URL} ...")
    mats = fetch_json(MATDB_URL)
    mat_map = {}
    for m in mats:
        all_names = [str(m.get("name", ""))] + [str(s) for s in m.get("synonyms", [])]
        mat_map[m["hashkey"]] = " | ".join(all_names).lower()

    print(f"Downloading NIST ISODB index from {ISODB_INDEX_URL} ...")
    index_rows = fetch_json(ISODB_INDEX_URL)
    candidates = pick_best_records(index_rows, mat_map)
    manifest = {}

    for mof_key, rows in candidates.items():
        print(f"\n=== {mof_key}: {len(rows)} pristine 298 K CO2 candidates ===")
        saved_for_mof = []
        seen_dois = set()
        for row in rows:
            fn = row["filename"]
            url = f"https://adsorption.nist.gov/isodb/api/isotherm/{fn}.json"
            try:
                rec = fetch_json(url)
                ads_name = rec.get("adsorbent", {}).get("name", "")
                if any(bad in ads_name.lower() for bad in EXCLUDE_ADS_SUBSTRINGS):
                    continue
                doi = rec.get("DOI", "")
                if doi in seen_dois and len(saved_for_mof) >= 1:
                    continue
                p_units = rec.get("pressureUnits", "bar")
                a_units = rec.get("adsorptionUnits", "mmol/g")
                pts = rec.get("isotherm_data", [])
                clean_pts = []
                for pt in pts:
                    p_raw = pt.get("pressure")
                    if p_raw is None:
                        continue
                    sp_list = pt.get("species_data", [])
                    a_raw = sp_list[0].get("adsorption") if sp_list else pt.get("total_adsorption")
                    if a_raw is None:
                        continue
                    p_bar = pressure_to_bar(float(p_raw), p_units)
                    u_mmol_g = unit_to_mmol_g(float(a_raw), a_units)
                    if p_bar < 0 or u_mmol_g < 0:
                        continue
                    clean_pts.append({"P_bar": p_bar, "uptake_mmol_g": u_mmol_g, "P_raw": p_raw, "uptake_raw": a_raw})
                if len(clean_pts) < 5:
                    continue
                clean_pts.sort(key=lambda x: x["P_bar"])
                p_min, p_max = clean_pts[0]["P_bar"], clean_pts[-1]["P_bar"]
                u_min, u_max = clean_pts[0]["uptake_mmol_g"], clean_pts[-1]["uptake_mmol_g"]
                # Guard against mislabeled Pa-as-bar records in ISODB or pure ultra-low pressure traces
                if p_max > 100.0 or p_max < 0.08:
                    continue
                # Ensure we have points in the sub-10 bar window unless it's high-pressure IRMOF-1
                if mof_key != "IRMOF-1" and p_min > 0.5:
                    continue

                slug = f"{mof_key.lower().replace('-', '')}_{fn}".replace("/", "_")
                json_path = OUT_DIR / f"{slug}.json"
                csv_path = OUT_DIR / f"{slug}.csv"
                rec_out = {
                    "mof": mof_key,
                    "filename": fn,
                    "DOI": doi,
                    "temperature_K": rec.get("temperature"),
                    "adsorbent_name": ads_name,
                    "original_pressure_units": p_units,
                    "original_adsorption_units": a_units,
                    "category": rec.get("category", ""),
                    "isotherm_type": rec.get("isotherm_type", ""),
                    "points": clean_pts,
                }
                json_path.write_text(json.dumps(rec_out, indent=2), encoding="utf-8")
                with open(csv_path, "w", encoding="utf-8", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=["P_bar", "uptake_mmol_g", "P_raw", "uptake_raw"])
                    w.writeheader()
                    w.writerows(clean_pts)

                print(
                    f"  [SAVED] {fn} | DOI={doi} | ads={ads_name} | T={rec_out['temperature_K']} K | "
                    f"orig={a_units} | n={len(clean_pts)} | P=[{p_min:.4g}, {p_max:.4g}] bar | "
                    f"U=[{u_min:.3f}, {u_max:.3f}] mmol/g"
                )
                seen_dois.add(doi)
                saved_for_mof.append({
                    "filename": fn,
                    "doi": doi,
                    "adsorbent_name": ads_name,
                    "temperature_K": rec_out["temperature_K"],
                    "original_adsorption_units": a_units,
                    "n_pts": len(clean_pts),
                    "p_range_bar": [p_min, p_max],
                    "u_range_mmol_g": [u_min, u_max],
                    "json": f"data/exp_isotherms/{slug}.json",
                    "csv": f"data/exp_isotherms/{slug}.csv",
                })
                if len(saved_for_mof) >= 2:
                    break
            except Exception as exc:
                print(f"  [SKIP] {fn}: {exc}")
        manifest[mof_key] = saved_for_mof

    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nSaved clean NIST ISODB manifest: {manifest_path}")


if __name__ == "__main__":
    main()
