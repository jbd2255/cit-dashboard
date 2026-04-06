import io
import os
import re
import time
import threading
import requests
import pandas as pd
import xml.etree.ElementTree as ET
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# DOL data state (shared across the app)
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_data = {
    "loading": False,
    "loaded": False,
    "error": None,
    "cit_df": None,       # Filtered F_5500 rows where TYPE_PLAN_ENTITY_CD == "C"
    "sched_c_df": None,   # F_SCH_C rows restricted to CIT ACK_IDs
    "cit_count": 0,
    "sched_c_count": 0,
}

# Google Drive file IDs — set via Railway environment variables (or .env locally)
GDRIVE_5500_ID    = os.environ.get("GDRIVE_5500_ID", "")
GDRIVE_SCHED_C_ID = os.environ.get("GDRIVE_SCHED_C_ID", "")


# ---------------------------------------------------------------------------
# Google Drive download helper
# ---------------------------------------------------------------------------
def _gdrive_download(file_id: str, label: str) -> pd.DataFrame:
    """Download a public Google Drive CSV and return a DataFrame."""
    session = requests.Session()
    url = "https://drive.google.com/uc"
    params = {"export": "download", "id": file_id}

    print(f"[DOL] Downloading {label} ...", flush=True)
    resp = session.get(url, params=params, timeout=300, stream=True)

    # Large files trigger a warning page — extract confirmation token
    if "text/html" in resp.headers.get("Content-Type", ""):
        token = None
        for k, v in resp.cookies.items():
            if "download_warning" in k:
                token = v
                break
        if not token:
            m = re.search(r'confirm=([^&"\'>\s]+)', resp.text)
            if m:
                token = m.group(1)
        params["confirm"] = token or "t"
        resp = session.get(url, params=params, timeout=300, stream=True)

    resp.raise_for_status()

    chunks, total = [], 0
    for chunk in resp.iter_content(chunk_size=4 * 1024 * 1024):
        chunks.append(chunk)
        total += len(chunk)
        print(f"[DOL]   {label}: {total/1024/1024:.0f} MB ...", flush=True)

    content = b"".join(chunks)
    print(f"[DOL] {label}: {len(content)/1024/1024:.1f} MB downloaded", flush=True)

    df = pd.read_csv(io.BytesIO(content), low_memory=False, dtype=str)
    df.columns = [c.strip().upper() for c in df.columns]
    print(f"[DOL] {label}: {len(df):,} rows × {len(df.columns)} cols", flush=True)
    return df


# ---------------------------------------------------------------------------
# Column-name detection (DOL uses different names across years)
# ---------------------------------------------------------------------------
def _find_col(df: pd.DataFrame, candidates: list) -> str | None:
    cols_up = {c.upper(): c for c in df.columns}
    for cand in candidates:
        if cand.upper() in cols_up:
            return cols_up[cand.upper()]
    # Substring fallback
    for cand in candidates:
        for col_up, col in cols_up.items():
            if cand.upper() in col_up:
                return col
    return None


# ---------------------------------------------------------------------------
# Background data loader
# ---------------------------------------------------------------------------
def _load_dol_data():
    with _lock:
        _data["loading"] = True
        _data["error"] = None

    try:
        if not GDRIVE_5500_ID or not GDRIVE_SCHED_C_ID:
            raise RuntimeError(
                "GDRIVE_5500_ID and GDRIVE_SCHED_C_ID environment variables are not set."
            )

        # ── F_5500 ────────────────────────────────────────────────────────
        df5 = _gdrive_download(GDRIVE_5500_ID, "F_5500_2023")

        type_col = _find_col(df5, ["TYPE_PLAN_ENTITY_CD", "TYPE_PLAN_ENTITY", "ENTITY_CD", "ENTITY_TYPE"])
        if type_col:
            cit_df = df5[df5[type_col].str.strip() == "C"].copy()
            print(f"[DOL] CIT rows after filter: {len(cit_df):,}", flush=True)
        else:
            print(f"[DOL] WARNING: entity-type column not found. Columns: {list(df5.columns[:30])}", flush=True)
            cit_df = df5.copy()

        # Keep only the columns we actually use to reduce RAM
        keep_5500 = []
        for candidates in [
            ["ACK_ID"],
            ["PLAN_NAME", "PLAN_NAME_DFE", "NAME_OF_PLAN"],
            ["SPONS_DFE_NAME", "SPONS_NAME", "SPONSOR_NAME", "DFE_NAME"],
            ["SPONS_DFE_EIN", "PLAN_EIN", "EIN"],
            ["FORM_TAX_PRD", "TAX_PRD", "PLAN_YEAR_END", "YEAR"],
            ["FORM_PLAN_YEAR_BEGIN_DATE", "PLAN_YEAR_BEGIN"],
        ]:
            c = _find_col(cit_df, candidates)
            if c and c not in keep_5500:
                keep_5500.append(c)

        cit_df = cit_df[keep_5500].reset_index(drop=True) if keep_5500 else cit_df
        del df5

        # ── F_SCH_C_PART1_ITEM2 ──────────────────────────────────────────
        df_c = _gdrive_download(GDRIVE_SCHED_C_ID, "F_SCH_C_PART1_ITEM2_2023")

        ack_col_5500 = _find_col(cit_df, ["ACK_ID"])
        ack_col_c    = _find_col(df_c,   ["ACK_ID"])

        if ack_col_5500 and ack_col_c:
            cit_ids = set(cit_df[ack_col_5500].str.strip())
            df_c = df_c[df_c[ack_col_c].str.strip().isin(cit_ids)].copy()
            print(f"[DOL] Schedule C rows for CITs: {len(df_c):,}", flush=True)

        # Keep useful Schedule C columns
        keep_c = []
        for candidates in [
            ["ACK_ID"],
            [
                "SCH_C_PART_I_IT_2_NM", "PART_I_ITEM2A_NM", "PART_1_ITEM2_NM",
                "SERV_PROV_NAME", "SP_NAME", "NAME", "PROVIDER_NAME",
                "SCH_C_PT1_IT2_NM", "PT1_IT2_NM",
            ],
            [
                "SCH_C_PART_I_IT_2_EIN", "PART_I_ITEM2A_EIN", "PART_1_ITEM2_EIN",
                "SERV_PROV_EIN", "SP_EIN",
            ],
            [
                "SCH_C_PART_I_IT_2_SERV_CD", "PART_I_ITEM2B_SERVICE_CD", "PART_1_ITEM2_SERV_CD",
                "SERV_CD", "SERVICE_CD", "SERVICE_CODE", "SVC_CD",
            ],
            [
                "SCH_C_PART_I_IT_2_TOT_COMP_AMT", "PART_I_ITEM2F_TOTAL_COMP_AMT",
                "PART_1_ITEM2_TOTAL_COMP", "TOTAL_COMP", "TOTAL_COMPENSATION",
                "TOT_COMP", "TOTAL_COMP_AMT",
            ],
            [
                "SCH_C_PART_I_IT_2_DIR_COMP_AMT", "PART_I_ITEM2D_DIRECT_COMP_AMT",
                "PART_1_ITEM2_DIRECT_COMP", "DIRECT_COMP", "DIR_COMP",
            ],
        ]:
            c = _find_col(df_c, candidates)
            if c and c not in keep_c:
                keep_c.append(c)

        df_c = df_c[keep_c].reset_index(drop=True) if keep_c else df_c
        print(f"[DOL] Schedule C kept columns: {keep_c}", flush=True)

        with _lock:
            _data["cit_df"]      = cit_df
            _data["sched_c_df"]  = df_c
            _data["cit_count"]   = len(cit_df)
            _data["sched_c_count"] = len(df_c)
            _data["loading"]     = False
            _data["loaded"]      = True

        print("[DOL] ✓ Data load complete.", flush=True)

    except Exception as exc:
        import traceback
        traceback.print_exc()
        with _lock:
            _data["loading"] = False
            _data["error"]   = str(exc)
        print(f"[DOL] ✗ Data load failed: {exc}", flush=True)


def _start_load():
    if GDRIVE_5500_ID and GDRIVE_SCHED_C_ID:
        threading.Thread(target=_load_dol_data, daemon=True).start()
    else:
        print("[DOL] Skipping data load — env vars not set.", flush=True)


_start_load()


# ---------------------------------------------------------------------------
# Service-provider role categorisation from DOL service codes
# Codes reference: DOL Schedule C instructions
# ---------------------------------------------------------------------------
_ROLE_CODES = {
    "trustee":       {"16", "25", "26"},
    "custodian":     {"18", "50"},
    "administrator": {"12", "17", "22", "23", "28"},
    "accountant":    {"14", "70"},
}


def _classify_role(code_str: str) -> str:
    if not code_str or pd.isna(code_str):
        return "other"
    tokens = set(re.findall(r"\d+", str(code_str)))
    for role, codes in _ROLE_CODES.items():
        if tokens & codes:
            return role
    return "other"


def _safe_float(val) -> float:
    try:
        return float(str(val).replace(",", "").strip())
    except (ValueError, TypeError):
        return 0.0


# ---------------------------------------------------------------------------
# DOL API routes
# ---------------------------------------------------------------------------
@app.route("/api/data-status")
def data_status():
    with _lock:
        return jsonify({
            "loading":       _data["loading"],
            "loaded":        _data["loaded"],
            "error":         _data["error"],
            "cit_count":     _data["cit_count"],
            "sched_c_count": _data["sched_c_count"],
        })


@app.route("/api/dol-search")
def dol_search():
    """Search CIT filings by plan name or sponsor/DFE name."""
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"error": "No query provided"}), 400

    with _lock:
        loaded  = _data["loaded"]
        loading = _data["loading"]
        error   = _data["error"]
        cit_df  = _data["cit_df"]
        sched_c = _data["sched_c_df"]

    if not loaded:
        msg = "Data is loading — please wait 60–90 s and try again." if loading \
              else (f"Data load failed: {error}" if error
                    else "DOL data not loaded. Set GDRIVE_5500_ID and GDRIVE_SCHED_C_ID.")
        return jsonify({"error": msg}), 503

    try:
        name_col    = _find_col(cit_df, ["PLAN_NAME", "PLAN_NAME_DFE", "NAME_OF_PLAN"])
        sponsor_col = _find_col(cit_df, ["SPONS_DFE_NAME", "SPONS_NAME", "SPONSOR_NAME", "DFE_NAME"])
        ack_col     = _find_col(cit_df, ["ACK_ID"])
        ein_col     = _find_col(cit_df, ["SPONS_DFE_EIN", "PLAN_EIN", "EIN"])
        year_col    = _find_col(cit_df, ["FORM_TAX_PRD", "TAX_PRD", "PLAN_YEAR_END",
                                          "FORM_PLAN_YEAR_BEGIN_DATE", "PLAN_YEAR_BEGIN"])

        q_up = q.upper()
        mask = pd.Series(False, index=cit_df.index)
        for col in [name_col, sponsor_col]:
            if col:
                mask |= cit_df[col].str.upper().str.contains(q_up, na=False, regex=False)

        hits = cit_df[mask].head(100)
        if hits.empty:
            return jsonify({"results": [], "total": 0})

        # Schedule C column names
        sc_ack   = _find_col(sched_c, ["ACK_ID"])
        sc_name  = _find_col(sched_c, [
            "SCH_C_PART_I_IT_2_NM", "PART_I_ITEM2A_NM", "PART_1_ITEM2_NM",
            "SERV_PROV_NAME", "SP_NAME", "NAME", "PROVIDER_NAME", "PT1_IT2_NM",
        ])
        sc_code  = _find_col(sched_c, [
            "SCH_C_PART_I_IT_2_SERV_CD", "PART_I_ITEM2B_SERVICE_CD", "PART_1_ITEM2_SERV_CD",
            "SERV_CD", "SERVICE_CD", "SERVICE_CODE", "SVC_CD",
        ])
        sc_total = _find_col(sched_c, [
            "SCH_C_PART_I_IT_2_TOT_COMP_AMT", "PART_I_ITEM2F_TOTAL_COMP_AMT",
            "PART_1_ITEM2_TOTAL_COMP", "TOTAL_COMP", "TOTAL_COMPENSATION",
            "TOT_COMP", "TOTAL_COMP_AMT",
        ])
        sc_dir   = _find_col(sched_c, [
            "SCH_C_PART_I_IT_2_DIR_COMP_AMT", "PART_I_ITEM2D_DIRECT_COMP_AMT",
            "PART_1_ITEM2_DIRECT_COMP", "DIRECT_COMP", "DIR_COMP",
        ])

        # Pre-index Schedule C by ACK_ID for fast lookup
        sc_index = {}
        if sc_ack:
            for row in sched_c.itertuples(index=False):
                aid = str(getattr(row, sc_ack, "")).strip()
                sc_index.setdefault(aid, []).append(row)

        results = []
        for _, row in hits.iterrows():
            ack_id       = str(row[ack_col]).strip()   if ack_col     else ""
            plan_name    = str(row[name_col]).strip()   if name_col    else ""
            sponsor_name = str(row[sponsor_col]).strip() if sponsor_col else ""
            ein          = str(row[ein_col]).strip()    if ein_col     else ""
            year         = str(row[year_col]).strip()   if year_col    else ""

            providers = {"custodians": [], "trustees": [],
                         "administrators": [], "accountants": [],
                         "total_fees": 0.0}

            for sp_row in sc_index.get(ack_id, []):
                sp_name  = str(getattr(sp_row, sc_name,  "")).strip() if sc_name  else ""
                svc_code = str(getattr(sp_row, sc_code,  "")).strip() if sc_code  else ""
                fee_raw  = getattr(sp_row, sc_total, None) if sc_total else None
                if fee_raw is None and sc_dir:
                    fee_raw = getattr(sp_row, sc_dir, None)
                fee = _safe_float(fee_raw)

                providers["total_fees"] += fee

                if not sp_name or sp_name.lower() in ("nan", ""):
                    continue

                role = _classify_role(svc_code)
                entry = {"name": sp_name, "fee": fee}
                if role == "custodian":
                    providers["custodians"].append(entry)
                elif role == "trustee":
                    providers["trustees"].append(entry)
                elif role == "administrator":
                    providers["administrators"].append(entry)
                elif role == "accountant":
                    providers["accountants"].append(entry)

            results.append({
                "plan_name":    plan_name,
                "sponsor_name": sponsor_name,
                "ein":          ein,
                "year":         year,
                "ack_id":       ack_id,
                **providers,
            })

        return jsonify({"results": results, "total": int(mask.sum())})

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.route("/api/provider-search")
def provider_search():
    """Reverse lookup: find all CITs that use a given service provider."""
    q    = request.args.get("q", "").strip()
    role = request.args.get("role", "").strip().lower()
    if not q:
        return jsonify({"error": "No query provided"}), 400

    with _lock:
        loaded  = _data["loaded"]
        loading = _data["loading"]
        cit_df  = _data["cit_df"]
        sched_c = _data["sched_c_df"]

    if not loaded:
        return jsonify({"error": "Data loading" if loading else "Data not loaded"}), 503

    try:
        sc_name  = _find_col(sched_c, [
            "SCH_C_PART_I_IT_2_NM", "PART_I_ITEM2A_NM", "PART_1_ITEM2_NM",
            "SERV_PROV_NAME", "SP_NAME", "NAME", "PROVIDER_NAME", "PT1_IT2_NM",
        ])
        sc_ack   = _find_col(sched_c, ["ACK_ID"])
        sc_code  = _find_col(sched_c, [
            "SCH_C_PART_I_IT_2_SERV_CD", "PART_I_ITEM2B_SERVICE_CD", "PART_1_ITEM2_SERV_CD",
            "SERV_CD", "SERVICE_CD", "SVC_CD",
        ])
        sc_total = _find_col(sched_c, [
            "SCH_C_PART_I_IT_2_TOT_COMP_AMT", "PART_I_ITEM2F_TOTAL_COMP_AMT",
            "PART_1_ITEM2_TOTAL_COMP", "TOTAL_COMP", "TOT_COMP",
        ])

        if not sc_name:
            return jsonify({"error": "Provider-name column not found in Schedule C data"}), 500

        q_up = q.upper()
        prov_rows = sched_c[sched_c[sc_name].str.upper().str.contains(q_up, na=False, regex=False)]

        if role and sc_code:
            codes = _ROLE_CODES.get(role, set())
            if codes:
                prov_rows = prov_rows[
                    prov_rows[sc_code].apply(
                        lambda v: bool(set(re.findall(r"\d+", str(v))) & codes)
                    )
                ]

        if prov_rows.empty:
            return jsonify({"results": [], "total": 0})

        ack_col_5500 = _find_col(cit_df, ["ACK_ID"])
        name_col     = _find_col(cit_df, ["PLAN_NAME", "PLAN_NAME_DFE", "NAME_OF_PLAN"])
        sponsor_col  = _find_col(cit_df, ["SPONS_DFE_NAME", "SPONS_NAME", "SPONSOR_NAME"])

        # Build a quick ACK_ID → plan name lookup
        plan_lookup = {}
        if ack_col_5500 and name_col:
            plan_lookup = dict(zip(
                cit_df[ack_col_5500].str.strip(),
                cit_df[name_col].str.strip()
            ))

        results = []
        for _, sp_row in prov_rows.head(200).iterrows():
            ack_id = str(sp_row[sc_ack]).strip() if sc_ack else ""
            fee = _safe_float(sp_row[sc_total]) if sc_total else 0.0
            results.append({
                "plan_name":     plan_lookup.get(ack_id, ""),
                "ack_id":        ack_id,
                "provider_name": str(sp_row[sc_name]).strip(),
                "service_code":  str(sp_row[sc_code]).strip() if sc_code else "",
                "fee":           fee,
            })
            if len(results) >= 150:
                break

        return jsonify({"results": results, "total": len(prov_rows)})

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
# Serve frontend
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return send_file("index.html")


# ---------------------------------------------------------------------------
# EDGAR helpers (unchanged from original)
# ---------------------------------------------------------------------------
HEADERS = {
    "User-Agent": "Joseph Govea josephgovea@gmail.com",
    "Accept-Encoding": "gzip, deflate",
    "Accept": "application/json, text/html, */*",
    "Connection": "keep-alive",
}


def edgar_get(url, retries=3):
    for i in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code == 429:
                time.sleep(3)
                continue
            return r
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(1)


# ---------------------------------------------------------------------------
# EDGAR /search (original logic, preserved exactly)
# ---------------------------------------------------------------------------
@app.route("/search")
def search():
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"error": "No query provided"}), 400

    try:
        ciks = []
        seen = set()

        ft_url = f"https://efts.sec.gov/EFTS/hits.json?q=%22{requests.utils.quote(q)}%22&forms=N-CEN"
        r = edgar_get(ft_url)
        if r.ok:
            data = r.json()
            hits = data.get("hits", {}).get("hits", [])
            for h in hits:
                src = h.get("_source", {})
                cik = src.get("entity_id", "")
                dn  = src.get("display_names", [])
                name = dn[0] if dn else src.get("entity_name", "")
                if cik and cik not in seen:
                    seen.add(cik)
                    ciks.append({"cik": cik, "name": name})
                if len(ciks) >= 20:
                    break

        search_terms = [q]
        words = q.split()
        if len(words) > 2:
            search_terms.append(" ".join(words[:2]))

        if len(ciks) < 5:
            for term in search_terms:
                co_url = (
                    f"https://www.sec.gov/cgi-bin/browse-edgar?"
                    f"company={requests.utils.quote(term)}&CIK=&type=N-CEN"
                    f"&dateb=&owner=include&count=100&search_text=&action=getcompany&output=atom"
                )
                r2 = edgar_get(co_url)
                if r2.ok:
                    matches = []
                    try:
                        root2 = ET.fromstring(r2.text)
                        ns2   = {"atom": "http://www.w3.org/2005/Atom"}
                        ci    = root2.find("atom:company-info", ns2)
                        if ci is not None:
                            cik_el  = ci.find("atom:cik", ns2)
                            name_el = ci.find("atom:conformed-name", ns2)
                            if cik_el is not None:
                                matches.append((cik_el.text.strip(),
                                                name_el.text.strip() if name_el is not None else ""))
                        for entry in root2.findall("atom:entry", ns2):
                            content  = entry.find("atom:content", ns2)
                            entry_ci = content.find("atom:company-info", ns2) if content is not None else None
                            if entry_ci is not None:
                                cik_el  = entry_ci.find("atom:cik", ns2)
                                name_el = entry_ci.find("atom:conformed-name", ns2)
                                if cik_el is not None:
                                    matches.append((cik_el.text.strip(),
                                                    (name_el.text or "").strip() if name_el is not None else ""))
                                    continue
                            eid = entry.find("atom:id", ns2)
                            if eid is not None and eid.text:
                                cik_m = re.search(r"cik=(\d+)", eid.text, re.IGNORECASE)
                                if cik_m:
                                    matches.append((cik_m.group(1), ""))
                    except Exception:
                        pass

                    if not matches:
                        matches = re.findall(
                            r"action=getcompany&amp;CIK=(\d+)[^>]*>\s*([^<]{3,80})</a>", r2.text
                        )
                    if not matches:
                        matches = re.findall(
                            r"CIK=(\d+)[^>]*>\s*([A-Z][^<]{2,79})</a>", r2.text
                        )

                    for cik, name in matches:
                        name = name.strip()
                        if cik not in seen:
                            seen.add(cik)
                            ciks.append({"cik": cik, "name": name})
                        if len(ciks) >= 20:
                            break

                if len(ciks) >= 5:
                    break

        if not ciks:
            return jsonify({"error": (
                f"No N-CEN filers found for '{q}'. "
                "Try the fund family name (e.g. 'Vanguard Chester Funds')."
            )}), 200

        results = []
        for item in ciks[:15]:
            detail = _fetch_ncen_detail(item["cik"], item["name"], "")
            if detail:
                results.append(detail)
            time.sleep(0.15)

        return jsonify({"results": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _fetch_ncen_detail(cik_raw, fallback_name, fallback_date):
    try:
        cik_padded = str(cik_raw).replace("CIK", "").strip().zfill(10)
        cik_int    = int(cik_padded)

        sub_url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
        sub_r   = edgar_get(sub_url)
        sub_r.raise_for_status()
        sub = sub_r.json()

        entity_name = sub.get("name", fallback_name)
        filings     = sub.get("filings", {}).get("recent", {})
        forms       = filings.get("form", [])
        accessions  = filings.get("accessionNumber", [])
        dates       = filings.get("filingDate", [])

        ncen_idx = next((i for i, f in enumerate(forms) if f == "N-CEN"), None)
        if ncen_idx is None:
            return None

        acc          = accessions[ncen_idx]
        filing_date  = dates[ncen_idx]
        acc_nodash   = acc.replace("-", "")
        edgar_url    = (
            f"https://www.sec.gov/cgi-bin/browse-edgar?"
            f"action=getcompany&CIK={cik_padded}&type=N-CEN&dateb=&owner=include&count=5"
        )

        xml_url = _find_ncen_xml(cik_int, acc_nodash, acc)
        if not xml_url:
            return {
                "name": entity_name, "cik": str(cik_int),
                "filingDate": filing_date,
                "custodians": [], "admins": [], "tas": [], "accountants": [],
                "edgarUrl": edgar_url,
            }

        time.sleep(0.15)
        xml_r = edgar_get(xml_url)
        xml_r.raise_for_status()
        result = _parse_ncen_xml(xml_r.text)
        result.update({"name": entity_name, "cik": str(cik_int),
                        "filingDate": filing_date, "edgarUrl": edgar_url})
        return result

    except Exception:
        return None


def _find_ncen_xml(cik_int, acc_nodash, acc):
    try:
        listing_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/"
        r = edgar_get(listing_url)
        if r.ok:
            matches = re.findall(r'href="([^"]+\.xml)"', r.text, re.IGNORECASE)
            for m in matches:
                full = ("https://www.sec.gov" + m) if m.startswith("/Archives") else \
                       m if m.startswith("http") else \
                       f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{m}"
                if "xsl" not in full.lower() and "/" not in m:
                    return full
            for m in matches:
                full = ("https://www.sec.gov" + m) if m.startswith("/Archives") else \
                       m if m.startswith("http") else \
                       f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{m}"
                if "xsl" not in full.lower():
                    return full
    except Exception:
        pass

    try:
        idx_url = (
            f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{acc}-index.htm"
        )
        r = edgar_get(idx_url)
        if r.ok:
            matches = re.findall(
                r'href="(/Archives/edgar/data/[^"]+\.xml)"', r.text, re.IGNORECASE
            )
            for m in matches:
                if "xsl" not in m.lower():
                    return "https://www.sec.gov" + m
    except Exception:
        pass

    return None


def _parse_ncen_xml(xml_text):
    custodians, admins, tas, accountants = [], [], [], []
    try:
        root = ET.fromstring(xml_text)

        def get_all(tag):
            out = []
            for el in root.iter():
                local = el.tag.split("}")[-1] if "}" in el.tag else el.tag
                if local == tag:
                    v = (el.text or "").strip()
                    if v:
                        out.append(v)
            return list(dict.fromkeys(out))

        for tag in ["custodianName", "custName", "nameOfCustodian"]:
            custodians.extend(get_all(tag))
        for tag in ["adminName", "administratorName", "nameOfAdmin", "fundAdminName"]:
            admins.extend(get_all(tag))
        for tag in ["transferAgentName", "tAName", "nameOfTransferAgent"]:
            tas.extend(get_all(tag))
        for tag in ["publicAccountantName", "accountantName", "auditorName", "nameOfAccountant"]:
            accountants.extend(get_all(tag))

    except Exception:
        pass

    return {
        "custodians":  list(dict.fromkeys(custodians))[:4],
        "admins":      list(dict.fromkeys(admins))[:3],
        "tas":         list(dict.fromkeys(tas))[:3],
        "accountants": list(dict.fromkeys(accountants))[:2],
    }


# ---------------------------------------------------------------------------
# EDGAR /debug endpoint (original, preserved)
# ---------------------------------------------------------------------------
@app.route("/debug")
def debug():
    cik = request.args.get("cik", "1548609")
    try:
        cik_padded = str(cik).zfill(10)
        cik_int    = int(cik_padded)
        sub_r      = edgar_get(f"https://data.sec.gov/submissions/CIK{cik_padded}.json")
        sub_r.raise_for_status()
        sub = sub_r.json()

        filings    = sub.get("filings", {}).get("recent", {})
        forms      = filings.get("form", [])
        accessions = filings.get("accessionNumber", [])
        ncen_idx   = next((i for i, f in enumerate(forms) if f == "N-CEN"), None)
        if ncen_idx is None:
            return jsonify({"error": "No N-CEN filing found"})

        acc       = accessions[ncen_idx]
        acc_nodash = acc.replace("-", "")
        xml_url   = _find_ncen_xml(cik_int, acc_nodash, acc)
        if not xml_url:
            return jsonify({"error": "Could not find XML file", "acc": acc})

        xml_r = edgar_get(xml_url)
        xml_r.raise_for_status()

        tags = []
        try:
            root = ET.fromstring(xml_r.text)
            tag_set = set()
            for el in root.iter():
                tag = el.tag
                if "}" in tag:
                    tag = tag.split("}")[1]
                tag_set.add(tag)
            tags = sorted(tag_set)
        except Exception as pe:
            tags = [f"XML parse error: {pe}"]

        return jsonify({"xml_url": xml_url, "all_tags": tags, "xml_snippet": xml_r.text[:8000]})

    except Exception as e:
        return jsonify({"error": str(e)})


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=False)
