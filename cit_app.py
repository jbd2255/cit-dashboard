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

# Google Drive file IDs -set via Railway environment variables (or .env locally)
GDRIVE_5500_ID    = os.environ.get("GDRIVE_5500_ID", "")
GDRIVE_SCHED_C_ID = os.environ.get("GDRIVE_SCHED_C_ID", "")
GDRIVE_SCHED_H_ID = os.environ.get("GDRIVE_SCHED_H_ID", "")


# ---------------------------------------------------------------------------
# Google Drive download helper
# ---------------------------------------------------------------------------
def _gdrive_download(file_id: str, label: str) -> pd.DataFrame:
    """Download a public Google Drive CSV and return a DataFrame."""
    session = requests.Session()

    # Use the newer usercontent endpoint -bypasses virus-scan confirmation page
    url = "https://drive.usercontent.google.com/download"
    params = {"id": file_id, "export": "download", "confirm": "t"}

    print(f"[DOL] Downloading {label} ...", flush=True)
    resp = session.get(url, params=params, timeout=600, stream=True)

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
    print(f"[DOL] {label}: {len(df):,} rows x {len(df.columns)} cols", flush=True)
    print(f"[DOL] {label} columns: {list(df.columns)}", flush=True)
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

        # CITs are DFEs (Direct Filing Entities) — filter on TYPE_DFE_PLAN_ENTITY_CD
        dfe_col  = _find_col(df5, ["TYPE_DFE_PLAN_ENTITY_CD"])
        type_col = _find_col(df5, ["TYPE_PLAN_ENTITY_CD"])

        if dfe_col:
            cit_df = df5[df5[dfe_col].str.strip() == "C"].copy()
        elif type_col:
            cit_df = df5[df5[type_col].str.strip() == "C"].copy()
        else:
            print(f"[DOL] WARNING: entity-type column not found.", flush=True)
            cit_df = df5.copy()

        print(f"[DOL] CIT rows after filter: {len(cit_df):,}", flush=True)

        # Keep only the columns we actually use to reduce RAM
        keep_5500 = []
        for candidates in [
            ["ACK_ID"],
            ["PLAN_NAME", "PLAN_NAME_DFE", "NAME_OF_PLAN"],
            ["SPONSOR_DFE_NAME", "SPONS_DFE_NAME", "SPONS_NAME", "DFE_NAME"],
            ["SPONS_DFE_EIN", "PLAN_EIN", "EIN"],
            ["ADMIN_NAME"],
            ["ADMIN_NAME_SAME_SPON_IND"],
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

        df_c = df_c.reset_index(drop=True)

        # ── F_SCH_H (accountant, custodian, fees) ─────────────────────────
        sch_h_lookup = {}  # ACK_ID -> dict of accountant, custodian, fees
        if GDRIVE_SCHED_H_ID:
            try:
                df_h = _gdrive_download(GDRIVE_SCHED_H_ID, "F_SCH_H_2023")
                ack_col_h = _find_col(df_h, ["ACK_ID"])

                acct_col    = _find_col(df_h, ["ACCOUNTANT_FIRM_NAME"])
                cust_col    = _find_col(df_h, ["FDCRY_TRUSTEE_CUST_NAME"])
                trust_col   = _find_col(df_h, ["FDCRY_TRUST_NAME"])
                trustee_fee = _find_col(df_h, ["TRUSTEE_CUSTODIAL_FEES_AMT"])
                invst_fee   = _find_col(df_h, ["INVST_MGMT_FEES_AMT"])
                admin_fee   = _find_col(df_h, ["CONTRACT_ADMIN_FEES_AMT"])
                audit_fee   = _find_col(df_h, ["IQPA_AUDIT_FEES_AMT"])
                total_admin = _find_col(df_h, ["TOT_ADMIN_EXPENSES_AMT"])
                total_exp   = _find_col(df_h, ["TOT_EXPENSES_AMT"])
                net_assets  = _find_col(df_h, ["NET_ASSETS_EOY_AMT"])

                if ack_col_h:
                    cit_ids = set(cit_df[ack_col_5500].str.strip())
                    df_h_cit = df_h[df_h[ack_col_h].str.strip().isin(cit_ids)]

                    for _, hrow in df_h_cit.iterrows():
                        aid = str(hrow[ack_col_h]).strip()
                        entry = {}

                        # Accountant
                        if acct_col:
                            v = str(hrow[acct_col]).strip()
                            if v and v.lower() != "nan":
                                entry["accountant"] = v

                        # Custodian/Trustee from Schedule H
                        if cust_col:
                            v = str(hrow[cust_col]).strip()
                            if v and v.lower() != "nan":
                                entry["custodian"] = v
                        if "custodian" not in entry and trust_col:
                            v = str(hrow[trust_col]).strip()
                            if v and v.lower() != "nan":
                                entry["custodian"] = v

                        # Fee breakdown
                        entry["trustee_fees"]    = _safe_float(hrow.get(trustee_fee)) if trustee_fee else 0
                        entry["invst_mgmt_fees"] = _safe_float(hrow.get(invst_fee))   if invst_fee   else 0
                        entry["admin_fees"]      = _safe_float(hrow.get(admin_fee))    if admin_fee   else 0
                        entry["audit_fees"]      = _safe_float(hrow.get(audit_fee))    if audit_fee   else 0
                        entry["total_admin_exp"] = _safe_float(hrow.get(total_admin))  if total_admin  else 0
                        entry["total_expenses"]  = _safe_float(hrow.get(total_exp))    if total_exp    else 0
                        entry["net_assets"]      = _safe_float(hrow.get(net_assets))   if net_assets   else 0

                        if entry:
                            sch_h_lookup[aid] = entry

                    print(f"[DOL] Schedule H data for CITs: {len(sch_h_lookup):,} filings", flush=True)
                del df_h
            except Exception as e:
                print(f"[DOL] Schedule H load skipped: {e}", flush=True)
        else:
            print("[DOL] No GDRIVE_SCHED_H_ID set -skipping Schedule H data.", flush=True)

        with _lock:
            _data["cit_df"]        = cit_df
            _data["sched_c_df"]    = df_c
            _data["sch_h_lookup"]  = sch_h_lookup
            _data["cit_count"]     = len(cit_df)
            _data["sched_c_count"] = len(df_c)
            _data["loading"]       = False
            _data["loaded"]        = True

        print("[DOL] Data load complete.", flush=True)

    except Exception as exc:
        import traceback
        traceback.print_exc()
        with _lock:
            _data["loading"] = False
            _data["error"]   = str(exc)
        print(f"[DOL] Data load failed: {exc}", flush=True)


def _start_load():
    if GDRIVE_5500_ID and GDRIVE_SCHED_C_ID:
        threading.Thread(target=_load_dol_data, daemon=True).start()
    else:
        print("[DOL] Skipping data load -- env vars not set.", flush=True)


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
        import math
        v = float(str(val).replace(",", "").strip())
        return 0.0 if math.isnan(v) or math.isinf(v) else v
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
        loaded      = _data["loaded"]
        loading     = _data["loading"]
        error       = _data["error"]
        cit_df      = _data["cit_df"]
        sched_c     = _data["sched_c_df"]
        sch_h       = _data.get("sch_h_lookup", {})

    if not loaded:
        msg = "Data is loading -please wait 60–90 s and try again." if loading \
              else (f"Data load failed: {error}" if error
                    else "DOL data not loaded. Set GDRIVE_5500_ID and GDRIVE_SCHED_C_ID.")
        return jsonify({"error": msg}), 503

    try:
        name_col       = _find_col(cit_df, ["PLAN_NAME"])
        sponsor_col    = _find_col(cit_df, ["SPONSOR_DFE_NAME", "SPONS_DFE_NAME"])
        ack_col        = _find_col(cit_df, ["ACK_ID"])
        ein_col        = _find_col(cit_df, ["SPONS_DFE_EIN"])
        year_col       = _find_col(cit_df, ["FORM_TAX_PRD"])
        admin_col      = _find_col(cit_df, ["ADMIN_NAME"])
        admin_same_col = _find_col(cit_df, ["ADMIN_NAME_SAME_SPON_IND"])

        q_up = q.upper()
        mask = pd.Series(False, index=cit_df.index)
        for col in [name_col, sponsor_col]:
            if col:
                mask |= cit_df[col].str.upper().str.contains(q_up, na=False, regex=False)

        hits = cit_df[mask].head(100)
        if hits.empty:
            return jsonify({"results": [], "total": 0})

        # Schedule C column names (actual 2023 names from DOL)
        sc_ack   = _find_col(sched_c, ["ACK_ID"])
        sc_name  = _find_col(sched_c, ["PROVIDER_OTHER_NAME"])
        sc_code  = _find_col(sched_c, ["PROVIDER_OTHER_SRVC_CODES"])
        sc_dir   = _find_col(sched_c, ["PROVIDER_OTHER_DIRECT_COMP_AMT"])
        sc_indir = _find_col(sched_c, ["PROV_OTHER_TOT_IND_COMP_AMT"])

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
            admin_name   = str(row[admin_col]).strip()  if admin_col   else ""
            admin_same   = str(row[admin_same_col]).strip().upper() if admin_same_col else ""

            # If admin name is empty but indicator says same as sponsor, use sponsor
            if (not admin_name or admin_name.lower() == "nan") and admin_same in ("Y", "YES"):
                admin_name = sponsor_name

            # For CITs, the sponsor/DFE IS the trustee (the bank/trust company)
            trustees = [{"name": sponsor_name, "fee": 0}] if sponsor_name and sponsor_name.lower() != "nan" else []

            # Administrator
            administrators = [{"name": admin_name, "fee": 0}] if admin_name and admin_name.lower() != "nan" else []

            # Schedule H data: accountant, custodian, and fee breakdown
            h = sch_h.get(ack_id, {})

            custodians = []
            if h.get("custodian"):
                custodians.append({"name": h["custodian"], "fee": h.get("trustee_fees", 0)})

            # Fee data from Schedule H
            trustee_fees    = h.get("trustee_fees", 0)
            invst_mgmt_fees = h.get("invst_mgmt_fees", 0)
            admin_fees      = h.get("admin_fees", 0)
            total_admin_exp = h.get("total_admin_exp", 0)
            total_expenses  = h.get("total_expenses", 0)
            net_assets      = h.get("net_assets", 0)

            # Supplement with any Schedule C data (rare for CITs)
            for sp_row in sc_index.get(ack_id, []):
                sp_name  = str(getattr(sp_row, sc_name,  "")).strip() if sc_name  else ""
                svc_code = str(getattr(sp_row, sc_code,  "")).strip() if sc_code  else ""
                fee_direct  = _safe_float(getattr(sp_row, sc_dir,   None)) if sc_dir   else 0
                fee_indirect = _safe_float(getattr(sp_row, sc_indir, None)) if sc_indir else 0
                fee = fee_direct + fee_indirect

                if not sp_name or sp_name.lower() in ("nan", ""):
                    continue

                role = _classify_role(svc_code)
                entry = {"name": sp_name, "fee": fee}
                if role == "custodian" and not custodians:
                    custodians.append(entry)

            results.append({
                "plan_name":        plan_name,
                "sponsor_name":     sponsor_name,
                "ein":              ein,
                "year":             year,
                "ack_id":           ack_id,
                "trustees":         trustees,
                "custodians":       custodians,
                "administrators":   administrators,
                "trustee_fees":     trustee_fees,
                "invst_mgmt_fees":  invst_mgmt_fees,
                "admin_fees":       admin_fees,
                "total_expenses":   total_expenses,
                "net_assets":       net_assets,
            })

        # Summary totals
        totals = {
            "net_assets":      sum(r["net_assets"]      for r in results),
            "trustee_fees":    sum(r["trustee_fees"]    for r in results),
            "invst_mgmt_fees": sum(r["invst_mgmt_fees"] for r in results),
            "admin_fees":      sum(r["admin_fees"]      for r in results),
            "total_expenses":  sum(r["total_expenses"]  for r in results),
        }

        return jsonify({"results": results, "total": int(mask.sum()), "totals": totals})

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.route("/api/summary")
def summary():
    """Aggregate stats: top custodians, trustees, administrators."""
    with _lock:
        loaded  = _data["loaded"]
        loading = _data["loading"]
        cit_df  = _data["cit_df"]
        sch_h   = _data.get("sch_h_lookup", {})

    if not loaded:
        return jsonify({"error": "Data loading" if loading else "Data not loaded"}), 503

    try:
        ack_col        = _find_col(cit_df, ["ACK_ID"])
        sponsor_col    = _find_col(cit_df, ["SPONSOR_DFE_NAME", "SPONS_DFE_NAME"])
        admin_col      = _find_col(cit_df, ["ADMIN_NAME"])
        admin_same_col = _find_col(cit_df, ["ADMIN_NAME_SAME_SPON_IND"])

        # Accumulators: name -> {count, net_assets, total_expenses}
        trustees     = {}
        custodians   = {}
        administrators = {}

        total_net_assets = 0.0
        total_expenses_all = 0.0
        filing_count = len(cit_df)

        for _, row in cit_df.iterrows():
            ack_id       = str(row[ack_col]).strip()     if ack_col     else ""
            sponsor_name = str(row[sponsor_col]).strip()  if sponsor_col else ""
            admin_name   = str(row[admin_col]).strip()    if admin_col   else ""
            admin_same   = str(row[admin_same_col]).strip().upper() if admin_same_col else ""

            if (not admin_name or admin_name.lower() == "nan") and admin_same in ("Y", "YES"):
                admin_name = sponsor_name

            h = sch_h.get(ack_id, {})
            na = h.get("net_assets", 0)
            te = h.get("total_expenses", 0)
            total_net_assets += na
            total_expenses_all += te

            cust_name = h.get("custodian", "")

            def _accum(d, name, na, te):
                if not name or name.lower() == "nan":
                    return
                if name not in d:
                    d[name] = {"count": 0, "net_assets": 0.0, "total_expenses": 0.0}
                d[name]["count"] += 1
                d[name]["net_assets"] += na
                d[name]["total_expenses"] += te

            _accum(trustees, sponsor_name, na, te)
            _accum(custodians, cust_name, na, te)
            _accum(administrators, admin_name, na, h.get("admin_fees", 0))

        def _top(d, limit=25):
            items = sorted(d.items(), key=lambda x: x[1]["net_assets"], reverse=True)
            return [{"name": k, **v} for k, v in items[:limit]]

        return jsonify({
            "filing_count":     filing_count,
            "total_net_assets": total_net_assets,
            "total_expenses":   total_expenses_all,
            "top_trustees":       _top(trustees),
            "top_custodians":     _top(custodians),
            "top_administrators": _top(administrators),
        })

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.route("/api/provider-search")
def provider_search():
    """Reverse lookup: find all CITs that use a given service provider.

    Searches across F_5500 (sponsor/trustee, administrator) and
    Schedule H (custodian, accountant).
    """
    q    = request.args.get("q", "").strip()
    role = request.args.get("role", "").strip().lower()
    if not q:
        return jsonify({"error": "No query provided"}), 400

    with _lock:
        loaded  = _data["loaded"]
        loading = _data["loading"]
        cit_df  = _data["cit_df"]
        sch_h   = _data.get("sch_h_lookup", {})

    if not loaded:
        return jsonify({"error": "Data loading" if loading else "Data not loaded"}), 503

    try:
        ack_col     = _find_col(cit_df, ["ACK_ID"])
        name_col    = _find_col(cit_df, ["PLAN_NAME"])
        sponsor_col = _find_col(cit_df, ["SPONSOR_DFE_NAME", "SPONS_DFE_NAME"])
        admin_col   = _find_col(cit_df, ["ADMIN_NAME"])

        q_up = q.upper()
        results = []

        for _, row in cit_df.iterrows():
            ack_id       = str(row[ack_col]).strip()     if ack_col     else ""
            plan_name    = str(row[name_col]).strip()     if name_col    else ""
            sponsor_name = str(row[sponsor_col]).strip()  if sponsor_col else ""
            admin_name   = str(row[admin_col]).strip()    if admin_col   else ""
            h            = sch_h.get(ack_id, {})
            custodian    = h.get("custodian", "")
            accountant   = h.get("accountant", "")

            # Determine which roles match the query
            matched_role = None
            matched_name = ""
            if q_up in sponsor_name.upper():
                matched_role = "trustee"
                matched_name = sponsor_name
            if q_up in admin_name.upper():
                matched_role = "administrator"
                matched_name = admin_name
            if q_up in custodian.upper():
                matched_role = "custodian"
                matched_name = custodian
            if q_up in accountant.upper():
                matched_role = "accountant"
                matched_name = accountant

            if not matched_role:
                continue

            # Filter by role if specified
            if role and matched_role != role:
                # Check if OTHER roles also match
                skip = True
                if role == "trustee"       and q_up in sponsor_name.upper(): skip = False; matched_role = "trustee"; matched_name = sponsor_name
                if role == "administrator" and q_up in admin_name.upper():   skip = False; matched_role = "administrator"; matched_name = admin_name
                if role == "custodian"     and q_up in custodian.upper():    skip = False; matched_role = "custodian"; matched_name = custodian
                if role == "accountant"    and q_up in accountant.upper():   skip = False; matched_role = "accountant"; matched_name = accountant
                if skip:
                    continue

            fee = 0.0
            if matched_role == "trustee":
                fee = h.get("trustee_fees", 0)
            elif matched_role == "custodian":
                fee = h.get("trustee_fees", 0)
            elif matched_role == "administrator":
                fee = h.get("admin_fees", 0)
            elif matched_role == "accountant":
                fee = h.get("audit_fees", 0)

            results.append({
                "plan_name":     plan_name,
                "ack_id":        ack_id,
                "provider_name": matched_name,
                "service_code":  matched_role,
                "fee":           fee,
            })

            if len(results) >= 200:
                break

        return jsonify({"results": results, "total": len(results)})

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
