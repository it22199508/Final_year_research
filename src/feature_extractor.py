# src/feature_extractor.py

import os
import warnings
import pandas as pd
import numpy as np

from config import (
    DATA_DIR,
    MODELS_DIR,
    TRAIN_USER_OFFSET,
    TRAIN_USER_COUNT,
    TEST_USER_OFFSET,
    TEST_USER_COUNT,
    TRAIN_FEATURES_FILE,
    TEST_FEATURES_FILE,
)
from privacy_utils import sha256_hex

warnings.filterwarnings("ignore")

CHUNK_SIZE = 200000


def safe_read_csv(path, usecols=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path, usecols=usecols, low_memory=False)


def read_filtered_chunks(path, user_col, selected_set, usecols=None):
    chunks = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=CHUNK_SIZE, low_memory=False):
        if user_col in chunk.columns:
            chunk[user_col] = chunk[user_col].astype(str)
            chunk = chunk[chunk[user_col].isin(selected_set)]
            if not chunk.empty:
                chunks.append(chunk)
    if chunks:
        return pd.concat(chunks, ignore_index=True)
    return pd.DataFrame(columns=usecols if usecols else [])


def parse_dt(series):
    return pd.to_datetime(series, errors="coerce")


def is_after_hours(hour):
    if pd.isna(hour):
        return False
    hour = int(hour)
    return hour >= 18 or hour <= 6


def get_existing_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def build_features_for_users(selected_users, split_name):
    print(f"\n📦 Building features for {split_name} users...")
    selected_set = set(selected_users)

    logon_path = os.path.join(DATA_DIR, "logon.csv")
    device_path = os.path.join(DATA_DIR, "device.csv")
    http_path = os.path.join(DATA_DIR, "http.csv")
    email_path = os.path.join(DATA_DIR, "email.csv")
    file_path = os.path.join(DATA_DIR, "file.csv")
    psych_path = os.path.join(DATA_DIR, "psychometric.csv")

    # ---------------- LOGON ----------------
    print("Reading logon.csv ...")
    logon = safe_read_csv(logon_path)
    print("logon columns:", list(logon.columns))

    logon["user"] = logon["user"].astype(str)
    logon = logon[logon["user"].isin(selected_set)].copy()
    logon["_dt"] = parse_dt(logon["date"])
    logon["_hour"] = logon["_dt"].dt.hour
    logon["_after_hours"] = logon["_hour"].apply(is_after_hours)
    logon["_is_logon"] = logon["activity"].astype(str).str.lower().str.contains("logon")

    assigned_pc = (
        logon[logon["_is_logon"]]
        .groupby(["user", "pc"])
        .size()
        .reset_index(name="count")
        .sort_values(["user", "count"], ascending=[True, False])
        .drop_duplicates(subset=["user"])
        .rename(columns={"pc": "assigned_pc"})
    )

    last_seen_pc = (
        logon.sort_values("_dt")
        .dropna(subset=["_dt"])
        .groupby("user")
        .tail(1)[["user", "pc"]]
        .rename(columns={"pc": "last_seen_pc"})
    )

    logon_agg = (
        logon.groupby("user")
        .agg(
            total_logons=("_is_logon", "sum"),
            after_hours_logons=("_after_hours", "sum"),
            unique_pcs=("pc", "nunique"),
        )
        .reset_index()
    )

    logon_agg["after_hours_ratio"] = np.where(
        logon_agg["total_logons"] > 0,
        logon_agg["after_hours_logons"] / logon_agg["total_logons"],
        0.0,
    )

    logon_with_assigned = logon.merge(
        assigned_pc[["user", "assigned_pc"]],
        on="user",
        how="left"
    )
    logon_with_assigned["foreign_pc"] = (
        logon_with_assigned["pc"].astype(str) != logon_with_assigned["assigned_pc"].astype(str)
    )

    foreign_pc_agg = (
        logon_with_assigned.groupby("user")
        .agg(foreign_pc_logons=("foreign_pc", "sum"))
        .reset_index()
    )

    logon_agg = logon_agg.merge(foreign_pc_agg, on="user", how="left")
    logon_agg["foreign_pc_logons"] = logon_agg["foreign_pc_logons"].fillna(0)
    logon_agg["foreign_pc_ratio"] = np.where(
        logon_agg["total_logons"] > 0,
        logon_agg["foreign_pc_logons"] / logon_agg["total_logons"],
        0.0,
    )

    # ---------------- DEVICE ----------------
    print("Reading device.csv ...")
    device = safe_read_csv(device_path)
    print("device columns:", list(device.columns))

    device["user"] = device["user"].astype(str)
    device = device[device["user"].isin(selected_set)].copy()
    device["_connect"] = device["activity"].astype(str).str.lower().str.contains("connect")

    device_agg = (
        device.groupby("user")
        .agg(
            usb_connects=("_connect", "sum"),
            total_device_events=("id", "count"),
        )
        .reset_index()
    )

    # ---------------- HTTP ----------------
    print("Reading http.csv in chunks ...")
    http = read_filtered_chunks(
        http_path,
        user_col="user",
        selected_set=selected_set,
        usecols=["id", "date", "user", "pc", "url", "content"]
    )
    print("http rows loaded:", len(http))

    if http.empty:
        http_agg = pd.DataFrame(columns=["user", "total_http", "http_uploads", "unique_domains", "upload_ratio"])
    else:
        http["url"] = http["url"].astype(str)
        http["_is_upload"] = http["url"].str.lower().str.contains(
            "upload|post|dropbox|drive|send|attach",
            na=False
        )
        http["_domain"] = http["url"].str.extract(r"https?://([^/]+)", expand=False)
        http["_domain"] = http["_domain"].fillna(http["url"].str.split("/").str[0])

        http_agg = (
            http.groupby("user")
            .agg(
                total_http=("id", "count"),
                http_uploads=("_is_upload", "sum"),
                unique_domains=("_domain", "nunique"),
            )
            .reset_index()
        )

        http_agg["upload_ratio"] = np.where(
            http_agg["total_http"] > 0,
            http_agg["http_uploads"] / http_agg["total_http"],
            0.0,
        )

    # ---------------- EMAIL ----------------
    print("Reading email.csv in chunks ...")
    email = read_filtered_chunks(
        email_path,
        user_col="user",
        selected_set=selected_set,
        usecols=["id", "date", "user", "pc", "to", "cc", "bcc", "from", "size", "attachments", "content"]
    )
    print("email rows loaded:", len(email))

    if email.empty:
        email_agg = pd.DataFrame(columns=[
            "user", "total_emails", "external_emails", "total_attachments",
            "avg_email_size", "external_email_ratio"
        ])
    else:
        to_col = get_existing_column(email, ["to"])
        cc_col = get_existing_column(email, ["cc"])
        bcc_col = get_existing_column(email, ["bcc"])
        size_col = get_existing_column(email, ["size"])
        attach_col = get_existing_column(email, ["attachment_count", "attachments", "attachment", "att_count"])

        email["_to"] = email[to_col].astype(str) if to_col else ""
        email["_cc"] = email[cc_col].astype(str) if cc_col else ""
        email["_bcc"] = email[bcc_col].astype(str) if bcc_col else ""

        email["_external"] = (
            ~email["_to"].str.contains("DTAA", case=False, na=False)
            | ~email["_cc"].str.contains("DTAA", case=False, na=False)
            | ~email["_bcc"].str.contains("DTAA", case=False, na=False)
        )

        email["_size"] = pd.to_numeric(email[size_col], errors="coerce").fillna(0) if size_col else 0
        email["_attachment_count"] = pd.to_numeric(email[attach_col], errors="coerce").fillna(0) if attach_col else 0

        email_agg = (
            email.groupby("user")
            .agg(
                total_emails=("id", "count"),
                external_emails=("_external", "sum"),
                total_attachments=("_attachment_count", "sum"),
                avg_email_size=("_size", "mean"),
            )
            .reset_index()
        )

        email_agg["external_email_ratio"] = np.where(
            email_agg["total_emails"] > 0,
            email_agg["external_emails"] / email_agg["total_emails"],
            0.0,
        )

    # ---------------- FILE ----------------
    print("Reading file.csv in chunks ...")
    file_df = read_filtered_chunks(
        file_path,
        user_col="user",
        selected_set=selected_set,
        usecols=["id", "date", "user", "pc", "filename", "content"]
    )
    print("file rows loaded:", len(file_df))

    if file_df.empty:
        file_agg = pd.DataFrame(columns=["user", "copied_files", "unique_filenames", "copy_to_removable"])
    else:
        filename_col = get_existing_column(file_df, ["filename", "file"])
        file_df["_filename"] = file_df[filename_col].astype(str) if filename_col else "unknown"

        file_agg = (
            file_df.groupby("user")
            .agg(
                copied_files=("id", "count"),
                unique_filenames=("_filename", "nunique"),
            )
            .reset_index()
        )

        file_agg["copy_to_removable"] = file_agg["copied_files"]

    # ---------------- PSYCHOMETRIC ----------------
    print("Reading psychometric.csv ...")
    psych = safe_read_csv(psych_path)
    print("psychometric columns:", list(psych.columns))

    user_id_col = get_existing_column(psych, ["user_id", "user"])
    if user_id_col is None:
        raise ValueError("psychometric.csv does not contain user_id/user column")

    psych["user"] = psych[user_id_col].astype(str)
    psych = psych[psych["user"].isin(selected_set)].copy()

    for col in ["O", "C", "E", "A", "N"]:
        psych[col] = pd.to_numeric(psych[col], errors="coerce").fillna(0) if col in psych.columns else 0

    psych["psych_risk"] = (
        (10 - psych["C"]) * 0.35 +
        psych["N"] * 0.35 +
        (10 - psych["A"]) * 0.15 +
        (10 - psych["O"]) * 0.15
    )

    psych_agg = psych[["user", "O", "C", "E", "A", "N", "psych_risk"]].copy()

    # ---------------- MERGE ----------------
    base = pd.DataFrame({"user": list(selected_users)})

    df = base.merge(logon_agg, on="user", how="left")
    df = df.merge(assigned_pc, on="user", how="left")
    df = df.merge(last_seen_pc, on="user", how="left")
    df = df.merge(device_agg, on="user", how="left")
    df = df.merge(http_agg, on="user", how="left")
    df = df.merge(email_agg, on="user", how="left")
    df = df.merge(file_agg, on="user", how="left")
    df = df.merge(psych_agg, on="user", how="left")

    for col in df.columns:
        if col not in ["user", "assigned_pc", "last_seen_pc"]:
            df[col] = df[col].fillna(0)

    df["assigned_pc"] = df["assigned_pc"].fillna("UNKNOWN")
    df["last_seen_pc"] = df["last_seen_pc"].fillna("UNKNOWN")

    # ---------------- DERIVED FEATURES ----------------
    df["activity_volume"] = (
        df.get("total_logons", 0)
        + df.get("total_device_events", 0)
        + df.get("total_http", 0)
        + df.get("total_emails", 0)
        + df.get("copied_files", 0)
    )

    df["device_intensity"] = np.where(
        df["activity_volume"] > 0,
        df.get("usb_connects", 0) / df["activity_volume"],
        0.0,
    )

    df["exfiltration_score"] = np.clip(
        0.35 * df.get("external_email_ratio", 0)
        + 0.25 * df.get("upload_ratio", 0)
        + 0.25 * np.tanh(df.get("copy_to_removable", 0) / 10.0)
        + 0.15 * df.get("foreign_pc_ratio", 0),
        0,
        1,
    )

    df["suspicious_activity"] = np.clip(
        0.30 * df.get("after_hours_ratio", 0)
        + 0.20 * df.get("foreign_pc_ratio", 0)
        + 0.20 * df.get("device_intensity", 0)
        + 0.15 * df.get("external_email_ratio", 0)
        + 0.15 * df.get("upload_ratio", 0),
        0,
        1,
    )

    df["temporal_anomaly"] = df.get("after_hours_ratio", 0)

    # ---------------- PRIVACY ----------------
    df["user_id"] = df["user"].astype(str).apply(sha256_hex)
    df.drop(columns=["user"], inplace=True)

    cols = ["user_id"] + [c for c in df.columns if c != "user_id"]
    df = df[cols]

    print(f"✅ {split_name} features built: {len(df)} users")
    return df


def main():
    print("=" * 80)
    print("CERT r4.2 FEATURE EXTRACTION")
    print("=" * 80)

    os.makedirs(MODELS_DIR, exist_ok=True)

    logon_path = os.path.join(DATA_DIR, "logon.csv")
    logon = safe_read_csv(logon_path, usecols=["user"])
    all_users = sorted(logon["user"].dropna().astype(str).unique())

    print(f"Total unique users found: {len(all_users)}")

    train_users = all_users[TRAIN_USER_OFFSET:TRAIN_USER_OFFSET + TRAIN_USER_COUNT]
    test_users = all_users[TEST_USER_OFFSET:TEST_USER_OFFSET + TEST_USER_COUNT]

    print(f"Train users selected: {len(train_users)}")
    print(f"Test users selected : {len(test_users)}")

    train_df = build_features_for_users(train_users, "TRAIN")
    test_df = build_features_for_users(test_users, "TEST")

    train_df.to_csv(TRAIN_FEATURES_FILE, index=False)
    test_df.to_csv(TEST_FEATURES_FILE, index=False)

    print("\n" + "=" * 80)
    print("✅ FEATURE EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"Saved train features: {TRAIN_FEATURES_FILE}")
    print(f"Saved test features : {TEST_FEATURES_FILE}")


if __name__ == "__main__":
    main()