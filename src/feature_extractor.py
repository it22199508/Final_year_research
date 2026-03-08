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


def safe_read_csv(path, usecols=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}")
    try:
        return pd.read_csv(path, usecols=usecols)
    except Exception:
        return pd.read_csv(path)


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

    logon = logon[logon["user"].astype(str).isin(selected_set)].copy()
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

    device = device[device["user"].astype(str).isin(selected_set)].copy()
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
    print("Reading http.csv ...")
    http = safe_read_csv(http_path)
    print("http columns:", list(http.columns))

    http = http[http["user"].astype(str).isin(selected_set)].copy()
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
    print("Reading email.csv ...")
    email = safe_read_csv(email_path)
    print("email columns:", list(email.columns))

    email = email[email["user"].astype(str).isin(selected_set)].copy()

    to_col = get_existing_column(email, ["to"])
    cc_col = get_existing_column(email, ["cc"])
    bcc_col = get_existing_column(email, ["bcc"])
    size_col = get_existing_column(email, ["size"])
    attach_col = get_existing_column(email, ["attachment_count", "attachments", "attachment", "att_count"])

    if to_col is None:
        email["_to"] = ""
    else:
        email["_to"] = email[to_col].astype(str)

    if cc_col is None:
        email["_cc"] = ""
    else:
        email["_cc"] = email[cc_col].astype(str)

    if bcc_col is None:
        email["_bcc"] = ""
    else:
        email["_bcc"] = email[bcc_col].astype(str)

    email["_external"] = (
        ~email["_to"].str.contains("DTAA", case=False, na=False)
        | ~email["_cc"].str.contains("DTAA", case=False, na=False)
        | ~email["_bcc"].str.contains("DTAA", case=False, na=False)
    )

    if size_col is not None:
        email["_size"] = pd.to_numeric(email[size_col], errors="coerce").fillna(0)
    else:
        email["_size"] = 0

    if attach_col is not None:
        email["_attachment_count"] = pd.to_numeric(email[attach_col], errors="coerce").fillna(0)
    else:
        email["_attachment_count"] = 0

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
    print("Reading file.csv ...")
    file_df = safe_read_csv(file_path)
    print("file columns:", list(file_df.columns))

    file_df = file_df[file_df["user"].astype(str).isin(selected_set)].copy()

    filename_col = get_existing_column(file_df, ["filename", "file"])

    if filename_col is None:
        file_df["_filename"] = "unknown"
    else:
        file_df["_filename"] = file_df[filename_col].astype(str)

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
        if col in psych.columns:
            psych[col] = pd.to_numeric(psych[col], errors="coerce").fillna(0)
        else:
            psych[col] = 0

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
    df = df.merge(device_agg, on="user", how="left")
    df = df.merge(http_agg, on="user", how="left")
    df = df.merge(email_agg, on="user", how="left")
    df = df.merge(file_agg, on="user", how="left")
    df = df.merge(psych_agg, on="user", how="left")

    for col in df.columns:
        if col != "user":
            df[col] = df[col].fillna(0)

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