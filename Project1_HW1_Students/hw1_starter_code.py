gpt-4o-mini# ============================================================
# COMP 4451: LLMs for Data Science and AI
# Assignment 1 — Few-Shot Prompt Data Challenge
# Student Name(s): ___________________________
# ============================================================
# Instructions: Complete the four sections marked with
# "YOUR CODE HERE" so that the script runs end-to-end without
# error. Do NOT modify any other section of the script.
# ============================================================

import os
import re
import sys
import io
import contextlib
import datetime

import pandas as pd
import numpy as np
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from io import StringIO
from scipy.stats import fisher_exact, chi2_contingency


# ============================================================
# LOGGING  (provided — do not edit)
# ============================================================
# All stdout is tee'd to output.log (append mode) so that
# successive runs accumulate a timestamped history.
# ============================================================

LOG_PATH = "output.log"


class TeeStream:
    """Duplicates writes to the real terminal AND an open file."""
    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log_file = log_file

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()


@contextlib.contextmanager
def tee_to_log(log_path=LOG_PATH):
    """Context manager: tee all stdout to *log_path* (append)."""
    with open(log_path, "a", encoding="utf-8") as f:
        header = (
            f"\n{'#' * 80}\n"
            f"# RUN: {datetime.datetime.now().isoformat()}\n"
            f"# Python: {sys.version.split()[0]}\n"
            f"{'#' * 80}\n"
        )
        f.write(header)
        sys.stdout.write(header)

        tee = TeeStream(sys.stdout, f)
        old = sys.stdout
        sys.stdout = tee
        try:
            yield
        finally:
            sys.stdout = old
            footer = f"\n{'#' * 80}\n# END OF RUN\n{'#' * 80}\n\n"
            f.write(footer)
            old.write(footer)
            old.write(f"[INFO] Output appended to {log_path}\n")


def _capture(fn, *args, **kwargs):
    """Call *fn* capturing stdout; return (result, captured_text)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fn(*args, **kwargs)
    return result, buf.getvalue()


# ============================================================
# SECTION 1 — INITIALIZE OPENAI CLIENT
# ============================================================
# Step 1. Load your .env file.  The helper find_dotenv() is
#         already imported — it searches upward from the
#         working directory automatically.
# Step 2. Read your OpenAI API key from the environment.
# Step 3. Create your OpenAI client and assign it to the
#         variable "client". Print a success message that
#         includes the first 8 characters of the key if the
#         client is created successfully. Print a warning
#         message and assign client = None if the key is not
#         found.
# ============================================================

# YOUR CODE HERE


# ============================================================
# SECTION 2 — LOAD DATA FROM LOCAL DIRECTORY
# ============================================================
# Create a function called "load_and_examine_data" that
# accepts a file path string and returns a pandas DataFrame.
#
# Step 1. Load the CSV into pandas. Use keep_default_na=False
#         and na_values=[''] so that string sentinels such as
#         'NA', 'NULL', and 'missing' are NOT silently
#         converted to NaN at read time.  Assign the result
#         to the variable "df".
# Step 2. Print the dataset dimensions (rows × columns).
# Step 3. Print the list of column names.
# Step 4. Print the first 5 rows of the dataset.
# Step 5. Print the data type of each column.
# Step 6. Return df.
# ============================================================

def load_and_examine_data(file_path):
    # YOUR CODE HERE
    pass


# ============================================================
# SECTIONS 3 & 4 — LLM FUNCTIONS
# ============================================================
# Both functions follow the same structure:
#   a) Guard:   if client is None, print an error, return None.
#   b) Prompt:  build a few-shot prompt string (details below).
#   c) API call: client.chat.response.create() with
#                model="gpt-4o-mini", a system role, a user
#                role, and explicit temperature and max_tokens.
#                Wrap in try/except. Extract the reply with
#                response.choices[0].message.content.
#
# The two functions differ in (b) only — what the prompt asks
# the model to do and how the response is handled.
# ============================================================

# ============================================================
# SECTION 3 — LLM FUNCTION: ANALYZE RAW DATA
# ============================================================
# Signature: llm_statistical_analysis_original(df, file_path='health_dataset.csv')
#
# Data:   Read the raw file into "data_csv" using
#         Path(file_path).read_text().
#         Do NOT use df.to_csv() — reading the file directly
#         preserves string sentinels ('NA', 'NULL', 'missing')
#         so the LLM sees them rather than NaN.
#
# Prompt: Build a few-shot prompt called "analysis_prompt"
#         with TWO examples. Each example should show an
#         input dataset and a structured analysis report
#         covering: missing values (including string
#         sentinels), outliers and invalid values, data-type
#         inconsistencies, formatting inconsistencies, and
#         duplicate records. End with:
#             "NOW ANALYZE THIS DATASET:"
#         followed by {data_csv}.
#         Hint: review health_dataset.csv first so your
#         examples reflect the kinds of issues present.
#
# Output: Assign the reply to "analysis_result", print it,
#         and return it.
# ============================================================

def llm_statistical_analysis_original(df, file_path='health_dataset.csv'):
    # YOUR CODE HERE
    pass


# ============================================================
# SECTION 4 — LLM FUNCTION: DATA PROCESSING & CLEANING
# ============================================================
# Signature: llm_data_cleaning(df, verbose=True)
#            When verbose=True, print shape and rows removed.
#
# Data:   Convert the DataFrame to "data_csv" using
#         df.to_csv(index=False).
#
# Prompt: Build a few-shot prompt called "cleaning_prompt"
#         with TWO examples. Each example should show a
#         dirty input dataset and the corresponding cleaned
#         CSV output. After the examples, include these rules:
#           1. Missing values: impute numeric with median or
#              remove row if multiple values are missing.
#              Leave missing categorical values as-is.
#           2. Outliers: remove rows with extreme or impossible
#              values (age > 120, weight > 200 kg, BP > 250,
#              negatives in positive-only columns).
#           3. Data types: cast to numeric where appropriate;
#              standardise binary categoricals (Yes/No → 1/0).
#           4. Formatting: standardise gender and
#              smoking_status.
#           5. Duplicates: remove duplicate rows,
#              keep first occurrence.
#           6. Date/age integrity (if date_of_birth and
#              enrollment_date are present): compute
#              age_from_dates = (enrollment_date - date_of_birth)
#              in years; flag and remove rows where
#              |age_from_dates - age > 2 years.
#           7. Structured missingness (MNAR): check whether
#              missing values in any column are concentrated
#              in a subgroup - which may indicate nonrandom missingness.
#              If MNAR is detected, do NOT impute with a global median —
#              leave the values as-is or use subgroup-specific
#              medians.
#         End with "NOW CLEAN THIS DATASET:" followed by
#         {data_csv}. Instruct the model to return ONLY the
#         cleaned CSV with headers — no explanations or
#         markdown. Reinforce this in the system role too.
#
# Output: Assign the reply to "cleaned_response". Extract
#         valid CSV lines (non-empty, contain a comma, do not
#         start with '#'), join them, and read into
#         "df_cleaned" via pd.read_csv(StringIO()). If
#         verbose=True, print original shape, cleaned shape,
#         and rows removed. Return df_cleaned.
#         If no valid lines are found, print an error and
#         return None.
# ============================================================

def llm_data_cleaning(df, verbose=True):
    # YOUR CODE HERE
    pass


# ============================================================
# PROGRAMMATIC DETECTION FUNCTIONS  (provided — do not edit)
# ============================================================

# Pre-compiled patterns for detect_data_type_inconsistencies
_FLOAT_PAT = re.compile(r'^\d+\.\d+$')
_INT_PAT   = re.compile(r'^\d+$')


def _has_string_values(series):
    """True when *series* may contain Python str values.

    Covers both legacy 'object' dtype and the newer pandas
    StringDtype.  In pandas >= 2.0, is_string_dtype() returns
    False for 'object' columns, so relying on it alone misses
    string sentinels like 'NA' or 'NULL' when the column also
    contains NaN (which forces 'object' instead of StringDtype).
    """
    return series.dtype == 'object' or pd.api.types.is_string_dtype(series)


def detect_missing_value_variants(df):
    """Detect all representations of missing values."""
    missing_variants = {
        'null_nan':       {},
        'string_na':      {},
        'string_null':    {},
        'string_missing': {},
        'empty_string':   {}
    }
    for col in df.columns:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            missing_variants['null_nan'][col] = null_count

        if _has_string_values(df[col]):
            na_count = df[col].astype(str).str.upper().eq('NA').sum()
            if na_count > 0:
                missing_variants['string_na'][col] = na_count

            null_str_count = df[col].astype(str).str.upper().eq('NULL').sum()
            if null_str_count > 0:
                missing_variants['string_null'][col] = null_str_count

            missing_str_count = df[col].astype(str).str.lower().eq('missing').sum()
            if missing_str_count > 0:
                missing_variants['string_missing'][col] = missing_str_count

            empty_count = (df[col] == '').sum()
            if empty_count > 0:
                missing_variants['empty_string'][col] = empty_count

    return missing_variants


def detect_outliers(df):
    """Detect IQR-based outliers in numeric columns."""
    outliers = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().all():
            continue
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        mask = (df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)
        vals = df.loc[mask, col].tolist()
        if vals:
            outliers[col] = {
                'count': len(vals),
                'values': vals,
                'indices': df.loc[mask].index.tolist()
            }
    return outliers


def detect_invalid_values(df):
    """Detect negative and domain-impossible values."""
    invalid_values = {}

    def _add(col, sub):
        invalid_values.setdefault(col, []).append(sub)

    for col in ['age', 'weight_kg', 'height_cm', 'systolic_bp',
                'diastolic_bp', 'cholesterol', 'bmi']:
        if col not in df.columns:
            continue
        num = pd.to_numeric(df[col], errors='coerce')
        neg = num < 0
        if neg.any():
            _add(col, {'issue': 'negative_values',
                       'count': int(neg.sum()),
                       'values': df.loc[neg, col].tolist()})

    for col, fn, label in [
        ('age',          lambda s: s > 120,              'unrealistic_age'),
        ('weight_kg',    lambda s: s > 300,              'unrealistic_weight'),
        ('systolic_bp',  lambda s: (s > 300) | (s < 40), 'unrealistic_systolic_bp'),
        ('diastolic_bp', lambda s: (s > 150) | (s < 40), 'unrealistic_diastolic_bp'),
    ]:
        if col not in df.columns:
            continue
        num = pd.to_numeric(df[col], errors='coerce')
        mask = fn(num)
        if mask.any():
            _add(col, {'issue': label,
                       'count': int(mask.sum()),
                       'values': df.loc[mask, col].tolist()})

    return invalid_values


def detect_data_type_inconsistencies(df):
    """Detect columns that should be numeric or have mixed types."""
    type_issues = {}

    for col in df.columns:
        issues = []

        if _has_string_values(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
            num = pd.to_numeric(df[col], errors='coerce')
            non_null_orig = df[col].notna().sum()
            non_null_conv = num.notna().sum()
            is_mostly_numeric = (non_null_orig > 0
                                 and non_null_conv / non_null_orig > 0.8)
            if is_mostly_numeric:
                issues.append({'issue': 'should_be_numeric'})
                str_vals = df[col].dropna().astype(str).str.strip()
                float_vals = str_vals[str_vals.str.match(_FLOAT_PAT)]
                int_vals   = str_vals[str_vals.str.match(_INT_PAT)]
                if (len(float_vals) + len(int_vals) > 0
                        and len(int_vals) > len(float_vals)
                        and len(float_vals) >= 1):
                    issues.append({'issue': 'float_notation_mix',
                                   'examples': float_vals.tolist()[:5]})

            types_dist = df[col].apply(lambda x: type(x).__name__).value_counts()
            if len(types_dist) > 1:
                issues.append({'issue': 'mixed_types',
                               'distribution': types_dist.to_dict()})

        if issues:
            type_issues[col] = issues

    return type_issues


def detect_formatting_inconsistencies(df):
    """Detect case inconsistencies in categorical columns."""
    formatting_issues = {}
    try:
        cols = df.select_dtypes(include=['object', 'string']).columns
    except NotImplementedError:
        cols = df.select_dtypes(include=['object']).columns

    for col in cols:
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) > 20:
            continue
        lower_vals = [str(v).lower() for v in unique_vals]
        if len(set(lower_vals)) < len(unique_vals):
            case_groups = {}
            for v in unique_vals:
                case_groups.setdefault(str(v).lower(), []).append(v)
            variations = {k: vs for k, vs in case_groups.items() if len(vs) > 1}
            if variations:
                formatting_issues[col] = [{'issue': 'case_inconsistency',
                                           'variations': variations}]
    return formatting_issues


def detect_duplicate_ids(df, id_column='patient_id'):
    """Detect duplicate patient IDs."""
    info = {'exact_duplicates': int(df.duplicated().sum()), 'duplicate_ids': {}}
    if id_column not in df.columns:
        return info
    counts = df[id_column].value_counts()
    for pid, cnt in counts[counts > 1].items():
        rows = df[df[id_column] == pid]
        info['duplicate_ids'][pid] = {
            'count': cnt,
            'is_exact_duplicate': len(rows.drop_duplicates()) == 1,
            'row_indices': rows.index.tolist()
        }
    return info


def detect_cross_field_inconsistencies(df, bmi_delta_threshold=2.0):
    """Flag rows where recorded BMI contradicts weight and height."""
    issues = []
    if not {'weight_kg', 'height_cm', 'bmi'}.issubset(df.columns):
        return issues
    weight    = pd.to_numeric(df['weight_kg'], errors='coerce')
    height    = pd.to_numeric(df['height_cm'], errors='coerce')
    bmi_given = pd.to_numeric(df['bmi'],       errors='coerce')
    valid = weight.notna() & height.notna() & bmi_given.notna() & (height > 0)
    bmi_calc = weight[valid] / (height[valid] / 100) ** 2
    delta    = (bmi_calc - bmi_given[valid]).abs()
    for idx in delta[delta > bmi_delta_threshold].index:
        pid = df.loc[idx, 'patient_id'] if 'patient_id' in df.columns else idx
        issues.append({
            'index': idx, 'patient_id': pid,
            'issue': 'bmi_weight_height_mismatch',
            'weight_kg': df.loc[idx, 'weight_kg'],
            'height_cm': df.loc[idx, 'height_cm'],
            'bmi_given': df.loc[idx, 'bmi'],
            'bmi_calculated': round(bmi_calc[idx], 2),
            'delta': round(float(delta[idx]), 2)
        })
    return issues


def detect_date_age_integrity(df, age_delta_threshold=2.0):
    """Flag rows where recorded age disagrees with date_of_birth."""
    issues = []
    if not {'age', 'date_of_birth', 'enrollment_date'}.issubset(df.columns):
        return issues
    dob     = pd.to_datetime(df['date_of_birth'],   errors='coerce')
    enroll  = pd.to_datetime(df['enrollment_date'], errors='coerce')
    age_rec = pd.to_numeric(df['age'],              errors='coerce')
    valid     = dob.notna() & enroll.notna() & age_rec.notna()
    age_dates = (enroll[valid] - dob[valid]).dt.days / 365.25
    delta     = (age_dates - age_rec[valid]).abs()
    for idx in delta[delta > age_delta_threshold].index:
        pid = df.loc[idx, 'patient_id'] if 'patient_id' in df.columns else idx
        issues.append({
            'index': idx, 'patient_id': pid,
            'issue': 'date_age_mismatch',
            'age_recorded':    df.loc[idx, 'age'],
            'date_of_birth':   df.loc[idx, 'date_of_birth'],
            'enrollment_date': df.loc[idx, 'enrollment_date'],
            'age_from_dates':  round(float(age_dates[idx]), 1),
            'delta_years':     round(float(delta[idx]), 1),
        })
    return issues


def detect_mnar_patterns(df, p_threshold=0.05, rate_ratio_threshold=3.0):
    """Test whether missingness in any column is associated with a subgroup (MNAR)."""
    SENTINELS = {'NA', 'NULL', 'MISSING', ''}
    findings  = []

    def _is_missing(series):
        base = series.isnull()
        if _has_string_values(series):
            base = base | series.astype(str).str.strip().str.upper().isin(SENTINELS)
        return base

    missing_indicators = {
        col: _is_missing(df[col])
        for col in df.columns if _is_missing(df[col]).sum() > 0
    }

    conditioning = {}
    for col in df.columns:
        if not _has_string_values(df[col]):
            continue
        clean = (df[col].astype(str).str.strip().str.lower()
                 .replace({'yes': '1', 'no': '0',
                           'nan': None, 'none': None, '<na>': None,
                           'na': None, 'null': None, 'missing': None, '': None}))
        if 2 <= clean.dropna().nunique() <= 8:
            conditioning[col] = clean

    if 'age' in df.columns:
        num_age   = pd.to_numeric(df['age'], errors='coerce')
        valid_age = num_age.where((num_age > 0) & (num_age <= 120))
        conditioning['age_group(>50_vs_≤50)'] = valid_age.apply(
            lambda x: '>50' if pd.notna(x) and x > 50
                      else ('≤50' if pd.notna(x) else None)
        )

    norm_cond = {}
    for name, series in conditioning.items():
        lower = series.astype(str).where(series.notna()).str.lower()
        if 2 <= lower.dropna().nunique() <= 8:
            norm_cond[name] = lower.where(series.notna())
        else:
            norm_cond[name] = series

    for miss_col, is_miss in missing_indicators.items():
        for cond_name, cond_series in norm_cond.items():
            if cond_name == miss_col or cond_series.notna().sum() < 10:
                continue
            try:
                valid = cond_series.notna()
                ct = pd.crosstab(cond_series[valid], is_miss[valid])
                if ct.shape[0] < 2 or ct.shape[1] < 2:
                    continue
                is_2x2 = ct.shape == (2, 2)
                if is_2x2:
                    _, p = fisher_exact(ct.values, alternative='two-sided')
                    chi2, min_exp = float('nan'), None
                else:
                    chi2, p, _, expected = chi2_contingency(ct, correction=False)
                    min_exp = float(expected.min())
                    if min_exp < 1:
                        continue
                if p >= p_threshold:
                    continue
                rates = {str(g): round(ct.loc[g].get(True, 0) / ct.loc[g].sum(), 3)
                         for g in ct.index if ct.loc[g].sum()}
                max_r, min_r = max(rates.values()), min(rates.values())
                if max_r == 0:
                    continue
                ratio = max_r / min_r if min_r > 0 else float('inf')
                if ratio >= rate_ratio_threshold or min_r == 0:
                    entry = {
                        'missing_column':         miss_col,
                        'conditioning_column':    cond_name,
                        'test':                   'fisher' if is_2x2 else 'chi2',
                        'p_value':                round(p, 6),
                        'missing_rates_by_group': rates,
                        'rate_ratio':             round(ratio, 2) if ratio != float('inf') else 'inf',
                    }
                    if not is_2x2:
                        entry['chi2'] = round(chi2, 3)
                    findings.append(entry)
            except Exception:
                continue
    return findings


# ============================================================
# PROGRAMMATIC ANALYSIS  (provided — do not edit)
# ============================================================

def programmatic_statistical_analysis(df, dataset_label="Dataset"):
    """Run all detection functions and return a summary dict."""
    print("\n" + "=" * 70)
    print(f"PROGRAMMATIC STATISTICAL ANALYSIS: {dataset_label}")
    print("=" * 70)

    if df is None:
        print("No data available for analysis")
        return None

    results = {}
    total_cells = df.shape[0] * df.shape[1]
    results['basic_info'] = {'shape': df.shape, 'total_cells': total_cells}

    print(f"\nDataset shape: {df.shape[0]} rows × {df.shape[1]} columns")

    print("\n--- MISSING VALUE DETECTION ---")
    missing_variants = detect_missing_value_variants(df)
    total_missing = 0
    for vtype, cols in missing_variants.items():
        if cols:
            print(f"{vtype.replace('_', ' ').title()}:")
            for col, cnt in cols.items():
                print(f"  - {col}: {cnt}")
                total_missing += cnt
    print(f"\nTOTAL MISSING: {total_missing} ({total_missing / total_cells * 100:.1f}%)")
    results['missing_values'] = {'by_variant': missing_variants,
                                 'total_missing': total_missing}

    print("\n--- OUTLIER DETECTION ---")
    outliers = detect_outliers(df)
    for col, info in outliers.items():
        print(f"{col}: {info['count']} outlier(s) — {info['values']}")
    if not outliers:
        print("No outliers detected")
    results['outliers'] = outliers

    print("\n--- INVALID / IMPOSSIBLE VALUES ---")
    invalid_values = detect_invalid_values(df)
    for col, issue_list in invalid_values.items():
        print(f"{col}:")
        for sub in issue_list:
            print(f"  {sub['issue']}: count={sub['count']}, values={sub['values']}")
    if not invalid_values:
        print("No invalid values detected")
    results['invalid_values'] = invalid_values

    print("\n--- DATA TYPE INCONSISTENCIES ---")
    type_issues = detect_data_type_inconsistencies(df)
    for col, issues in type_issues.items():
        print(f"{col}: {[i['issue'] for i in issues]}")
    if not type_issues:
        print("No data type inconsistencies detected")
    results['type_issues'] = type_issues

    print("\n--- FORMATTING INCONSISTENCIES ---")
    formatting_issues = detect_formatting_inconsistencies(df)
    for col, issues in formatting_issues.items():
        for issue in issues:
            if 'variations' in issue:
                print(f"{col} case variations:")
                for std, variants in issue['variations'].items():
                    print(f"  '{std}' appears as: {variants}")
    if not formatting_issues:
        print("No formatting inconsistencies detected")
    results['formatting_issues'] = formatting_issues

    print("\n--- DUPLICATE DETECTION ---")
    dup_info = detect_duplicate_ids(df)
    print(f"Exact duplicate rows: {dup_info['exact_duplicates']}")
    if dup_info['duplicate_ids']:
        for pid, info in dup_info['duplicate_ids'].items():
            print(f"  {pid}: {info['count']} times (rows {info['row_indices']})")
    else:
        print("No duplicate IDs found")
    results['duplicates'] = dup_info

    print("\n--- CROSS-FIELD CONSISTENCY (BMI) ---")
    cross_field = detect_cross_field_inconsistencies(df)
    if cross_field:
        print(f"BMI/weight/height mismatches: {len(cross_field)} row(s)")
        for item in cross_field:
            print(f"  {item['patient_id']}: weight={item['weight_kg']}, "
                  f"height={item['height_cm']}, bmi_given={item['bmi_given']}, "
                  f"bmi_calc={item['bmi_calculated']}, delta={item['delta']}")
    else:
        print("No cross-field inconsistencies detected")
    results['cross_field_issues'] = cross_field

    print("\n--- DATE / AGE INTEGRITY ---")
    date_age = detect_date_age_integrity(df)
    if date_age:
        print(f"Date/age mismatches: {len(date_age)} row(s)")
        for item in date_age:
            print(f"  {item['patient_id']}: age_recorded={item['age_recorded']}, "
                  f"age_from_dates={item['age_from_dates']}, "
                  f"delta={item['delta_years']} yrs")
    else:
        print("No date/age integrity violations detected")
    results['date_age_violations'] = date_age

    print("\n--- MNAR: STRUCTURED MISSINGNESS ---")
    mnar = detect_mnar_patterns(df)
    if mnar:
        print(f"MNAR patterns: {len(mnar)} finding(s)")
        for f in mnar:
            test_str = (f"χ²={f['chi2']}, " if 'chi2' in f else "") + f"test={f['test']}"
            print(f"  '{f['missing_column']}' ~ '{f['conditioning_column']}': "
                  f"{test_str}, p={f['p_value']}, rates={f['missing_rates_by_group']}")
    else:
        print("No significant MNAR patterns detected")
    results['mnar_findings'] = mnar

    # total_invalid_issues counts individual *values*, not findings
    total_invalid_values = sum(
        sub['count'] for il in invalid_values.values() for sub in il
    )

    summary = {
        'total_rows':               df.shape[0],
        'total_columns':            df.shape[1],
        'missing_values_total':     total_missing,
        'missing_percent':          total_missing / total_cells * 100,
        'total_outliers':           sum(i['count'] for i in outliers.values()),
        'total_invalid_issues':     total_invalid_values,
        'type_issue_columns':       len(type_issues),
        'formatting_issue_columns': len(formatting_issues),
        'duplicate_ids':            len(dup_info['duplicate_ids']),
        'cross_field_issues':       len(cross_field),
        'date_age_violations':      len(date_age),
        'mnar_patterns':            len(mnar),
    }

    print("\n--- SUMMARY ---")
    for label, key in [
        ("Missing values",           'missing_values_total'),
        ("Outliers",                 'total_outliers'),
        ("Invalid values",           'total_invalid_issues'),
        ("Type issues (columns)",    'type_issue_columns'),
        ("Formatting issues (cols)", 'formatting_issue_columns'),
        ("Duplicate IDs",            'duplicate_ids'),
        ("Cross-field issues",       'cross_field_issues'),
        ("Date/age violations",      'date_age_violations'),
        ("MNAR patterns",            'mnar_patterns'),
    ]:
        print(f"  {label:<28} {summary[key]}")

    results['summary'] = summary
    return results


# ============================================================
# PROGRAMMATIC CLEANING  (provided — do not edit)
# ============================================================

def programmatic_data_cleaning(df):
    """Deterministic rule-based cleaning. Returns (df_cleaned, log, rows_removed)."""
    df = df.copy()
    log = []
    original_rows = len(df)

    # 1. String sentinels → NaN
    for col in df.columns:
        if _has_string_values(df[col]):
            mask = df[col].astype(str).str.strip().str.upper().isin({'NA', 'NULL', 'MISSING'})
            if mask.any():
                df.loc[mask, col] = np.nan
                log.append(f"  {col}: {int(mask.sum())} string sentinel(s) → NaN")

    # 2. Cast numeric columns
    for col in ['age', 'weight_kg', 'height_cm', 'systolic_bp', 'cholesterol', 'bmi']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 3. Standardise diabetes
    if 'diabetes' in df.columns:
        df['diabetes'] = df['diabetes'].replace({'Yes': '1', 'No': '0'})
        df['diabetes'] = pd.to_numeric(df['diabetes'], errors='coerce').astype('Int64')
        log.append("  diabetes: 'Yes'/'No' → 1/0, cast to integer")

    # 4. Remove outlier / impossible rows
    drop_mask = pd.Series(False, index=df.index)
    for col, fn in {
        'age':          lambda s: s.gt(120),
        'weight_kg':    lambda s: s.gt(300),
        'systolic_bp':  lambda s: s.gt(300) | s.lt(40),
        'diastolic_bp': lambda s: s.gt(150) | s.lt(40),
        'bmi':          lambda s: s.gt(80),
        'cholesterol':  lambda s: s.lt(0),
    }.items():
        if col in df.columns:
            drop_mask |= fn(pd.to_numeric(df[col], errors='coerce')).fillna(False)
    if drop_mask.sum():
        log.append(f"  Removed {int(drop_mask.sum())} row(s) with outlier/impossible values")
        df = df[~drop_mask].copy()

    # 5. Remove date/age violations
    if {'age', 'date_of_birth', 'enrollment_date'}.issubset(df.columns):
        dob    = pd.to_datetime(df['date_of_birth'],   errors='coerce')
        enroll = pd.to_datetime(df['enrollment_date'], errors='coerce')
        age_r  = pd.to_numeric(df['age'], errors='coerce')
        valid  = dob.notna() & enroll.notna() & age_r.notna()
        delta  = ((enroll[valid] - dob[valid]).dt.days / 365.25 - age_r[valid]).abs()
        viol   = delta[delta > 2].index
        if len(viol):
            pids = df.loc[viol, 'patient_id'].tolist() if 'patient_id' in df.columns else list(viol)
            log.append(f"  Removed {len(viol)} row(s) with date/age violations: {pids}")
            df = df.drop(index=viol).copy()

    # 6. Remove duplicate patient_ids
    if 'patient_id' in df.columns:
        n_before = len(df)
        dup_ids  = df[df.duplicated('patient_id', keep=False)]['patient_id'].unique().tolist()
        df = df.drop_duplicates('patient_id', keep='first').reset_index(drop=True)
        if n_before - len(df):
            log.append(f"  Removed {n_before - len(df)} duplicate row(s) for IDs: {dup_ids} (kept first)")

    # 7. Median-impute numeric (derive MNAR skip-set from data)
    mnar_findings = detect_mnar_patterns(df)
    mnar_skip = {f['missing_column'] for f in mnar_findings}

    imputed_num = []
    for col in df.select_dtypes(include=[np.number]).columns:
        n_miss = int(df[col].isnull().sum())
        if n_miss:
            if col in mnar_skip:
                log.append(f"  {col}: {n_miss} missing value(s) flagged as MNAR — not imputed")
            else:
                med = df[col].median()
                df[col] = df[col].fillna(med)
                imputed_num.append(f"{col}({n_miss}→{med:.1f})")
    if imputed_num:
        log.append(f"  Median-imputed: {', '.join(imputed_num)}")

    # 8. Standardise categorical formatting
    if 'gender' in df.columns:
        g_map = {'M':'M','m':'M','male':'M','Male':'M',
                 'F':'F','f':'F','female':'F','Female':'F'}
        df['gender'] = df['gender'].map(
            lambda x: g_map.get(str(x).strip(), x) if pd.notna(x) else x)
        log.append("  gender: standardised to M/F")

    if 'smoking_status' in df.columns:
        s_map = {'Never':'Never','never':'Never',
                 'Former':'Former','former':'Former',
                 'Current':'Current','current':'Current'}
        df['smoking_status'] = df['smoking_status'].map(
            lambda x: s_map.get(str(x).strip(), x) if pd.notna(x) else x)
        log.append("  smoking_status: standardised to Never/Former/Current")

    return df, log, original_rows - len(df)


# ============================================================
# COMPARISON REPORT  (provided — do not edit)
# ============================================================

def print_comparison_report(orig_summary, prog_summary, llm_summary,
                             prog_log, prog_rows_removed,
                             llm_rows_removed):
    SEP  = "=" * 70
    LINE = "-" * 70

    def _row(label, o, p, l, flag=""):
        print(f"  {label:<26} {str(o):>8}    {str(p):>13}    {str(l):>11}{flag}")

    # STEP 1
    print(f"\n{SEP}")
    print("STEP 1 — ACTUAL DATA CONCERNS  (original data, programmatic)")
    print(SEP)
    print(f"  {'Issue Category':<28} {'Count':>10}")
    print(f"  {LINE[:42]}")
    print(f"  {'Rows':<28} {orig_summary['total_rows']:>10}")
    print(f"  {LINE[:42]}")
    for label, key in [
        ("Missing values",      'missing_values_total'),
        ("Outliers (IQR)",      'total_outliers'),
        ("Invalid values",      'total_invalid_issues'),
        ("Type inconsistencies",'type_issue_columns'),
        ("Formatting issues",   'formatting_issue_columns'),
        ("Duplicate IDs",       'duplicate_ids'),
        ("Cross-field (BMI)",   'cross_field_issues'),
        ("Date/age violations", 'date_age_violations'),
        ("MNAR patterns",       'mnar_patterns'),
    ]:
        print(f"  {label:<28} {orig_summary[key]:>10}")

    # STEP 2
    o, p, l = orig_summary, prog_summary, llm_summary

    COMPARISON_KEYS = [
        ("Missing values",       'missing_values_total'),
        ("Outliers (IQR)",       'total_outliers'),
        ("Invalid values",       'total_invalid_issues'),
        ("Type inconsistencies", 'type_issue_columns'),
        ("Formatting issues",    'formatting_issue_columns'),
        ("Duplicate IDs",        'duplicate_ids'),
        ("Cross-field (BMI)",    'cross_field_issues'),
        ("Date/age violations",  'date_age_violations'),
        ("MNAR patterns",        'mnar_patterns'),
    ]

    print(f"\n{SEP}")
    print("STEP 2 — THREE-WAY DATA QUALITY COMPARISON")
    print(SEP)
    print(f"  {'Issue Category':<26} {'ORIGINAL':>8}    {'PROG-CLEANED':>13}    {'LLM-CLEANED':>11}")
    print(f"  {LINE}")

    _row("Rows", o['total_rows'], p['total_rows'], l['total_rows'])
    print(f"  {LINE}")

    for label, key in COMPARISON_KEYS:
        ov, pv, lv = o.get(key, 0), p.get(key, 0), l.get(key, 0)
        flag = " ← LLM missed" if lv > pv else ""
        _row(label, ov, pv, lv, flag)
    print(f"  {LINE}")

    # STEP 3
    misses = [(lbl, p.get(k, 0), l.get(k, 0))
              for lbl, k in COMPARISON_KEYS
              if l.get(k, 0) > p.get(k, 0)]
    print(f"\n{SEP}")
    print("STEP 3 — WHAT THE LLM MISSED")
    print(SEP)
    if misses:
        for lbl, pv, lv in misses:
            print(f"  ✗ {lbl}: {lv} remaining (programmatic cleaned to {pv})")
    else:
        print("  ✓ LLM resolved all categories that programmatic cleaning resolved")

    # STEP 4
    print(f"\n{SEP}")
    print("STEP 4 — CLEANING ACTION LOG")
    print(SEP)
    print(f"\n  Programmatic cleaning  ({prog_rows_removed} rows removed from {o['total_rows']}):")
    for entry in prog_log:
        print(f"  {entry}")
    print(f"\n  LLM cleaning  ({llm_rows_removed} rows removed from {o['total_rows']}):")
    print( "  (LLM reasoning not captured — see raw LLM output for details)")

    # STEP 5
    print(f"\n{SEP}")
    print("STEP 5 — METHOD COMPARISON")
    print(SEP)

    def _delta(key):
        return max(0, o.get(key, 0) - p.get(key, 0)), \
               max(0, o.get(key, 0) - l.get(key, 0))

    print(f"\n  {'':30} {'Programmatic':>15}   {'LLM':>10}")
    print(f"  {LINE[:58]}")
    print(f"  {'Rows removed':<30} {prog_rows_removed:>15}   {llm_rows_removed:>10}")

    for lbl, key in [
        ("Missing values resolved",      'missing_values_total'),
        ("Invalid values resolved",      'total_invalid_issues'),
        ("Duplicates resolved",          'duplicate_ids'),
        ("Date/age violations resolved", 'date_age_violations'),
        ("MNAR patterns resolved",       'mnar_patterns'),
    ]:
        dp, dl = _delta(key)
        print(f"  {lbl:<30} {dp:>15}   {dl:>10}")


# ============================================================
# MAIN  (provided — do not edit)
# ============================================================

def main():
    """Orchestrate the full pipeline and print the comparison report."""
    SEP = "=" * 70

    print(SEP)
    print("COMPREHENSIVE DATA QUALITY ANALYSIS")
    print("Comparing Programmatic vs LLM Cleaning Methods")
    print(SEP)

    # Load data
    print("\nLoading original data...")
    df_original, _ = _capture(load_and_examine_data, 'health_dataset.csv')
    if df_original is None:
        print("  ERROR: load_and_examine_data returned None — check Section 2.")
        return
    n_orig = len(df_original)
    print(f"  {n_orig} rows × {df_original.shape[1]} columns loaded from health_dataset.csv")

    # LLM cleaning
    print("\nRunning LLM cleaning...")
    df_llm_cleaned = llm_data_cleaning(df_original.copy(), verbose=False)
    if df_llm_cleaned is None:
        print("  ERROR: LLM cleaning failed — check Section 4.")
        return
    n_llm = len(df_llm_cleaned)
    print(f"  Done.  {n_orig} rows → {n_llm} rows "
          f"({n_orig - n_llm} removed, {(n_orig - n_llm) / n_orig * 100:.1f}%)")

    # Programmatic cleaning
    print("\nRunning programmatic cleaning...")
    df_prog_cleaned, prog_log, prog_rows_removed = programmatic_data_cleaning(
        df_original.copy()
    )
    n_prog = len(df_prog_cleaned)
    print(f"  Done.  {n_orig} rows → {n_prog} rows "
          f"({prog_rows_removed} removed, {prog_rows_removed / n_orig * 100:.1f}%)")

    # Analyse all three datasets (capture detail for log)
    print("\nRunning programmatic analysis on all three datasets...")
    labels_and_dfs = [
        ("ORIGINAL DATA",     df_original),
        ("PROG-CLEANED DATA", df_prog_cleaned),
        ("LLM-CLEANED DATA",  df_llm_cleaned),
    ]
    analyses = {}
    detail_texts = []
    for label, frame in labels_and_dfs:
        result, text = _capture(programmatic_statistical_analysis,
                                frame.copy(), label)
        analyses[label] = result
        detail_texts.append(text)
    print("  Done.")

    print(f"\n{SEP}")
    print("DETAILED PER-DATASET ANALYSIS  (for reference / log)")
    print(SEP)
    for text in detail_texts:
        print(text)

    # Print report
    print_comparison_report(
        orig_summary      = analyses["ORIGINAL DATA"]['summary'],
        prog_summary      = analyses["PROG-CLEANED DATA"]['summary'],
        llm_summary       = analyses["LLM-CLEANED DATA"]['summary'],
        prog_log          = prog_log,
        prog_rows_removed = prog_rows_removed,
        llm_rows_removed  = n_orig - n_llm,
    )

    # Export
    df_prog_cleaned.to_csv('prog_cleaned_health_data.csv', index=False)
    df_llm_cleaned.to_csv('llm_cleaned_health_data.csv',   index=False)
    print(f"\n{SEP}")
    print("Exports written:")
    print(f"  prog_cleaned_health_data.csv  ({n_prog} rows)")
    print(f"  llm_cleaned_health_data.csv   ({n_llm} rows)")
    print(SEP)


if __name__ == "__main__":
    with tee_to_log(LOG_PATH):
        main()
