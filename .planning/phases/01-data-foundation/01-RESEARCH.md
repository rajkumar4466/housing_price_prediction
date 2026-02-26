# Phase 1: Data Foundation - Research

**Researched:** 2026-02-26
**Domain:** NJ housing data generation, JSONL dataset creation, shared prompt formatting
**Confidence:** MEDIUM-HIGH

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| DATA-01 | Generate NJ housing dataset with 7 features (bedrooms, bathrooms, sqft, lot size, year built, zip code, property type) | Section: Standard Stack (pandas, numpy, scikit-learn); Architecture Patterns: Feature ranges, synthetic generation |
| DATA-02 | Create train/validation/test splits (70/15/15) | Section: Architecture Patterns (train_test_split stratified by price range); Code Examples |
| DATA-03 | Implement shared `format_prompt()` in `lambda/prompt_utils.py` importable from notebooks and Lambda handler | Section: Architecture Patterns (Shared Prompt Module); Don't Hand-Roll; Code Examples |
| DATA-04 | Generate synthetic data with county-level NJ price distributions (log-normal + multipliers); at least 30% from public NJ datasets; document schema | Section: NJ County Price Statistics table; Architecture Patterns (Synthetic Data Generation); Common Pitfalls (Synthetic Distribution Mismatch) |
</phase_requirements>

---

## Summary

Phase 1 creates the training data foundation for the entire ML pipeline. Its primary deliverables are `train.jsonl`, `val.jsonl`, `test.jsonl` (70/15/15 splits), and `lambda/prompt_utils.py` containing `format_prompt()`. The data must blend real public NJ records with synthetic records that use county-level log-normal price distributions. Getting this phase right prevents the highest-cost failure in the entire project: synthetic data with unrealistic distributions that forces a full retrain later.

The public data source is the NJ Division of Taxation SR1A Sales File (available for 2020-2026 at nj.gov/treasury/taxation/lpt/statdata.shtml). The SR1A files are ZIP archives with a fixed-width or delimited layout that requires parsing the official NJ SR1A File Layout PDF. The files contain property sale price, sale date, class (property type), and assessment data but do NOT consistently contain bedrooms, bathrooms, and square footage — these fields are partially present and vary by county submission quality. This means SR1A data can supply sale prices and property type and zip code but may need to be supplemented with MODIV Property Assessment lists for structural details. Plan to use SR1A as the real-data backbone and fill missing structural fields synthetically per county distribution.

The synthetic generation strategy is: generate price as log-normal with county-specific mu and sigma derived from the 2024-2025 NJ county median price table. Then back-calculate sqft, bedrooms, bathrooms, lot size, and year built as correlated features consistent with that price point. This is more realistic than generating features independently and computing price from them. Validate the final synthetic distribution histogram against the county medians before splitting.

**Primary recommendation:** Define `format_prompt()` in `lambda/prompt_utils.py` first, write 5 test cases that import it from a notebook path, confirm it works, then generate data that uses it. This ensures the prompt format is locked before any record is written to JSONL.

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pandas | 3.0.1 | Load SR1A CSV/fixed-width data, feature engineering, JSONL export | Industry-standard tabular data manipulation; `to_json(orient='records', lines=True)` for JSONL |
| numpy | 2.4.2 | Log-normal random sampling for synthetic price generation | `numpy.random.lognormal(mu, sigma, n)` is the canonical API; broadcasting for county multipliers |
| scikit-learn | 1.8.0 | Stratified train/val/test splitting | `train_test_split` with `stratify=price_bin` handles class imbalance in price distribution |
| scipy | 1.17.1 | Statistical validation of synthetic data against known NJ distributions | `scipy.stats.lognorm` for fitting and comparing distributions; `ks_2samp` for distribution equality test |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| matplotlib | 3.10.8 | Price histogram validation plots | Visualize generated price distribution against NJ county medians before committing to splits |
| json (stdlib) | Python stdlib | JSONL file writing | Write records line-by-line; no extra dependency needed |
| pathlib (stdlib) | Python stdlib | Cross-environment file paths | Lambda imports `lambda/prompt_utils.py`; use `pathlib.Path` for robust path resolution |
| zipfile (stdlib) | Python stdlib | Extracting NJ SR1A ZIP archives | SR1A files from nj.gov are delivered as ZIP archives |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| pandas JSONL export | HuggingFace `datasets` library | `datasets` is overkill for writing JSONL; pandas is simpler and already needed for data prep |
| numpy log-normal | scipy log-normal | Both work; numpy is simpler API for bulk generation; use scipy for fitting/validation |
| scikit-learn stratified split | Manual bucketing + split | scikit-learn handles edge cases (empty strata, etc.) correctly |

**Installation:**
```bash
pip install pandas==3.0.1 numpy==2.4.2 scikit-learn==1.8.0 scipy==1.17.1 matplotlib==3.10.8
```

---

## Architecture Patterns

### Recommended Project Structure

```
housing_price_predictor/
├── lambda/
│   └── prompt_utils.py       # format_prompt() defined HERE — shared by notebooks and handler
├── data/
│   ├── raw/                  # SR1A downloaded files (gitignored)
│   ├── processed/            # Cleaned CSVs before JSONL conversion
│   └── splits/               # train.jsonl, val.jsonl, test.jsonl (gitignored — too large for git)
├── notebooks/
│   └── 01_data_prep.ipynb    # Entire Phase 1 implementation
└── scripts/
    └── validate_distribution.py  # Optional: standalone histogram check
```

### Pattern 1: Shared Prompt Module (CRITICAL)

**What:** `lambda/prompt_utils.py` is created in Phase 1 and imported (not copied) by all notebooks and the Lambda handler. It is the single source of truth for prompt format.

**When to use:** Always. This is the most important anti-fragility choice in the entire pipeline. A prompt format change after training requires full retraining.

**Example:**
```python
# lambda/prompt_utils.py
# Source: Project requirement DATA-03 + FEATURES.md anti-pattern analysis

def format_prompt(bedrooms: int, bathrooms: float, sqft: int, lot_size: float,
                  year_built: int, zip_code: str, property_type: str) -> str:
    """
    Format 7 housing features as a text prompt for LLM training and inference.
    This function MUST be identical at training time and inference time.

    Returns a string ending with 'Predicted price: $' so the model learns
    to generate the price immediately after this prefix.
    """
    return (
        f"Property: {property_type} in zip {zip_code}. "
        f"{bedrooms} bedrooms, {bathrooms} bathrooms, {sqft} sqft living area, "
        f"{lot_size:.2f} acre lot, built in {year_built}. "
        f"Predicted price: $"
    )


def parse_price_from_output(generated_text: str) -> float:
    """
    Extract the first sequence of digits (with optional decimal) from generated text.
    Returns None if no parseable number found.

    Called at inference time to extract numeric price from model output.
    """
    import re
    # Remove commas (model may generate "450,000")
    cleaned = generated_text.replace(",", "")
    match = re.search(r"\d+(?:\.\d+)?", cleaned)
    if match:
        return float(match.group())
    return None
```

**Import pattern in notebooks:**
```python
# In 01_data_prep.ipynb — run from repo root or add parent to path
import sys
sys.path.insert(0, "/path/to/housing_price_predictor")  # or use Google Drive mount path
from lambda.prompt_utils import format_prompt
```

### Pattern 2: Synthetic Data Generation with County-Level Priors

**What:** Generate price from log-normal distribution using county mu/sigma, then derive correlated features from the price. Do NOT generate features first and compute price.

**When to use:** For all synthetic records (the majority — 70%+ of the dataset).

**Example:**
```python
# Source: numpy.random.lognormal docs + NJ county price statistics

import numpy as np

# NJ county median prices (2024-2025, from ATTOM/NJRealtors data)
# mu = log(median), sigma calibrated to match observed NJ price std (~0.4 for most counties)
COUNTY_PRICE_PARAMS = {
    "Bergen":      {"mu": np.log(721200), "sigma": 0.45, "zip_prefix": ["070", "071", "074", "076", "077"]},
    "Morris":      {"mu": np.log(665000), "sigma": 0.40, "zip_prefix": ["079"]},
    "Essex":       {"mu": np.log(660900), "sigma": 0.50, "zip_prefix": ["070", "071", "072"]},
    "Hudson":      {"mu": np.log(659200), "sigma": 0.45, "zip_prefix": ["070", "073"]},
    "Somerset":    {"mu": np.log(624300), "sigma": 0.38, "zip_prefix": ["088", "089"]},
    "Union":       {"mu": np.log(588700), "sigma": 0.42, "zip_prefix": ["070", "072", "079"]},
    "Passaic":     {"mu": np.log(569100), "sigma": 0.43, "zip_prefix": ["074", "075", "077"]},
    "Hunterdon":   {"mu": np.log(574800), "sigma": 0.35, "zip_prefix": ["088"]},
    "Cape May":    {"mu": np.log(545400), "sigma": 0.50, "zip_prefix": ["082"]},
    "Middlesex":   {"mu": np.log(540800), "sigma": 0.40, "zip_prefix": ["088", "089"]},
    "Ocean":       {"mu": np.log(506000), "sigma": 0.42, "zip_prefix": ["087", "082"]},
    "Sussex":      {"mu": np.log(429700), "sigma": 0.38, "zip_prefix": ["074", "078"]},
    "Mercer":      {"mu": np.log(410100), "sigma": 0.42, "zip_prefix": ["085", "086"]},
    "Burlington":  {"mu": np.log(395500), "sigma": 0.38, "zip_prefix": ["080", "081", "083", "084", "085", "086"]},
    "Atlantic":    {"mu": np.log(360200), "sigma": 0.48, "zip_prefix": ["082", "083"]},
    "Gloucester":  {"mu": np.log(361100), "sigma": 0.38, "zip_prefix": ["080", "081", "083"]},
    "Camden":      {"mu": np.log(341700), "sigma": 0.42, "zip_prefix": ["080", "081"]},
    "Warren":      {"mu": np.log(340000), "sigma": 0.40, "zip_prefix": ["078", "088"]},
    "Salem":       {"mu": np.log(259700), "sigma": 0.40, "zip_prefix": ["080", "081"]},
    "Cumberland":  {"mu": np.log(251200), "sigma": 0.42, "zip_prefix": ["083"]},
    "Monmouth":    {"mu": np.log(702500), "sigma": 0.45, "zip_prefix": ["077", "087"]},
}

def generate_synthetic_record(county: str, rng: np.random.Generator) -> dict:
    params = COUNTY_PRICE_PARAMS[county]
    price = float(rng.lognormal(params["mu"], params["sigma"]))
    price = max(100_000, min(price, 3_000_000))  # clip outliers

    # Derive features correlated with price
    # sqft: 300-800 $/sqft in NJ depending on county
    price_per_sqft = rng.uniform(250, 700)
    sqft = int(price / price_per_sqft)
    sqft = max(500, min(sqft, 8000))

    bedrooms = min(6, max(1, int(sqft / 500) + rng.integers(-1, 2)))
    bathrooms = round(bedrooms * rng.uniform(0.5, 1.2) * 0.5) * 0.5  # 0.5 increments
    bathrooms = max(1.0, min(bathrooms, 5.0))
    lot_size = round(float(rng.lognormal(np.log(0.25), 0.8)), 2)  # acres, log-normal
    lot_size = max(0.05, min(lot_size, 10.0))
    year_built = int(rng.integers(1900, 2024))

    # Sample zip from county prefix range
    prefix = rng.choice(params["zip_prefix"])
    zip_code = prefix + str(rng.integers(100, 999)).zfill(3)

    property_type = rng.choice(
        ["Single Family", "Condo", "Townhouse", "Multi-Family"],
        p=[0.60, 0.20, 0.15, 0.05]
    )

    return {
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "sqft": sqft,
        "lot_size": lot_size,
        "year_built": year_built,
        "zip_code": zip_code,
        "property_type": property_type,
        "price": round(price, -2),  # round to nearest $100
        "source": "synthetic",
        "county": county,
    }
```

### Pattern 3: Stratified Train/Val/Test Split (70/15/15)

**What:** Bin prices into quartiles, use scikit-learn stratified split to ensure each price range is represented in all three splits.

**When to use:** Always — small NJ datasets are prone to price distribution skew if split naively.

**Example:**
```python
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def stratified_split(df: pd.DataFrame,
                     train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
                     seed=42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9

    # Create price bins for stratification (quartiles)
    df["price_bin"] = pd.qcut(df["price"], q=4, labels=False, duplicates="drop")

    train_val, test = train_test_split(
        df, test_size=test_ratio, stratify=df["price_bin"], random_state=seed
    )
    adjusted_val_ratio = val_ratio / (train_ratio + val_ratio)
    train, val = train_test_split(
        train_val, test_size=adjusted_val_ratio, stratify=train_val["price_bin"], random_state=seed
    )

    # Drop helper column
    for split in [train, val, test]:
        split.drop(columns=["price_bin"], inplace=True)

    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
```

### Pattern 4: JSONL Export with Prompt Formatting

**What:** Convert each record to `{"prompt": "...", "price": 450000}` format and write to JSONL.

**Example:**
```python
import json
from lambda.prompt_utils import format_prompt

def export_to_jsonl(df: pd.DataFrame, path: str) -> None:
    with open(path, "w") as f:
        for _, row in df.iterrows():
            prompt = format_prompt(
                bedrooms=int(row["bedrooms"]),
                bathrooms=float(row["bathrooms"]),
                sqft=int(row["sqft"]),
                lot_size=float(row["lot_size"]),
                year_built=int(row["year_built"]),
                zip_code=str(row["zip_code"]),
                property_type=str(row["property_type"]),
            )
            record = {"prompt": prompt, "price": float(row["price"])}
            f.write(json.dumps(record) + "\n")
    print(f"Wrote {len(df)} records to {path}")
```

### Pattern 5: Distribution Validation Before Splitting

**What:** After generating all records, validate price histogram matches known NJ medians before writing splits.

**Example:**
```python
import matplotlib.pyplot as plt
import numpy as np

def validate_price_distribution(df: pd.DataFrame) -> None:
    prices = df["price"].values
    known_nj_median = 560_000  # 2024 statewide median (NJRealtors)
    known_nj_mean_log = np.log(known_nj_median)

    generated_median = np.median(prices)
    generated_mean_log = np.mean(np.log(prices))

    pct_diff = abs(generated_median - known_nj_median) / known_nj_median * 100
    print(f"Generated median: ${generated_median:,.0f}")
    print(f"Known NJ median:  ${known_nj_median:,.0f}")
    print(f"Difference:       {pct_diff:.1f}%")

    if pct_diff > 20:
        raise ValueError(
            f"Generated median {generated_median:.0f} differs from "
            f"NJ median {known_nj_median} by {pct_diff:.1f}% > 20% threshold. "
            "Adjust county distribution parameters."
        )

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(np.log(prices), bins=50, alpha=0.7, label="Generated")
    ax.axvline(known_nj_mean_log, color="red", linestyle="--", label="NJ 2024 median (log)")
    ax.set_xlabel("log(price)")
    ax.set_title("Price Distribution Validation")
    ax.legend()
    plt.tight_layout()
    plt.savefig("data/price_distribution_check.png", dpi=100)
    plt.show()
```

### Anti-Patterns to Avoid

- **Hard-coding the prompt template in the notebook:** The notebook must import from `lambda/prompt_utils.py`, not define the template inline. Inline definitions diverge silently from the Lambda handler.
- **Generating features independently and computing price:** This creates unrealistic feature-price correlations. Generate price from county distribution first, then derive features from price.
- **Writing all records as `source: synthetic`:** At least 30% must be from real public NJ data (DATA-04 requirement). Track the `source` field per record.
- **Naively splitting by index:** Use stratified split by price quartile. Naive splits produce val/test sets that don't represent the full price range.
- **Skipping the 20% tolerance check:** If statewide generated median diverges from NJ median by more than 20%, the county `mu`/`sigma` parameters are wrong. Fail loudly rather than silently writing bad data.

---

## NJ County Price Statistics (2024-2025)

Reference data for synthetic generation mu/sigma parameters. Source: ATTOM September 2025 data, NJRealtors 2024 year-end report, and Redfin county market pages. Confidence: MEDIUM (multiple sources agree on order of magnitude; exact figures vary by source and month).

| County | Median Price (2024-25) | Log-Normal mu | Suggested sigma | Notes |
|--------|----------------------|---------------|-----------------|-------|
| Bergen | $721,200 | 13.49 | 0.45 | Most expensive; includes luxury markets |
| Monmouth | $702,500 | 13.46 | 0.45 | Shore properties inflate median |
| Morris | $665,000 | 13.41 | 0.40 | Stable suburban market |
| Essex | $660,900 | 13.40 | 0.50 | Wide spread: Montclair vs Newark |
| Hudson | $659,200 | 13.40 | 0.45 | Urban condos (Hoboken/JC) inflate median |
| Somerset | $624,300 | 13.34 | 0.38 | Stable; limited new construction |
| Passaic | $569,100 | 13.25 | 0.43 | Mixed urban/suburban |
| Hunterdon | $574,800 | 13.26 | 0.35 | Rural; lower volume |
| Cape May | $545,400 | 13.21 | 0.50 | Shore seasonal market; high variance |
| Middlesex | $540,800 | 13.20 | 0.40 | Large county; diverse submarkets |
| Union | $588,700 | 13.29 | 0.42 | Near NYC commuter premium |
| Ocean | $506,000 | 13.13 | 0.42 | Retirement communities + shore |
| Sussex | $429,700 | 12.97 | 0.38 | Rural; limited market |
| Mercer | $410,100 | 12.92 | 0.42 | Includes Trenton (affordable) + Princeton |
| Burlington | $395,500 | 12.89 | 0.38 | Suburban; consistent market |
| Atlantic | $360,200 | 12.79 | 0.48 | Atlantic City market; high variance |
| Gloucester | $361,100 | 12.80 | 0.38 | Suburban South Jersey |
| Camden | $341,700 | 12.74 | 0.42 | Includes City of Camden (affordable) |
| Warren | $340,000 | 12.74 | 0.40 | Rural; estimate, limited sources |
| Salem | $259,700 | 12.47 | 0.40 | Most affordable in South Jersey |
| Cumberland | $251,200 | 12.43 | 0.42 | Statewide minimum; agricultural area |

**Statewide 2024:** Median $560,000, overall NJRealtors year-end report.

---

## NJ Public Data Sources (DATA-04 Requirement)

At least 30% of records must come from real public NJ datasets.

### Primary: NJ Division of Taxation SR1A Sales File

- **URL:** https://www.nj.gov/treasury/taxation/lpt/statdata.shtml
- **Files:** ZIP archives for 2020-2026 (year-to-date)
- **Format:** Fixed-width or delimited file; requires parsing via SR1A File Layout PDF
- **Fields confirmed present:** Sale price, sale date, property class code (which maps to property type), county code, municipality code
- **Fields uncertain:** Bedrooms, bathrooms, square footage — these are collected on the SR1A form but field availability varies by county and year. The MODIV Property Assessment file (also on the same page, 2021-2025) includes more structural detail.
- **Practical approach:** Download SR1A 2024 file, parse sale price + property class + municipality. Join with MODIV assessment data on parcel number to get sqft and structural details for the subset that matches. Fill remaining missing structural fields synthetically using county-derived distributions.
- **Record count:** Statewide NJ has approximately 86,440 closed sales in 2025 per NJRealtors. Full SR1A file likely contains 50,000-100,000 records per year. Exact count not confirmed — verify on download.
- **Confidence:** MEDIUM — file exists and structure is documented, but field completeness for sqft/bedrooms/bathrooms is uncertain until downloaded and inspected.

### Alternative/Supplemental: NJ MODIV Property Assessment Files

- **URL:** https://www.nj.gov/treasury/taxation/lpt/statdata.shtml (same page, separate section)
- **Files:** Available 2021-2025
- **Purpose:** Assessment files include structural details (living area, year built) that SR1A may be missing
- **Use:** Join to SR1A on parcel/block/lot identifier to enrich records with structural fields

### Fallback: Data.gov NJ Property Searches

- **URL:** https://data.gov (search "New Jersey property sales")
- **Notes:** data.gov aggregates state and local government data. NJ-specific property datasets may be available but schemas vary widely. Confidence: LOW until verified by searching.

### Schema to Document (DATA-04 Requirement)

Create `data/SCHEMA.md` documenting:

```markdown
## Dataset Schema

| Field | Type | Source | Range | Notes |
|-------|------|--------|-------|-------|
| bedrooms | int | synthetic / SR1A | 1-6 | Integer |
| bathrooms | float | synthetic / SR1A | 1.0-5.0 | Half-bath increments (1.0, 1.5, ...) |
| sqft | int | synthetic / MODIV | 500-8000 | Living area only (not lot) |
| lot_size | float | synthetic / MODIV | 0.05-10.0 | Acres |
| year_built | int | synthetic / MODIV | 1900-2024 | |
| zip_code | str | synthetic / SR1A | NJ zip codes | 5-digit string, leading zero preserved |
| property_type | str | synthetic / SR1A class code | Single Family, Condo, Townhouse, Multi-Family | Mapped from SR1A class 2 = residential |
| price | float | SR1A sale price / synthetic | $100,000-$3,000,000 | Sale price (not assessment) |
| source | str | metadata | "real" / "synthetic" | Track for DATA-04 audit |
| county | str | derived from zip / SR1A county code | 21 NJ counties | |
```

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Train/val/test splitting with stratification | Manual bucketing and random index selection | `sklearn.train_test_split(stratify=...)` | Correctly handles edge cases: empty strata, rounding of split sizes, reproducible seed |
| Log-normal random sampling | Custom inverse-CDF or Box-Muller implementation | `numpy.random.lognormal(mu, sigma, n)` | One line; vectorized; correct parameterization |
| Distribution comparison | Manual histogram binning and chi-square | `scipy.stats.ks_2samp` | Kolmogorov-Smirnov test handles continuous distributions correctly; doesn't require binning |
| Parsing fixed-width or delimited NJ SR1A files | Manual string slicing | `pandas.read_fwf()` or `pandas.read_csv()` with correct delimiter | pandas handles encoding, missing fields, type inference |
| JSONL writing | Manual JSON construction | `json.dumps(record) + "\n"` per line | stdlib; no extra dependency; handles Unicode and float edge cases |

**Key insight:** All data manipulation in this phase uses standard library + pandas/numpy/scikit-learn. Nothing requires a custom implementation.

---

## Common Pitfalls

### Pitfall 1: Synthetic Price Distribution Mismatch — Highest Recovery Cost

**What goes wrong:** Synthetic prices generated with a flat or uniform distribution (or a simple sqft * constant formula) produce a dataset where the model trains to near-zero loss on training data but achieves MAPE > 50% on real NJ properties. The distribution of generated prices doesn't match NJ market statistics.

**Why it happens:** County-level price variation in NJ is enormous: Bergen County median $721k vs. Cumberland County median $251k. Without county-level priors, flat generation treats the entire state as homogeneous. The model never learns the zip-code/county price signal.

**How to avoid:** Generate price from county log-normal distribution (mu = log(county_median), sigma = 0.4-0.5). Validate generated median is within 20% of known NJ statewide median ($560k) before writing JSONL.

**Warning signs:** Generated median price is below $350k or above $700k; all zip codes have similar price histograms; distribution validation function raises `ValueError`.

### Pitfall 2: Prompt Format Defined in Notebook, Not in `lambda/prompt_utils.py`

**What goes wrong:** Phase 1 defines the prompt template inline in the notebook. Phase 4 defines a slightly different template in the Lambda handler. The model sees training-time prompts like `"...{sqft} sqft..."` but inference-time prompts like `"...{sqft} square feet..."`. Silent accuracy degradation — model generates wrong prices or fails to parse.

**Why it happens:** It's easier to write the template inline when building the notebook. The disconnect isn't apparent until Lambda is deployed.

**How to avoid:** Create `lambda/prompt_utils.py` FIRST in Phase 1, before generating any records. All notebooks `import from lambda.prompt_utils`. Test the import before writing 10,000 records.

**Warning signs:** The word "sqft" appears in the notebook with different surrounding text than in the Lambda handler. Two different variables named `PROMPT_TEMPLATE` in the codebase.

### Pitfall 3: SR1A File Missing Structural Fields (Bedrooms, Bathrooms, Sqft)

**What goes wrong:** The SR1A sales file is downloaded expecting it to contain all 7 required features. On inspection, the bedrooms, bathrooms, and sqft fields are empty or inconsistently populated across counties. Hours are spent trying to clean bad data from the real dataset.

**Why it happens:** SR1A is a real estate transfer form filed at time of sale. Structural details are populated by the municipal assessor — completeness varies by county and submission year.

**How to avoid:** Download the SR1A 2024 file on Day 1 of implementation and inspect field completeness. If structural fields are sparse, use MODIV Property Assessment files as the structural source and join to SR1A on parcel ID. Plan to synthesize structural fields for records where MODIV join fails.

**Warning signs:** More than 30% of real records have null/zero sqft; more than 50% have null bedrooms after parsing SR1A.

### Pitfall 4: JSONL Records Reference Prompt Template Directly Instead of Calling `format_prompt()`

**What goes wrong:** The JSONL records are written as `{"text": "Property: Single Family in zip 07650...Predicted price: $350000"}` with the price embedded in the prompt string. The training notebook later expects `{"prompt": "...", "price": 350000}` as separate fields. The `SFTTrainer` receives the entire "prompt + price" as the target sequence, which means the model is trained to regenerate the price as text but loss is computed over both the prompt tokens AND the price token — making it harder to isolate price prediction quality.

**How to avoid:** Keep prompt and price as separate JSONL fields: `{"prompt": "...", "price": 350000.0}`. The training notebook composes the full training text from these fields, which lets it control what the model is trained to generate.

**Warning signs:** JSONL `text` field contains the numeric price concatenated to the prompt; no separate `price` field exists.

### Pitfall 5: Zip Codes Stored as Integer (Losing Leading Zero)

**What goes wrong:** NJ zip codes beginning with `07` (North Jersey) are stored as integers, dropping the leading zero. `07650` becomes `7650`. The model learns to predict based on 4-digit zip codes for North Jersey but 5-digit for South Jersey, creating an inconsistent feature.

**How to avoid:** Store `zip_code` as a zero-padded string: `str(zip_code).zfill(5)`. Use `dtype=str` when reading zip code columns from CSV. Validate all zip codes in the final dataset are 5 characters.

**Warning signs:** `df["zip_code"].str.len().min()` returns 4 in the dataset.

---

## Code Examples

Verified patterns from Python stdlib and official library documentation:

### numpy.random.lognormal API

```python
# Source: https://numpy.org/doc/stable/reference/random/generated/numpy.random.lognormal.html
# numpy 2.4.x — use Generator API (not legacy numpy.random.lognormal)
import numpy as np

rng = np.random.default_rng(seed=42)

# mu = mean of the underlying normal distribution = log(desired_median)
# sigma = std of the underlying normal distribution; ~0.4 for NJ housing prices
county_median = 721_200  # Bergen County
mu = np.log(county_median)
sigma = 0.45

# Generate 1000 synthetic prices
prices = rng.lognormal(mean=mu, sigma=sigma, size=1000)
print(f"Generated median: ${np.median(prices):,.0f}")  # Should be near 721,200
```

### scikit-learn Stratified Split

```python
# Source: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
from sklearn.model_selection import train_test_split
import pandas as pd

df["price_bin"] = pd.qcut(df["price"], q=4, labels=False, duplicates="drop")

# First split: separate test set (15%)
train_val, test = train_test_split(df, test_size=0.15, stratify=df["price_bin"], random_state=42)

# Second split: separate val from train (15% of total = 15/85 of remaining)
train, val = train_test_split(train_val, test_size=0.15/0.85, stratify=train_val["price_bin"], random_state=42)

print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
print(f"Ratios: {len(train)/len(df):.2%} / {len(val)/len(df):.2%} / {len(test)/len(df):.2%}")
# Expected: ~70% / ~15% / ~15%
```

### JSONL Read/Write

```python
# Writing JSONL (stdlib only)
import json

records = [{"prompt": "...", "price": 450000.0}, ...]
with open("train.jsonl", "w") as f:
    for record in records:
        f.write(json.dumps(record) + "\n")

# Reading JSONL (stdlib only)
with open("train.jsonl", "r") as f:
    records = [json.loads(line) for line in f]
```

### Parsing NJ SR1A Fixed-Width File

```python
# SR1A files are typically fixed-width; consult SR1A_FileLayout_Description.pdf for column positions
# General pattern (column positions are placeholders — verify against actual layout PDF):
import pandas as pd

# After downloading and extracting SR1A ZIP from nj.gov:
# Read with read_fwf (fixed-width format) if whitespace-delimited
# OR read_csv with appropriate separator if comma/pipe-delimited
# The layout PDF specifies the actual positions

# Example of how to filter to residential sales only:
# SR1A property class codes: class 2 = residential (single family, condo, etc.)
df_sr1a = pd.read_fwf("SR1A_2024.txt", ...)  # fill positions from layout PDF
df_residential = df_sr1a[df_sr1a["property_class"] == 2].copy()
df_residential["source"] = "real"
```

### scipy KS Test for Distribution Validation

```python
# Source: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html
from scipy import stats
import numpy as np

# Compare generated prices to a reference log-normal distribution
generated_prices = df_synthetic["price"].values
reference_prices = np.random.lognormal(np.log(560_000), 0.45, size=10_000)

stat, pvalue = stats.ks_2samp(generated_prices, reference_prices)
print(f"KS statistic: {stat:.3f}, p-value: {pvalue:.3f}")
# p > 0.05 means distributions are not significantly different — good
# p < 0.05 means distributions differ — investigate county mu/sigma parameters
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Generate features independently, compute price as formula | Generate price from county log-normal first, derive features from price | Standard ML data practice | More realistic correlations; less distribution mismatch |
| Store JSONL as `{"text": "...price..."}` (full text) | Store as `{"prompt": "...", "price": float}` (separate fields) | HuggingFace SFTTrainer standard | Trainer can correctly compute loss over price token only |
| Single static prompt template string in notebook | Shared `prompt_utils.py` module imported everywhere | Anti-fragility pattern | Zero-divergence risk between training and inference |
| Random train/test split | Stratified split by price quartile | Best practice for skewed regression data | Prevents empty price ranges in val/test |

**Deprecated/outdated:**
- `numpy.random.lognormal()` (legacy interface): Use `numpy.random.default_rng(seed).lognormal()` (Generator API) for reproducible, seedable generation.
- Hard-coded prompt strings in data generation: Replaced by shared `format_prompt()` function.

---

## Open Questions

1. **SR1A Field Completeness for Sqft/Bedrooms/Bathrooms**
   - What we know: SR1A file exists and is downloadable. Sale price, date, and property class are consistently present.
   - What's unclear: Whether the 2024 SR1A file has populated sqft/bedroom/bathroom fields. This is the biggest unknown for DATA-04 (30% real data requirement).
   - Recommendation: Download and inspect SR1A 2024 file on Day 1 of implementation. If structural fields are sparse (>30% null), download MODIV assessment file and join. If join still leaves gaps, mark those records as `source: real-partial` and fill structural fields synthetically using county distributions. The 30% requirement is met as long as the price and property class come from real sales data.

2. **Exact SR1A File Delimiter/Format**
   - What we know: Files are distributed as ZIP archives. The official layout PDF specifies field positions.
   - What's unclear: Whether the file is fixed-width (read_fwf) or delimited (read_csv). The layout PDF is needed to determine this.
   - Recommendation: Download the SR1A_FileLayout_Description.pdf from nj.gov before writing any SR1A parsing code. Do not assume the format.

3. **Synthetic Record Count Target**
   - What we know: DATA-04 requires at least 30% real data. No upper bound on total records is specified.
   - What's unclear: How many total records are needed for effective QLoRA fine-tuning on Qwen2.5-0.5B. More records = better generalization but slower training.
   - Recommendation: Target 5,000-10,000 total records. At 30% real (if SR1A has sufficient coverage), that means 1,500-3,000 real records and 3,500-7,000 synthetic. 10,000 total is a reasonable upper bound for 20-minute Colab training. Adjust based on SR1A record count after download.

4. **NJ Zip Code to County Mapping**
   - What we know: NJ zip code prefixes correlate with counties but there is overlap (multiple counties share prefixes).
   - What's unclear: A definitive NJ zip-code-to-county CSV is not included in SR1A data.
   - Recommendation: Use the USPS zip code database or a published NJ zip-to-county mapping. For synthetic records, generate zip from county prefix table (as shown in Pattern 2 code). For SR1A records, use the county code field directly.

---

## Sources

### Primary (HIGH confidence)
- NJ Division of Taxation Statistical Data page — SR1A Sales File availability confirmed: https://www.nj.gov/treasury/taxation/lpt/statdata.shtml
- numpy.random.lognormal documentation: https://numpy.org/doc/stable/reference/random/generated/numpy.random.lognormal.html
- scikit-learn train_test_split documentation: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
- scipy.stats.ks_2samp documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html
- Project STACK.md — library versions (pandas 3.0.1, numpy 2.4.2, scikit-learn 1.8.0, scipy 1.17.1) verified 2026-02-26

### Secondary (MEDIUM confidence)
- ATTOM September 2025 NJ county median prices — sourced via WebSearch (multiple aggregator sites reporting consistent order-of-magnitude values)
- NJRealtors 2025 year-end report — statewide median $525,000-$560,000 range: https://www.insidernj.com/press-release/new-jersey-realtors-releases-year-end-housing-data/
- Redfin Bergen County median $755K (December 2025): https://www.redfin.com/county/1892/NJ/Bergen-County/housing-market
- Synthetic data generation with county multiplier pattern: https://medium.com/@meetlimbachiya2005/house-price-prediction-with-a-self-generated-dataset-a-comprehensive-walkthrough-d29c6f9fd92a

### Tertiary (LOW confidence — verify before use)
- SR1A file field completeness for bedrooms/bathrooms/sqft — inferred from form structure, not confirmed by downloading a file
- NJ zip code prefix to county mapping — derived from known NJ geography, not from an authoritative source
- Warren County median price ($340,000) — estimated from surrounding counties; limited direct data

---

## Metadata

**Confidence breakdown:**
- Standard stack (pandas, numpy, scikit-learn, scipy): HIGH — verified versions from project STACK.md (2026-02-26)
- Architecture patterns (prompt module, synthetic generation, stratified split): HIGH — pattern is based on well-documented Python libraries and project requirements
- NJ county price statistics: MEDIUM — multiple sources agree on order of magnitude; exact figures vary by source and recency
- SR1A file structure/field completeness: MEDIUM-LOW — file existence confirmed, but field completeness for structural features (sqft, beds, baths) is unverified until downloaded
- Common pitfalls: HIGH — distribution mismatch and prompt divergence pitfalls are documented in project PITFALLS.md with cross-references to FEATURES.md

**Research date:** 2026-02-26
**Valid until:** 2026-03-28 (30 days — county price statistics are stable; SR1A file availability is stable)
