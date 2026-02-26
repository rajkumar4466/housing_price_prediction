"""
Shared prompt formatting utilities for NJ housing price prediction.

This module is the single source of truth for prompt format.
It is imported by:
  - notebooks/01_data_prep.ipynb  (training data generation)
  - notebooks/02_train.ipynb       (training loop)
  - notebooks/03_evaluate.ipynb    (evaluation)
  - lambda/handler.py              (Lambda inference handler)

WARNING: Changing the prompt template after training data has been written
requires regenerating all data AND retraining from scratch.
"""

import re


def format_prompt(
    bedrooms: int,
    bathrooms: float,
    sqft: int,
    lot_size: float,
    year_built: int,
    zip_code: str,
    property_type: str,
) -> str:
    """
    Format 7 NJ housing features as a text prompt for LLM training and inference.

    The returned string ends with 'Predicted price: $' so the model learns to
    generate the price immediately after this prefix.

    Args:
        bedrooms:      Integer number of bedrooms (1–6)
        bathrooms:     Float number of bathrooms in 0.5 increments (1.0–5.0)
        sqft:          Integer living area in square feet (500–8000)
        lot_size:      Float lot size in acres (0.05–10.0)
        year_built:    Integer year the property was built (1900–2024)
        zip_code:      5-digit NJ zip code string (leading zero preserved, e.g. "07650")
        property_type: One of "Single Family", "Condo", "Townhouse", "Multi-Family"

    Returns:
        Prompt string ending with "Predicted price: $"
    """
    return (
        f"Property: {property_type} in zip {zip_code}. "
        f"{bedrooms} bedrooms, {bathrooms} bathrooms, {sqft} sqft living area, "
        f"{lot_size:.2f} acre lot, built in {year_built}. "
        f"Predicted price: $"
    )


def parse_price_from_output(generated_text: str) -> float | None:
    """
    Extract the first numeric price from model-generated text.

    Handles comma-formatted numbers (e.g. "450,000") and decimals.
    Returns None if no parseable number is found.

    Called at inference time to extract the predicted price from model output.
    """
    cleaned = generated_text.replace(",", "")
    match = re.search(r"\d+(?:\.\d+)?", cleaned)
    if match:
        return float(match.group())
    return None
