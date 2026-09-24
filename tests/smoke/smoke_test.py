"""
Smoke test for the tinyConvNeXt Azure ML Managed Online Endpoint.

Sends a pre-processed reference payload (tests/smoke/test_payload.json) to the
scoring endpoint and validates:
  1. HTTP 200 response within a generous timeout (endpoint scales to 0
     replicas, so a cold start can take well over a minute).
  2. The response has the expected shape: probabilities for all 3 known
     classes (pizza, steak, sushi), summing to ~1.

No assertion is made about which class should win — this is a structural /
liveness check, not a regression test against a known label.

Required environment variables:
  AZURE_ENDPOINT_URL  - e.g. https://amls-imhmj.spaincentral.inference.ml.azure.com/score
  AZURE_ENDPOINT_KEY  - the endpoint's primary (or secondary) key

Exit code is non-zero on any failure, so this can be used directly as a
GitHub Actions step.
"""

import json
import os
import sys
import time
from pathlib import Path

import requests

CLASSES = ["pizza", "steak", "sushi"]

PAYLOAD_PATH = Path(__file__).parent / "test_payload.json"
TIMEOUT_SECONDS = 120


def extract_probs(result):
    """Normalize the various shapes the endpoint might return into a
    {class_name: probability} dict."""
    # {"predictions": [...]} wrapper
    if isinstance(result, dict) and "predictions" in result:
        result = result["predictions"]

    # Unwrap a single-item batch: [x] -> x
    if isinstance(result, list) and len(result) == 1:
        result = result[0]

    # Dict keyed by class name: {"pizza": 0.1, "steak": 0.3, "sushi": 0.6}
    if isinstance(result, dict):
        return result

    # Bare list of floats, positional per CLASSES order
    if isinstance(result, list) and len(result) == len(CLASSES):
        return dict(zip(CLASSES, result))

    return None


def main() -> int:
    endpoint_url = os.environ.get("AZURE_ENDPOINT_URL")
    endpoint_key = os.environ.get("AZURE_ENDPOINT_KEY")

    if not endpoint_url or not endpoint_key:
        print("ERROR: AZURE_ENDPOINT_URL and AZURE_ENDPOINT_KEY must be set", file=sys.stderr)
        return 1

    if not PAYLOAD_PATH.exists():
        print(f"ERROR: payload file not found at {PAYLOAD_PATH}", file=sys.stderr)
        return 1

    with open(PAYLOAD_PATH) as f:
        payload = json.load(f)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {endpoint_key}",
    }

    print(f"POST {endpoint_url} (timeout={TIMEOUT_SECONDS}s, cold start possible)...")
    start = time.monotonic()
    try:
        response = requests.post(
            endpoint_url,
            headers=headers,
            json=payload,
            timeout=TIMEOUT_SECONDS,
        )
    except requests.exceptions.RequestException as exc:
        print(f"FAIL: request error: {exc}", file=sys.stderr)
        return 1
    elapsed = time.monotonic() - start
    print(f"Response received in {elapsed:.1f}s (status {response.status_code})")

    if response.status_code != 200:
        print(f"FAIL: expected status 200, got {response.status_code}", file=sys.stderr)
        print(f"Body: {response.text[:1000]}", file=sys.stderr)
        return 1

    try:
        result = response.json()
    except ValueError:
        print(f"FAIL: response is not valid JSON: {response.text[:1000]}", file=sys.stderr)
        return 1

    probs = extract_probs(result)

    if probs is None:
        print(f"FAIL: unrecognized response shape: {result}", file=sys.stderr)
        return 1

    missing = [c for c in CLASSES if c not in probs]
    if missing:
        print(f"FAIL: response missing expected classes {missing}: {probs}", file=sys.stderr)
        return 1

    try:
        values = [float(probs[c]) for c in CLASSES]
    except (TypeError, ValueError):
        print(f"FAIL: probabilities are not numeric: {probs}", file=sys.stderr)
        return 1

    total = sum(values)
    if not (0.95 <= total <= 1.05):
        print(f"FAIL: probabilities sum to {total:.4f}, expected ~1.0: {probs}", file=sys.stderr)
        return 1

    predicted_class = max(probs, key=probs.get)
    print(f"Predicted: {predicted_class} (probs={ {c: round(float(probs[c]), 4) for c in CLASSES} })")
    print("PASS: endpoint healthy, response well-formed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
