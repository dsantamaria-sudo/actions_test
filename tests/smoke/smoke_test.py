"""
Smoke test for the tinyConvNeXt Azure ML Managed Online Endpoint.

Sends a pre-processed reference payload (tests/smoke/test_payload.json) to the
scoring endpoint and validates:
  1. HTTP 200 response within a generous timeout (endpoint scales to 0
     replicas, so a cold start can take well over a minute).
  2. The response has the expected shape: 3 class probabilities that sum to ~1.
  3. The predicted class matches the known class of the reference image
     ("steak").

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
EXPECTED_CLASS = "steak"

PAYLOAD_PATH = Path(__file__).parent / "test_payload.json"
TIMEOUT_SECONDS = 120


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

    # Azure ML MLflow pyfunc scoring typically returns either a bare list of
    # predictions, or {"predictions": [...]}. Handle both.
    if isinstance(result, dict) and "predictions" in result:
        predictions = result["predictions"]
    else:
        predictions = result

    # Unwrap a single-item batch: [[p0, p1, p2]] -> [p0, p1, p2]
    if isinstance(predictions, list) and len(predictions) == 1 and isinstance(predictions[0], list):
        probs = predictions[0]
    else:
        probs = predictions

    if not isinstance(probs, list) or len(probs) != len(CLASSES):
        print(f"FAIL: expected {len(CLASSES)} class probabilities, got: {probs}", file=sys.stderr)
        return 1

    try:
        probs = [float(p) for p in probs]
    except (TypeError, ValueError):
        print(f"FAIL: probabilities are not numeric: {probs}", file=sys.stderr)
        return 1

    total = sum(probs)
    if not (0.95 <= total <= 1.05):
        print(f"FAIL: probabilities sum to {total:.4f}, expected ~1.0", file=sys.stderr)
        return 1

    predicted_idx = probs.index(max(probs))
    predicted_class = CLASSES[predicted_idx]
    print(f"Predicted: {predicted_class} (probs={[round(p, 4) for p in probs]})")

    if predicted_class != EXPECTED_CLASS:
        print(
            f"FAIL: expected class '{EXPECTED_CLASS}', got '{predicted_class}'",
            file=sys.stderr,
        )
        return 1

    print("PASS: endpoint healthy, response well-formed, prediction correct.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
