import os
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, r"C:\Users\Usuario\Desktop")
import mlflow.pytorch


def main() -> None:
    if len(sys.argv) < 2:
        print("Uso: load_model.py <ruta_a_artifacts>")
        sys.exit(1)

    # MLflow interpreta model_uri como una URI (scheme://resto). En Windows,
    # "C:\..." hace que "C" se lea como scheme y falle con
    # "Could not find a registered artifact repository for: c:\...".
    # Path.as_uri() genera el "file:///C:/..." correcto que MLflow sí reconoce.
    artifacts_path = Path(sys.argv[1]).resolve().as_uri()

    try:
        model = mlflow.pytorch.load_model(artifacts_path)
        print("MODEL:", model)
    except Exception: # noqa: BLE001
        traceback.print_exc()


if __name__ == "__main__":
    main()