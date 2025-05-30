from pathlib import Path

def create_predictions_directory():
    base_dir = Path(__file__).parent.parent
    prediction_base = base_dir / "predictions"

    prediction_base.mkdir(parents=True, exist_ok=True)
    return prediction_base