"""Evaluate a model on the dataset.


Usage:
    python evaluate_model.py <model_file>

    model_file: Path to the model file.
"""

import argparse
import importlib
import os
import sys

import joblib


def import_custom_models_for_transform(src_dir: str, transform: str) -> None:
    """Dynamically import all symbols from train_speed or train_tempo."""
    module_name = "train_speed" if transform == "speed" else "train_tempo"

    sys.path.insert(0, src_dir)

    try:
        mod = importlib.import_module(module_name)
        # Emulate `from module import *`
        for name in dir(mod):
            if not name.startswith("_"):
                globals()[name] = getattr(mod, name)
    except Exception as e:
        print(f"Warning: could not import from {module_name}: {e}")
        print("If your model uses custom classes from that module, loading may fail.")


def infer_transform_from_filename(data_file: str) -> str | None:
    """Infer the transform type (speed or tempo) from the data file name."""
    filename = os.path.basename(data_file).lower()
    if "speed" in filename:
        return "speed"
    elif "tempo" in filename:
        return "tempo"
    else:
        return None


def evaluate(
    model_file: str, data_file: str, show_confusion_matrix: bool = True
) -> None:
    """Evaluate a model on the eval1 dataset.

    Args:
        model_file (str): Path to the model file.
    """

    print(f"Evaluating {model_file}")

    # Check that model file is no larger than 20 MB
    if os.path.getsize(model_file) > 80 * 1024 * 1024:
        print("ERROR: Model file is larger than the allowed 80 MB limit.")
        return

    model = joblib.load(model_file)
    print(model)

    eval_data = joblib.load(open(data_file, "rb"))

    x_test = eval_data["features"]
    y_test = eval_data["target"]

    score = model.score(x_test, y_test)
    print("Score:", score * 100, "%")

    if show_confusion_matrix:
        # Lazy import to avoid dependency if not plotting
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay

        y_pred = model.predict(x_test)

        disp = ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            normalize="true",
            cmap="Blues",
            display_labels=(["very slow", "slow", "normal", "fast", "very fast"]),
        )
        disp.ax_.set_title("Confusion Matrix")

        plt.show()


def main() -> None:
    """Evaluate a trained model from command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument(
        "src_dir",
        type=str,
        help="Path to your Python source directory",
    )
    parser.add_argument(
        "model_file",
        type=str,
        help="Path to the model file (joblib format)",
    )
    parser.add_argument(
        "data_file",
        type=str,
        help="Path to evaluation data file (joblib format)",
    )
    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="Suppress plotting the confusion matrix",
    )
    args = parser.parse_args()

    transform = infer_transform_from_filename(args.data_file)
    if transform is None:
        print("ERROR: Could not infer transform type from data file name.")
        return

    # Import custom classes for unpickling before loading the model
    import_custom_models_for_transform(args.src_dir, transform)

    evaluate(args.model_file, args.data_file, not args.no_plot)


if __name__ == "__main__":
    main()
