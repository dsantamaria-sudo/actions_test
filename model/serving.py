import mlflow.pyfunc
import numpy as np
import torch


class SoftmaxClassifier(mlflow.pyfunc.PythonModel):
    """Serving-only wrapper: applies softmax on top of a trained classifier's raw logits
    and labels each output with its class name. Kept separate from the trained nn.Module
    itself -- CrossEntropyLoss already applies softmax internally during training, so baking
    it into the model's forward() would double-apply it and break training.

    The weights are held as a plain attribute (pickled with the wrapper via log_model's
    python_model= argument) rather than passed through log_model's artifacts= mechanism --
    that path builds its stored relative path with os.path.join, which bakes a Windows
    backslash into the model's metadata when saved on Windows, breaking the path when the
    model is loaded on a Linux serving container (mlflow 3.13.0)."""

    def __init__(self, model: torch.nn.Module, class_names: list[str]):
        self.model = model
        self.model.eval()
        self.class_names = list(class_names)

    def predict(self, context, model_input, params=None):
        x = torch.as_tensor(np.asarray(model_input), dtype=torch.float32)
        with torch.inference_mode():
            probs = torch.softmax(self.model(x), dim=-1).numpy()
        return [dict(zip(self.class_names, row.tolist())) for row in probs]
