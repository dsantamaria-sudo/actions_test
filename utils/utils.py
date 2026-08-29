import random
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn
from torchmetrics.classification import (
    MulticlassAUROC,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

# matplotlib, mlflow and model.serving are imported lazily inside the functions
# that use them (display_random_images / train). Keeps `import utils.utils` --
# and therefore the CI test suite for accuracy_fn / train_step / test_step --
# free of the MLflow + plotting dependency stack.


def display_random_images(
    dataloader: torch.utils.data.DataLoader,
    classes: list[str] | None = None,
    n: int = 10,
    display_shape: bool = True,
    seed: int | None = None,
):

    import matplotlib.pyplot as plt

    # 2. Adjust display if n is too high
    if n > 10:
        n = 10
        display_shape = False
        print(
            "For display, purposes, n shouldn't be larger than 10, setting to 10 and removing shape display."
        )

    # 3. Set the seed
    if seed:
        random.seed(seed)

    # 4. Get a batch of images from the dataloader
    image_batch, label_batch = next(iter(dataloader))

    # 5. Get random sample indexes within the batch
    n = min(n, len(image_batch))
    random_samples_idx = random.sample(range(len(image_batch)), k=n)

    # 6. Setup plot
    plt.figure(figsize=(16, 8))

    # 7. Loop through random indexes and plot them with matplotlib
    for i, targ_sample in enumerate(random_samples_idx):
        targ_image, targ_label = image_batch[targ_sample], label_batch[targ_sample]

        # 8. Adjust tensor dimensions for plotting
        targ_image_adjust = targ_image.permute(
            1, 2, 0
        )  # [color_channels, height, width] -> [height, width, color_channels]

        # Plot adjusted samples
        plt.subplot(1, n, i + 1)
        plt.imshow(targ_image_adjust)
        plt.axis("off")
        if classes:
            title = f"Class: {classes[targ_label]}"
            if display_shape:
                title = title + f"\nshape: {targ_image_adjust.shape}"
        plt.title(title)

    plt.show()


def accuracy_fn(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    correct = torch.eq(y_true, y_pred).sum().item()
    return (correct / len(y_pred)) * 100


def train_step(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_classes: int,
    accuracy_fn: Callable = accuracy_fn,
    average: str = "macro",
    device: torch.device = "cpu",
) -> dict[str, float]:

    model.train()
    train_loss, train_acc = 0.0, 0.0

    precision_metric = MulticlassPrecision(num_classes=num_classes, average=average).to(
        device
    )
    recall_metric = MulticlassRecall(num_classes=num_classes, average=average).to(
        device
    )
    f1_metric = MulticlassF1Score(num_classes=num_classes, average=average).to(device)
    auroc_metric = MulticlassAUROC(num_classes=num_classes, average=average).to(device)
    precision_per_class = MulticlassPrecision(num_classes=num_classes, average=None).to(
        device
    )
    recall_per_class = MulticlassRecall(num_classes=num_classes, average=None).to(
        device
    )
    f1_per_class = MulticlassF1Score(num_classes=num_classes, average=None).to(device)
    auroc_per_class = MulticlassAUROC(num_classes=num_classes, average=None).to(device)

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)
        preds = y_pred.argmax(dim=1)
        probs = torch.softmax(y_pred, dim=1)

        # 2. Loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        train_acc += accuracy_fn(y_true=y, y_pred=preds)

        precision_metric.update(preds, y)
        recall_metric.update(preds, y)
        f1_metric.update(preds, y)
        auroc_metric.update(probs, y)
        precision_per_class.update(preds, y)
        recall_per_class.update(preds, y)
        f1_per_class.update(preds, y)
        auroc_per_class.update(probs, y)

        # 3. Backprop
        optimizer.zero_grad()
        loss.backward()

        # 4. Optimizer step
        optimizer.step()

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    return {
        "loss": train_loss,
        "acc": train_acc,
        "precision": precision_metric.compute().item(),
        "recall": recall_metric.compute().item(),
        "f1": f1_metric.compute().item(),
        "auroc": auroc_metric.compute().item(),
        "precision_per_class": precision_per_class.compute().tolist(),
        "recall_per_class": recall_per_class.compute().tolist(),
        "f1_per_class": f1_per_class.compute().tolist(),
        "auroc_per_class": auroc_per_class.compute().tolist(),
    }


def test_step(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    num_classes: int,
    accuracy_fn: Callable = accuracy_fn,
    average: str = "macro",
    device: torch.device = "cpu",
    track_confusion_matrix: bool = False,
) -> dict[str, float]:

    model.eval()
    test_loss, test_acc = 0.0, 0.0

    precision_metric = MulticlassPrecision(num_classes=num_classes, average=average).to(
        device
    )
    recall_metric = MulticlassRecall(num_classes=num_classes, average=average).to(
        device
    )
    f1_metric = MulticlassF1Score(num_classes=num_classes, average=average).to(device)
    auroc_metric = MulticlassAUROC(num_classes=num_classes, average=average).to(device)
    precision_per_class = MulticlassPrecision(num_classes=num_classes, average=None).to(
        device
    )
    recall_per_class = MulticlassRecall(num_classes=num_classes, average=None).to(
        device
    )
    f1_per_class = MulticlassF1Score(num_classes=num_classes, average=None).to(device)
    auroc_per_class = MulticlassAUROC(num_classes=num_classes, average=None).to(device)

    confusion_matrix_metric = (
        MulticlassConfusionMatrix(num_classes=num_classes).to(device)
        if track_confusion_matrix
        else None
    )

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred = model(X)
            preds = test_pred.argmax(dim=1)
            probs = torch.softmax(test_pred, dim=1)

            # 2. Loss
            test_loss += loss_fn(test_pred, y).item()
            test_acc += accuracy_fn(y_true=y, y_pred=preds)
            precision_metric.update(preds, y)
            recall_metric.update(preds, y)
            f1_metric.update(preds, y)
            auroc_metric.update(probs, y)
            precision_per_class.update(preds, y)
            recall_per_class.update(preds, y)
            f1_per_class.update(preds, y)
            auroc_per_class.update(probs, y)
            if confusion_matrix_metric is not None:
                confusion_matrix_metric.update(preds, y)

    test_loss /= len(dataloader)
    test_acc /= len(dataloader)

    results = {
        "loss": test_loss,
        "acc": test_acc,
        "precision": precision_metric.compute().item(),
        "recall": recall_metric.compute().item(),
        "f1": f1_metric.compute().item(),
        "auroc": auroc_metric.compute().item(),
        "precision_per_class": precision_per_class.compute().tolist(),
        "recall_per_class": recall_per_class.compute().tolist(),
        "f1_per_class": f1_per_class.compute().tolist(),
        "auroc_per_class": auroc_per_class.compute().tolist(),
    }
    if confusion_matrix_metric is not None:
        results["confusion_matrix"] = confusion_matrix_metric

    return results


def train(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    num_classes: int,
    class_names: list[str] | None = None,
    accuracy_fn: Callable = accuracy_fn,
    loss_fn: nn.Module | None = None,
    epochs: int = 5,
    average: str = "macro",
    device: torch.device = "cpu",
    experiment_name: str = "model_training",
    run_name: str | None = None,
    dataset_name: str | None = None,
    registered_model_name: str | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    tracking_uri: str | None = None,
) -> dict[str, list[float]]:

    import matplotlib.pyplot as plt
    import mlflow
    import mlflow.pyfunc
    from mlflow.models import infer_signature
    from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
    from tqdm.auto import tqdm

    from model.serving import SoftmaxClassifier

    metric_keys = ["loss", "acc", "precision", "recall", "f1", "auroc"]
    results = {f"train_{k}": [] for k in metric_keys}
    results.update({f"test_{k}": [] for k in metric_keys})

    class_labels = (
        class_names
        if class_names and len(class_names) == num_classes
        else [f"class_{i}" for i in range(num_classes)]
    )

    if tracking_uri is not None:
        # Azure ML's MLflow tracking server manages experiment creation and
        # artifact storage itself -- no local artifact_location needed.
        mlflow.set_tracking_uri(tracking_uri)
    else:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        if mlflow.get_experiment_by_name(experiment_name) is None:
            mlflow.create_experiment(
                experiment_name, artifact_location=Path("mlruns").resolve().as_uri()
            )
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        optimizer_params = optimizer.param_groups[0]

        # Best-effort dataset metadata: torchvision ImageFolder exposes `.root`,
        # any Dataset exposes `len()` -- both are optional so this stays generic.
        train_dataset = getattr(train_dataloader, "dataset", None)
        test_dataset = getattr(test_dataloader, "dataset", None)
        dataset_params = {
            "dataset_name": dataset_name,
            "train_data_dir": str(getattr(train_dataset, "root", "")) or None,
            "test_data_dir": str(getattr(test_dataset, "root", "")) or None,
            "num_train_samples": len(train_dataset)
            if train_dataset is not None
            else None,
            "num_test_samples": len(test_dataset) if test_dataset is not None else None,
        }
        dataset_params = {k: v for k, v in dataset_params.items() if v is not None}

        mlflow.log_params(
            {
                "model": model.__class__.__name__,
                "loss_fn": loss_fn.__class__.__name__,
                "optimizer": optimizer.__class__.__name__,
                "lr": optimizer_params.get("lr"),
                "weight_decay": optimizer_params.get("weight_decay"),
                "scheduler": scheduler.__class__.__name__
                if scheduler is not None
                else "None",
                "batch_size": train_dataloader.batch_size,
                "epochs": epochs,
                "device": str(device),
                "num_classes": num_classes,
                "metric_average": average,
                **dataset_params,
            }
        )

        test_metrics = {}
        for epoch in tqdm(range(epochs)):
            epoch_start = time.time()

            train_metrics = train_step(
                model=model,
                dataloader=train_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                num_classes=num_classes,
                accuracy_fn=accuracy_fn,
                average=average,
                device=device,
            )

            is_last_epoch = epoch == epochs - 1
            test_metrics = test_step(
                model=model,
                dataloader=test_dataloader,
                loss_fn=loss_fn,
                num_classes=num_classes,
                accuracy_fn=accuracy_fn,
                average=average,
                device=device,
                track_confusion_matrix=is_last_epoch,
            )

            epoch_time = time.time() - epoch_start

            print(
                f"Epoch: {epoch} | "
                f"Train loss: {train_metrics['loss']:.2f} | Train acc: {train_metrics['acc']:.2f}% | "
                f"Test loss: {test_metrics['loss']:.2f} | Test acc: {test_metrics['acc']:.2f}%"
            )

            metrics_to_log = {
                "loss/train": round(train_metrics["loss"], 2),
                "acc/train": round(train_metrics["acc"], 2),
                "precision/train": round(train_metrics["precision"], 2),
                "recall/train": round(train_metrics["recall"], 2),
                "f1/train": round(train_metrics["f1"], 2),
                "auroc/train": round(train_metrics["auroc"], 2),
                "loss/test": round(test_metrics["loss"], 2),
                "acc/test": round(test_metrics["acc"], 2),
                "precision/test": round(test_metrics["precision"], 2),
                "recall/test": round(test_metrics["recall"], 2),
                "f1/test": round(test_metrics["f1"], 2),
                "auroc/test": round(test_metrics["auroc"], 2),
                "epoch_time_sec": round(epoch_time, 2),
                "lr": optimizer.param_groups[0]["lr"],
            }

            if scheduler is not None:
                scheduler.step()

            for i, label in enumerate(class_labels):
                metrics_to_log[f"precision/train/{label}"] = round(
                    train_metrics["precision_per_class"][i], 2
                )
                metrics_to_log[f"recall/train/{label}"] = round(
                    train_metrics["recall_per_class"][i], 2
                )
                metrics_to_log[f"f1/train/{label}"] = round(
                    train_metrics["f1_per_class"][i], 2
                )
                metrics_to_log[f"auroc/train/{label}"] = round(
                    train_metrics["auroc_per_class"][i], 2
                )
                metrics_to_log[f"precision/test/{label}"] = round(
                    test_metrics["precision_per_class"][i], 2
                )
                metrics_to_log[f"recall/test/{label}"] = round(
                    test_metrics["recall_per_class"][i], 2
                )
                metrics_to_log[f"f1/test/{label}"] = round(
                    test_metrics["f1_per_class"][i], 2
                )
                metrics_to_log[f"auroc/test/{label}"] = round(
                    test_metrics["auroc_per_class"][i], 2
                )

            mlflow.log_metrics(metrics_to_log, step=epoch)

            for k in metric_keys:
                results[f"train_{k}"].append(train_metrics[k])
                results[f"test_{k}"].append(test_metrics[k])

        confusion_matrix_metric = test_metrics.get("confusion_matrix")
        if confusion_matrix_metric is not None:
            fig, _ = confusion_matrix_metric.plot(labels=class_names)
            mlflow.log_figure(fig, "confusion_matrix.png")
            plt.close(fig)

        # Log the trained model as a deployable MLflow Model (signature + input
        # example let `mlflow models serve` / the Model Registry validate inputs).
        model.eval()
        sample_inputs, _ = next(iter(test_dataloader))
        sample_inputs = sample_inputs.to(device)

        signature = infer_signature(sample_inputs.cpu().numpy())

        model_dir = str(Path(__file__).resolve().parent.parent / "model")

        # Serve softmax probabilities per class instead of raw logits. This wraps the
        # trained model rather than changing its forward() -- CrossEntropyLoss already
        # applies softmax internally during training, so baking it into the model itself
        # would double-apply it and break training. See model/serving.py.
        # Log CPU-resident weights: torch.load() has no map_location by default, so a
        # CUDA-tagged checkpoint fails to deserialize on a GPU-less serving container.
        python_model = SoftmaxClassifier(
            model=model.to("cpu"), class_names=class_labels
        )

        if tracking_uri is not None and tracking_uri.startswith("azureml://"):
            # azureml-mlflow only implements the classic MLflow API (<=2.16) -- it has no
            # /api/2.0/mlflow/logged-models endpoint. mlflow>=3's log_model() always calls
            # that endpoint via Model.log() -> _create_logged_model(), regardless of
            # flavor, so it 404s against Azure ML's native tracking endpoint. Work around
            # it by saving the model locally (no REST call) then uploading it as a plain
            # run artifact and registering it through the classic Model Registry API,
            # both of which azureml-mlflow does support.
            with tempfile.TemporaryDirectory() as tmp_dir:
                local_model_path = str(Path(tmp_dir) / "model")
                mlflow.pyfunc.save_model(
                    path=local_model_path,
                    python_model=python_model,
                    signature=signature,
                    input_example=sample_inputs.cpu().numpy()[:1],
                    code_paths=[model_dir],
                )
                mlflow.log_artifacts(local_model_path, artifact_path="model")

            if registered_model_name is not None:
                # mlflow.register_model() also 404s here: it internally falls back to
                # _get_logged_models_from_run() -> search_logged_models(), which is the
                # same unsupported /logged-models endpoint. Bypass it with the classic,
                # lower-level MlflowClient calls (create_registered_model +
                # create_model_version), which azureml-mlflow does support -- confirmed
                # working end-to-end against the AMLS workspace.
                run_id = mlflow.active_run().info.run_id
                client = mlflow.MlflowClient()
                try:
                    client.create_registered_model(registered_model_name)
                except mlflow.exceptions.MlflowException as e:
                    if e.error_code not in (
                        "RESOURCE_ALREADY_EXISTS",
                        "ALREADY_EXISTS",
                    ):
                        raise
                source = RunsArtifactRepository.get_underlying_uri(
                    f"runs:/{run_id}/model"
                )
                client.create_model_version(
                    name=registered_model_name, source=source, run_id=run_id
                )
        else:
            mlflow.pyfunc.log_model(
                name=run_name,
                python_model=python_model,
                signature=signature,
                input_example=sample_inputs.cpu().numpy()[:1],
                registered_model_name=registered_model_name,
                code_paths=[model_dir],
            )

    return results
