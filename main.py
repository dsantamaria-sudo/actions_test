import sys
from pathlib import Path

import torch
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from torch import nn
from torchvision import transforms

from data.dataLoader import create_food101_dataloaders
from model.model import tinyConvNeXt
from utils.utils import train

# mlflow prints run URLs with emoji; Windows consoles default to cp1252, which
# can't encode them and crashes the run right as it finishes.
sys.stdout.reconfigure(encoding="utf-8")


def main():

    #mlflow_tracking_uri = "http://localhost:5000"
    mlflow_tracking_uri = "https://aml-container.greencoast-3d77fdca.spaincentral.azurecontainerapps.io"
    ml_client = MLClient( # noqa: F841
        DefaultAzureCredential(),
        subscription_id="48ee8f92-9a0c-4ecd-b6ce-83845f85bf75",
        resource_group_name="AML",
        workspace_name="AMLS",
    )
    # mlflow_tracking_uri = ml_client.workspaces.get("AMLS").mlflow_tracking_uri

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = Path("data/")
    food101_dir = data_path / "food101"
    CLASSES = ["pizza", "steak", "sushi"]

    # Augmentation only on train -- test must stay deterministic so it reflects
    # real inference conditions, not artificially inflated/deflated by random crops.
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=128, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.Resize(size=(128, 128)),
        transforms.ToTensor()
    ])

    BATCH_SIZE = 32

    train_dataloader, test_dataloader, class_names = create_food101_dataloaders(
                                                                    data_dir=food101_dir,
                                                                    classes=CLASSES,
                                                                    train_transform=train_transform,
                                                                    test_transform=test_transform,
                                                                    BATCH_SIZE=BATCH_SIZE)

    # display_random_images(train_dataloader,
    #                       n=5,
    #                       classes=class_names,
    #                       seed=None)

    model = tinyConvNeXt(input_features=3, output_features=len(class_names), hidden_units=96).to(device)

    #Loss function
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    #Optimizer (ConvNeXt is trained with AdamW in the original paper)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    EPOCHS = 6  
    WARMUP_EPOCHS = 5

    # Warmup (avoids the AdamW cold-start loss spike seen in early epochs) then
    # cosine decay (shrinks the step size as training progresses, instead of
    # orbiting a good minimum forever at a constant lr).
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=WARMUP_EPOCHS)
    cosine_decay = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS - WARMUP_EPOCHS)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine_decay], milestones=[WARMUP_EPOCHS])

    run_name = f"{model.__class__.__name__}_v6"

    results = train(model=model,
                     train_dataloader=train_dataloader,
                     test_dataloader=test_dataloader,
                     optimizer=optimizer,
                     num_classes=len(class_names),
                     class_names=class_names,
                     loss_fn=loss_fn,
                     epochs=EPOCHS,
                     device=device,
                     experiment_name="pizza_steak_sushi",
                     run_name=run_name,
                     dataset_name="pizza_steak_sushi",
                     scheduler=scheduler,
                     tracking_uri=mlflow_tracking_uri,
                     registered_model_name=run_name)

    return results

if __name__ == "__main__":
    main()
