import os

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score, mean_squared_error
import numpy as np
import json
from pathlib import Path
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
from piano_syllabus import PianoSyllabusDataset, collate_fn_with_dynamic_padding

# Config
NUM_CLASSES = 12
MAX_LEN = 150  # máximo número de tracks
BATCH_SIZE = 32
MAX_EPOCHS = 10000
LOCAL = False
ALIAS = "my_first_prob"


class SimpleAveragingClassifier(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim=512, lr=0.0005):
        super().__init__()
        self.save_hyperparameters()
        self.reduce_tracks = nn.Linear(MAX_LEN, 1)
        self.projection = nn.Linear(input_dim, hidden_dim)
        self.non_linear = nn.ReLU()
        self.reduce = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(1, NUM_CLASSES)

        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x, lengths):
        B, N, D = x.shape
        if N < MAX_LEN:
            pad = torch.zeros(B, MAX_LEN - N, D, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
        elif N > MAX_LEN:
            x = x[:, :MAX_LEN]

        x = x.transpose(1, 2)
        x = self.reduce_tracks(x).squeeze(-1)  # [B, D]
        x = self.projection(x)
        x = self.non_linear(x)
        x = self.reduce(x)
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        x, y, lengths = batch
        logits = self(x, lengths)
        return nn.functional.cross_entropy(logits, y)

    def validation_step(self, batch, batch_idx):
        x, y, lengths = batch
        logits = self(x, lengths)
        preds = torch.argmax(logits, dim=1)
        self.validation_step_outputs.append({"y_true": y.cpu(), "y_pred": preds.cpu()})
        return

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        y_true = torch.cat([x["y_true"] for x in outputs])
        y_pred = torch.cat([x["y_pred"] for x in outputs])
        acc = balanced_accuracy_score(y_true, y_pred)

        class_mses = []
        for c in torch.unique(y_true):
            idx = y_true == c
            mse = mean_squared_error(y_true[idx], y_pred[idx])
            class_mses.append(mse)
        macro_mse = np.mean(class_mses)

        self.log("val_accuracy", acc)
        self.log("val_macro_mse", macro_mse)
        self.acc = acc
        self.mse = macro_mse
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        x, y, lengths = batch
        logits = self(x, lengths)
        preds = torch.argmax(logits, dim=1)
        self.test_step_outputs.append({"y_true": y.cpu(), "y_pred": preds.cpu()})

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        y_true = torch.cat([x["y_true"] for x in outputs])
        y_pred = torch.cat([x["y_pred"] for x in outputs])
        acc = balanced_accuracy_score(y_true, y_pred)

        class_mses = []
        for c in torch.unique(y_true):
            idx = y_true == c
            mse = mean_squared_error(y_true[idx], y_pred[idx])
            class_mses.append(mse)
        macro_mse = np.mean(class_mses)

        print(f"[TEST] Balanced Accuracy: {acc:.4f} | Macro MSE: {macro_mse:.4f}")
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def run_split(split_id, json_path, embeddings_dir):
    with open(json_path) as f:
        all_splits = json.load(f)
    split = all_splits[str(split_id)]
    train_dataset = PianoSyllabusDataset(split["train"], embeddings_dir)
    val_dataset = PianoSyllabusDataset(split["val"], embeddings_dir)
    test_dataset = PianoSyllabusDataset(split["test"], embeddings_dir)

    first_path = list(split["train"].keys())[0]
    sample = torch.load(embeddings_dir / first_path[:3] / f"{first_path}.pt")
    input_dim = sample.shape[-1]

    model = SimpleAveragingClassifier(input_dim=input_dim)

    run_name = f"{ALIAS}{split_id}"
    wandb_params = {
        "project": "difficulty",
        "offline": LOCAL,
        "entity": "mtg-upf",
        "save_dir": "/gpfs/projects/upf97/logs/",
        "log_model": True,
        "name": run_name,
        "group": "avgprobe",
    }
    wandb_logger = WandbLogger(**wandb_params) if not LOCAL else None

    callbacks = [
        EarlyStopping(monitor="val_accuracy", mode="max", patience=4, verbose=True)
    ]

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=MAX_EPOCHS,
        precision="16-mixed",
        log_every_n_steps=1,
        num_sanity_val_steps=1,
        check_val_every_n_epoch=5,
        logger=wandb_logger,
        callbacks=callbacks,
        # profiler = "simple"
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_with_dynamic_padding, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_with_dynamic_padding, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_with_dynamic_padding)

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)
    return model.acc, model.mse


if __name__ == "__main__":
    json_path = "split_audio.json"
    os.system("rsync -avz --progress /gpfs/projects/upf97/embeddings_ssl/pq1scuuq/PianoSyllabus /scratch/tmp")
    embeddings_dir = Path("/scratch/tmp/PianoSyllabus")

    accs, mses = [], []
    for split_id in range(5):
        print(f"=== Split {split_id} ===")
        acc, mse = run_split(split_id, json_path, embeddings_dir)
        accs.append(acc)
        mses.append(mse)
        print(f"[VAL] Balanced Accuracy: {acc:.4f} | Macro MSE: {mse:.4f}")

    print("\\n=== Resultados Finales ===")
    print(f"Balanced Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"Macro MSE:         {np.mean(mses):.4f} ± {np.std(mses):.4f}")