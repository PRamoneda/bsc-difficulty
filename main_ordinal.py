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
PATIENCE = 10
MAX_LEN = 150  # máximo número de tracks
BATCH_SIZE = 32
MAX_EPOCHS = 10000
HIDDEN_DIM = 128
LOCAL = False
ALIAS = "my_first_prob_bs4"


class SimpleAveragingClassifier(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim=512, lr=0.0005, split_id=0):
        super().__init__()
        self.save_hyperparameters()
        # Atención para reducir los tracks
        self.attn = nn.Linear(input_dim, 1)

        # Proyección y capas no lineales
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Clasificación final
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, NUM_CLASSES)
        )

        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.split_id = split_id

    def forward(self, x, lengths=None):
        B, N, D = x.shape

        # Padding o truncado
        if N < MAX_LEN:
            pad = torch.zeros(B, MAX_LEN - N, D, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
        elif N > MAX_LEN:
            x = x[:, :MAX_LEN]

        # Atención para ponderar los tracks
        attn_weights = torch.softmax(self.attn(x), dim=1)  # [B, N, 1]
        x = (x * attn_weights).sum(dim=1)  # [B, D]

        # Proyección y clasificación
        x = self.projection(x)
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        x, y, lengths = batch
        logits = self(x, lengths)
        self.log(f"train_loss_{self.split_id}", nn.functional.cross_entropy(logits, y), on_step=False, on_epoch=True, prog_bar=True)
        return nn.functional.cross_entropy(logits, y)

    def validation_step(self, batch, batch_idx):
        x, y, lengths = batch
        logits = self(x, lengths)
        preds = torch.argmax(logits, dim=1)
        self.validation_step_outputs.append({"y_true": y.cpu(), "y_pred": preds.cpu()})
        # Log validation loss
        loss = nn.functional.cross_entropy(logits, y)
        self.log(f"val_loss_{self.split_id}", loss, on_step=False, on_epoch=True, prog_bar=True)
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

        self.log(f"val_accuracy_{self.split_id}", acc)
        self.log(f"val_macro_mse_{self.split_id}", macro_mse)
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


def run_split(split_id, all_splits, all_embeddings):
    split = all_splits[str(split_id)]
    train_dataset = PianoSyllabusDataset(split["train"], all_embeddings)
    val_dataset = PianoSyllabusDataset(split["val"], all_embeddings)
    test_dataset = PianoSyllabusDataset(split["test"], all_embeddings)

    first_sample = train_dataset[0][0]
    input_dim = first_sample.shape[-1]

    model = SimpleAveragingClassifier(input_dim=input_dim, hidden_dim=HIDDEN_DIM, split_id=split_id)

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
        EarlyStopping(monitor=f"val_accuracy_{split_id}", mode="max", patience=PATIENCE, verbose=True)
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

    # Cargar splits
    with open(json_path) as f:
        all_splits = json.load(f)

    # Obtener todos los nombres de piezas únicos en los splits
    all_piece_names = set()
    for split in all_splits.values():
        for part in ["train", "val", "test"]:
            all_piece_names.update(split[part].keys())

    # Cargar todos los embeddings una vez
    all_embeddings = {}
    for piece_name in all_piece_names:
        emb_path = embeddings_dir / piece_name[:3] / f"{piece_name}.pt"
        if emb_path.exists():
            embedding = torch.load(emb_path)
            if embedding.ndim == 4 and embedding.size(0) == 1:
                embedding = embedding.squeeze(0)
            embedding = embedding.permute(2, 0, 1)
            embedding = embedding.clone().detach().float()
            embedding = embedding.mean(dim=-1).transpose(0, 1)
            all_embeddings[piece_name] = embedding
        else:
            print(f"[INFO] Excluido: {emb_path} no existe")

    accs, mses = [], []
    for split_id in range(5):
        print(f"=== Split {split_id} ===")
        acc, mse = run_split(split_id, all_splits, all_embeddings)
        accs.append(acc)
        mses.append(mse)
        print(f"[VAL] Balanced Accuracy: {acc:.4f} | Macro MSE: {mse:.4f}")

    print("\n=== Resultados Finales ===")
    print(f"Balanced Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"Macro MSE:         {np.mean(mses):.4f} ± {np.std(mses):.4f}")
