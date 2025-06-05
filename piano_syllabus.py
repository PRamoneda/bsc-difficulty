import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


# class PianoSyllabusDataset(Dataset):
#     def __init__(self, split_data: dict, embeddings_dir: Path):
#         """
#         split_data: dict con {nombre_pieza: dificultad}
#         embeddings_dir: carpeta con archivos .pt
#         """
#         self.embeddings_dir = embeddings_dir
#         self.pieces = []
#
#         for piece_name, difficulty in split_data.items():
#             emb_path = embeddings_dir / piece_name[:3] / f"{piece_name}.pt"
#             if emb_path.exists():
#                 self.pieces.append((piece_name, difficulty))
#             else:
#                 print(f"[INFO] Excluido: {emb_path} no existe")
#
#     def __len__(self):
#         return len(self.pieces)
#
#     def __getitem__(self, idx):
#         piece_name, difficulty = self.pieces[idx]
#         emb_path = self.embeddings_dir / piece_name[:3] / f"{piece_name}.pt"
#         embedding = torch.load(emb_path)
#
#         # Si viene con dimensión extra tipo [1, N_tracks, 562, 1024], la quitamos
#         if embedding.ndim == 4 and embedding.size(0) == 1:
#             embedding = embedding.squeeze(0)
#
#         embedding = embedding.clone().detach().float()
#         label = torch.tensor(difficulty, dtype=torch.long)
#         return embedding, label

# class PianoSyllabusDataset(Dataset):
#     def __init__(self, split_data: dict, embeddings_dir: Path):
#         """
#         Carga todo en memoria: lista de tuplas (embedding_tensor, label_tensor)
#         """
#         self.samples = []
#
#         for piece_name, difficulty in list(split_data.items()):
#             emb_path = embeddings_dir / piece_name[:3] / f"{piece_name}.pt"
#             if emb_path.exists():
#                 embedding = torch.load(emb_path)
#
#
#                 if embedding.ndim == 4 and embedding.size(0) == 1:
#                     embedding = embedding.squeeze(0)
#
#                 embedding = embedding.permute(2, 0, 1)
#                 embedding = embedding.clone().detach().float()
#                 embedding = embedding.mean(dim=-1).transpose(0, 1)  # [B, D, 150]
#
#                 print(embedding.shape)
#                 label = torch.tensor(difficulty, dtype=torch.long)
#
#                 self.samples.append((embedding, label))
#             else:
#                 print(f"[INFO] Excluido: {emb_path} no existe")
#
#     def __len__(self):
#         return len(self.samples)
#
#     def __getitem__(self, idx):
#         return self.samples[idx]


class PianoSyllabusDataset(Dataset):
    def __init__(self, split_data: dict, embeddings_dict: dict):
        self.samples = []
        for piece_name, difficulty in split_data.items():
            if piece_name in embeddings_dict:
                embedding = embeddings_dict[piece_name]
                label = torch.tensor(difficulty, dtype=torch.long)
                self.samples.append((embedding, label))
            else:
                print(f"[INFO] Embedding no encontrado para {piece_name}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn_with_dynamic_padding(batch):
    """
    batch: lista de tuplas (embedding_tensor, label)
    embedding_tensor: [tracks, 562, dim]
    """
    embeddings, labels = zip(*batch)
    max_tracks = max(e.size(0) for e in embeddings)
    dim = embeddings[0].size(-1)

    padded_embeddings = torch.zeros(len(batch), max_tracks, dim)
    lengths = []

    for i, emb in enumerate(embeddings):
        n_tracks = emb.size(0)
        padded_embeddings[i, :n_tracks] = emb
        lengths.append(n_tracks)

    labels = torch.stack(labels)  # tamaño [B]
    lengths = torch.tensor(lengths, dtype=torch.long)  # tamaño [B]
    return padded_embeddings, labels, lengths


if __name__ == "__main__":
    json_path = "split_audio.json"
    embeddings_dir = Path("/gpfs/projects/upf97/embeddings_ssl/pq1scuuq/PianoSyllabus")

    with open(json_path) as f:
        all_splits = json.load(f)

    split_data = all_splits["0"]["train"]
    dataset = PianoSyllabusDataset(split_data, embeddings_dir)

    print(f"[INFO] Dataset cargado con {len(dataset)} muestras.")

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        collate_fn=collate_fn_with_dynamic_padding,
        shuffle=True
    )

    for batch in dataloader:
        padded_embeddings, labels, lengths = batch
        print(f"[DEBUG] padded_embeddings: {padded_embeddings.shape}")  # [B, max_tracks, 562, 1024]
        print(f"[DEBUG] labels: {labels}")  # [B]
        print(f"[DEBUG] lengths: {lengths}")  # [B]
        break
