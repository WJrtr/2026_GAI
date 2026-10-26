"""Brainrot dataset + label vocabulary (deterministic, built from train.csv)."""
import csv
import json
import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Resolve data paths relative to the project root so scripts work from any cwd.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
TRAIN_CSV = os.path.join(DATA, "train.csv")
VOCAB_JSON = os.path.join(DATA, "vocab.json")
TRAINSET = os.path.join(DATA, "trainset")


def build_vocab(csv_path=TRAIN_CSV, save_path=VOCAB_JSON):
    animals, objects = set(), set()
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            animals.add(r["animal"])
            objects.add(r["object"])
    animals, objects = sorted(animals), sorted(objects)
    vocab = {
        "animals": animals,
        "objects": objects,
        "a2i": {a: i for i, a in enumerate(animals)},
        "o2i": {o: i for i, o in enumerate(objects)},
    }
    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
    return vocab


def load_vocab(path=VOCAB_JSON, csv_path=TRAIN_CSV):
    if not os.path.exists(path):
        return build_vocab(csv_path, path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def prompt_of(animal, obj):
    return f"a {animal} and a {obj}"


class BrainrotDataset(Dataset):
    """Returns (image[-1,1], animal_idx, object_idx, prompt_str)."""

    def __init__(self, root=TRAINSET, csv_path=TRAIN_CSV,
                 image_size=64, hflip=True, vocab=None):
        self.root = root
        self.image_size = image_size
        self.vocab = vocab or load_vocab(csv_path=csv_path)
        self.rows = []
        with open(csv_path, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                self.rows.append(r)
        tfm = []
        if hflip:
            tfm.append(transforms.RandomHorizontalFlip())
        tfm += [transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3)]  # -> [-1, 1]
        self.tf = transforms.Compose(tfm)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(os.path.join(self.root, r["id"])).convert("RGB")
        x = self.tf(img)
        a = self.vocab["a2i"][r["animal"]]
        o = self.vocab["o2i"][r["object"]]
        return x, a, o, prompt_of(r["animal"], r["object"])


if __name__ == "__main__":
    v = build_vocab()
    print("animals:", v["animals"])
    print("objects:", v["objects"])
    ds = BrainrotDataset()
    x, a, o, p = ds[0]
    print("sample:", x.shape, x.min().item(), x.max().item(), a, o, p)
    print("len:", len(ds))
