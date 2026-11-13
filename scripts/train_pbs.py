"""Train the RBC morphology baseline from the nine official Mendeley ZIP files."""

import argparse
from collections import Counter
from io import BytesIO
from pathlib import Path
import re
import sys
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pbs import CELL_TYPES, CROP_SIZE, MODEL_PATH, morphology_model, prepare_crop

EXPECTED_COUNTS = (1211, 24, 2076, 1426, 382, 851, 222, 562, 354)


def read_dataset(root):
    images, labels, groups = [], [], []
    for class_number, expected in enumerate(EXPECTED_COUNTS, start=1):
        paths = sorted(root.glob(f"{class_number} *.zip"))
        if len(paths) != 1:
            raise ValueError(f"Expected one official archive starting with {class_number} in {root}.")
        with ZipFile(paths[0]) as archive:
            files = sorted(name for name in archive.namelist() if name.lower().endswith(".png"))
            if len(files) != expected:
                raise ValueError(f"{paths[0].name} contains {len(files)} images, expected {expected}.")
            for file in files:
                match = re.fullmatch(r"(.+) objek#\d+.*", Path(file).stem, flags=re.IGNORECASE)
                if match is None:
                    raise ValueError(f"Unable to identify the parent microscope field for {file}.")
                with Image.open(BytesIO(archive.read(file))) as crop:
                    images.append(prepare_crop(np.asarray(crop.convert("L"))))
                labels.append(class_number - 1)
                groups.append(match[1].casefold())
    return np.stack(images), np.array(labels), np.array(groups)


def augment(images, rng):
    output = np.empty_like(images)
    for index, image in enumerate(images):
        matrix = cv2.getRotationMatrix2D(((CROP_SIZE - 1) / 2,) * 2,
                                         rng.uniform(-180, 180), rng.uniform(0.9, 1.08))
        warped = cv2.warpAffine(image, matrix, (CROP_SIZE, CROP_SIZE))
        output[index] = np.clip(warped * rng.uniform(0.82, 1.18), 0, 255)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", type=Path, help="Directory of the nine RBCdataset ZIP archives")
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")

    torch.manual_seed(42)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    rng = np.random.default_rng(42)
    images, labels, groups = read_dataset(args.archives)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train, test = next(
        (train, test) for train, test in splitter.split(images, labels, groups)
        if len(set(labels[train])) == len(set(labels[test])) == len(CELL_TYPES)
    )
    assert set(groups[train]).isdisjoint(groups[test])

    model = morphology_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    counts = np.bincount(labels[train])
    weights = torch.tensor((counts.max() / counts) ** 0.55, dtype=torch.float32)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in np.array_split(rng.permutation(train), (len(train) + 95) // 96):
            inputs = torch.from_numpy(augment(images[batch], rng)).unsqueeze(1).float().div_(255)
            loss = criterion(model(inputs), torch.from_numpy(labels[batch]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()
        print(f"Epoch {epoch + 1:02d} | training loss {np.mean(losses):.3f}", flush=True)

    model.eval()
    predictions = []
    with torch.inference_mode():
        for batch in np.array_split(test, (len(test) + 95) // 96):
            inputs = torch.from_numpy(images[batch]).unsqueeze(1).float().div_(255)
            rotations = torch.cat([torch.rot90(inputs, k, (2, 3)) for k in range(4)])
            probabilities = model(rotations).softmax(dim=1).reshape(4, len(batch), -1).mean(dim=0)
            predictions.extend(probabilities.argmax(dim=1).tolist())

    evaluation = {
        "train_crops": int(len(train)),
        "test_crops": int(len(test)),
        "test_fields": int(len(set(groups[test]))),
        "macro_f1": float(f1_score(labels[test], predictions, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(labels[test], predictions)),
        "test_support": [int(Counter(labels[test])[index]) for index in range(len(CELL_TYPES))],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "evaluation": evaluation}, args.output)
    print(evaluation)


if __name__ == "__main__":
    main()
