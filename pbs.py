"""Peripheral blood smear segmentation and research RBC morphology inference."""

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from skimage.segmentation import watershed
from torch import nn

CELL_TYPES = (
    "Elliptocyte / ovalocyte",
    "Pencil cell",
    "Teardrop cell",
    "Normal morphology",
    "Stomatocyte",
    "Target cell",
    "Hypochromic cell",
    "Spherocyte",
    "Acanthocyte / burr cell",
)
WORKING_SIZE = (800, 600)
MODEL_PATH = Path(__file__).resolve().parent / "dashboard_data" / "pbs_morphology.pth"
CROP_SIZE = 40
COLORS = (
    (78, 224, 209), (255, 172, 87), (255, 108, 123),
    (121, 213, 128), (202, 150, 255), (99, 189, 255),
    (255, 224, 104), (255, 143, 215), (162, 181, 255),
)


@dataclass
class Candidate:
    box: tuple[int, int, int, int]
    contour: np.ndarray
    crop: np.ndarray


def decode_image(data):
    if len(data) > 20 * 1024 * 1024:
        raise ValueError("Choose an image smaller than 20 MB.")
    try:
        with Image.open(BytesIO(data), formats=["PNG", "JPEG", "TIFF"]) as image:
            if image.width * image.height > 30_000_000:
                raise ValueError("Choose an image under 30 megapixels.")
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError("Choose a single-frame microscope image.")
            return ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as error:
        raise ValueError("The file could not be read as a PNG, JPEG or TIFF image.") from error


def segment_smear(image):
    """Return the working image, green channel and candidate RBC crops."""
    if min(image.size) < 300:
        raise ValueError("Upload a full microscope field at least 300 pixels wide and high.")

    # Preserve cell shape when the uploaded field has a different aspect ratio.
    working = np.asarray(ImageOps.contain(image, WORKING_SIZE, Image.Resampling.LANCZOS))
    green = np.ascontiguousarray(working[:, :, 1])
    smooth = cv2.medianBlur(green, 5)
    height, width = green.shape
    if float(np.percentile(smooth, 95) - np.percentile(smooth, 5)) < 15:
        return working, green, []

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    _, dark = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    edges = cv2.Canny(smooth, 12, 32)
    edges = cv2.dilate(edges, kernel)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(green)
    for contour in contours:
        if cv2.contourArea(contour) <= 8_000:
            cv2.drawContours(filled, [contour], -1, 255, cv2.FILLED)
    foreground = cv2.morphologyEx(dark | filled, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(foreground, contours, -1, 255, cv2.FILLED)

    # Separate touching cells using local peaks in the distance transform.
    distance = cv2.distanceTransform(foreground, cv2.DIST_L2, 5)
    peak_window = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    peaks = np.uint8((distance >= cv2.dilate(distance, peak_window)) & (distance >= 8))
    _, labels = cv2.connectedComponents(peaks)
    if labels.max() == 0:
        return working, green, []
    markers = watershed(-distance, labels, mask=foreground != 0, watershed_line=True)

    dark_nucleus_cutoff = 0.45 * np.percentile(smooth, 90)
    cells = []
    for label in range(1, int(labels.max()) + 1):
        mask = np.uint8(markers == label) * 255
        x, y, w, h = cv2.boundingRect(mask)
        if x <= 1 or y <= 1 or x + w >= width - 1 or y + h >= height - 1:
            continue
        local_mask = cv2.erode(mask[y:y+h, x:x+w], kernel,
                               borderType=cv2.BORDER_CONSTANT, borderValue=0)
        area = cv2.countNonZero(local_mask)
        if not 500 <= area <= 3_800 or min(w, h) < 11:
            continue

        pixels = smooth[y:y+h, x:x+w][local_mask != 0]
        # Reject very dark nuclei and very faint background fragments.
        if np.median(pixels) < dark_nucleus_cutoff:
            continue
        outlines, _ = cv2.findContours(local_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(outlines, key=cv2.contourArea)
        if len(contour) >= 5:
            axes = cv2.fitEllipse(contour)[1]
            if max(axes) > 4.5 * min(axes):
                continue
        crop = cv2.bitwise_and(green[y:y+h, x:x+w], green[y:y+h, x:x+w], mask=local_mask)
        cells.append(Candidate((x, y, w, h), contour + (x, y), crop))
    return working, green, sorted(cells, key=lambda cell: (cell.box[1], cell.box[0]))


def prepare_crop(gray):
    """Centre an isolated green-channel cell on the classifier's black canvas."""
    x, y, width, height = cv2.boundingRect(gray)
    canvas = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.uint8)
    if not width or not height:
        return canvas
    scale = (CROP_SIZE - 6) / max(width, height)
    resized = cv2.resize(gray[y:y+height, x:x+width],
                         (max(1, round(width * scale)), max(1, round(height * scale))),
                         interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    h, w = resized.shape
    top, left = (CROP_SIZE - h) // 2, (CROP_SIZE - w) // 2
    canvas[top:top+h, left:left+w] = resized
    return canvas


def morphology_model():
    layers = []
    for input_channels, output_channels in ((1, 16), (16, 32), (32, 48)):
        layers.extend((
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.MaxPool2d(2),
        ))
    return nn.Sequential(
        *layers, nn.Flatten(), nn.Linear(48 * 5 * 5, 64), nn.ReLU(),
        nn.Dropout(0.2), nn.Linear(64, len(CELL_TYPES)),
    )


def load_morphology_model(path=MODEL_PATH):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = morphology_model()
    model.load_state_dict(checkpoint["model_state"])
    return model.eval(), checkpoint["evaluation"]


@torch.inference_mode()
def classify_cells(model, crops):
    inputs = torch.from_numpy(np.stack([prepare_crop(crop) for crop in crops]))
    inputs = inputs.unsqueeze(1).float().div_(255)
    results = []
    for batch in inputs.split(64):
        rotations = torch.cat([torch.rot90(batch, k, (2, 3)) for k in range(4)])
        probabilities = model(rotations).softmax(dim=1).reshape(4, len(batch), -1).mean(dim=0)
        scores, labels = probabilities.max(dim=1)
        results.extend(zip(labels.tolist(), scores.tolist()))
    return results
