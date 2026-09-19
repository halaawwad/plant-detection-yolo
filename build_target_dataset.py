import random
import shutil
from pathlib import Path

import cv2
import numpy as np


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def collect_images(directory: Path) -> list[Path]:
    images: list[Path] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        images.extend(sorted(directory.glob(pattern)))
    return images


def augment_variants(image: np.ndarray) -> list[tuple[str, np.ndarray]]:
    variants: list[tuple[str, np.ndarray]] = [("orig", image)]
    variants.append(("flip", cv2.flip(image, 1)))

    for angle in (-12, 12):
        height, width = image.shape[:2]
        center = (width / 2, height / 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        variants.append((f"rot{angle:+d}", rotated))

    brighter = cv2.convertScaleAbs(image, alpha=1.05, beta=12)
    darker = cv2.convertScaleAbs(image, alpha=0.92, beta=-10)
    variants.append(("bright", brighter))
    variants.append(("dark", darker))
    return variants


def center_crop_variant(image: np.ndarray, ratio: float, x_bias: float = 0.0, y_bias: float = 0.0) -> np.ndarray:
    height, width = image.shape[:2]
    crop_width = max(32, int(width * ratio))
    crop_height = max(32, int(height * ratio))
    max_x = max(width - crop_width, 0)
    max_y = max(height - crop_height, 0)
    x = int(max_x * (0.5 + x_bias))
    y = int(max_y * (0.5 + y_bias))
    x = min(max(x, 0), max_x)
    y = min(max(y, 0), max_y)
    return image[y : y + crop_height, x : x + crop_width]


def hard_negative_variants(image: np.ndarray) -> list[tuple[str, np.ndarray]]:
    variants: list[tuple[str, np.ndarray]] = []
    seen_shapes = set()

    def add_variant(name: str, variant: np.ndarray) -> None:
        if variant.size == 0:
            return
        key = (name, variant.shape[0], variant.shape[1])
        if key in seen_shapes:
            return
        seen_shapes.add(key)
        variants.append((name, variant))

    for name, variant in augment_variants(image):
        add_variant(name, variant)

    crop_specs = [
        ("center", 0.8, 0.0, 0.0),
        ("left", 0.75, -0.45, 0.0),
        ("right", 0.75, 0.45, 0.0),
        ("top", 0.75, 0.0, -0.45),
        ("bottom", 0.75, 0.0, 0.45),
    ]
    for name, ratio, x_bias, y_bias in crop_specs:
        cropped = center_crop_variant(image, ratio=ratio, x_bias=x_bias, y_bias=y_bias)
        add_variant(name, cropped)
        add_variant(f"{name}_flip", cv2.flip(cropped, 1))

    return variants


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    debug_samples_dir = project_dir / "pi_deploy" / "debug_samples"
    rose_boost_dir = project_dir / "rose_boost"
    source_dirs = [
        rose_boost_dir,
        project_dir / "NEW",
        project_dir / "plant12",
        debug_samples_dir / "plant",
    ]
    soil_dir = project_dir / "Dataset"
    target_dir = project_dir / "Dataset_target_specific"
    hard_negative_dirs = [
        debug_samples_dir / "soil",
        debug_samples_dir / "other",
    ]

    positive_images = []
    for source_dir in source_dirs:
        if source_dir.exists():
            images = collect_images(source_dir)
            positive_images.extend(images)
            if source_dir == rose_boost_dir:
                # Give the rose examples a small extra weight without flooding the dataset.
                positive_images.extend(images)

    if len(positive_images) < 8:
        joined_sources = ", ".join(str(path) for path in source_dirs)
        raise RuntimeError(f"Expected at least 8 positive images across: {joined_sources}, found {len(positive_images)}")

    random.seed(42)
    shuffled = positive_images[:]
    random.shuffle(shuffled)
    total = len(shuffled)
    train_count = max(12, int(total * 0.7))
    val_count = max(3, int(total * 0.15))
    if train_count + val_count >= total:
        val_count = max(2, total - train_count - 1)
    test_count = max(1, total - train_count - val_count)

    train_pos = shuffled[:train_count]
    val_pos = shuffled[train_count : train_count + val_count]
    test_pos = shuffled[train_count + val_count : train_count + val_count + test_count]

    negative_sources = {
        "train": sorted((soil_dir / "train" / "soil").glob("*.*")),
        "val": sorted((soil_dir / "val" / "soil").glob("*.*")),
        "test": sorted((soil_dir / "test" / "soil").glob("*.*")),
    }

    hard_negative_images = []
    for hard_negative_dir in hard_negative_dirs:
        if hard_negative_dir.exists():
            for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                hard_negative_images.extend(sorted(hard_negative_dir.glob(pattern)))

    if not all(negative_sources.values()):
        raise RuntimeError("Soil negative images are missing from Dataset.")

    ensure_clean_dir(target_dir)
    for split in ("train", "val", "test"):
        (target_dir / split / "plant").mkdir(parents=True, exist_ok=True)
        (target_dir / split / "no_plant").mkdir(parents=True, exist_ok=True)

    # Training positives: duplicate with lightweight augmentations because NEW is intentionally small.
    train_positive_count = 0
    for source_path in train_pos:
        image = cv2.imread(str(source_path))
        if image is None:
            continue
        for suffix, augmented in augment_variants(image):
            out_name = f"{source_path.stem}_{suffix}.jpg"
            write_image(target_dir / "train" / "plant" / out_name, augmented)
            train_positive_count += 1

    for source_path in val_pos:
        image = cv2.imread(str(source_path))
        if image is not None:
            write_image(target_dir / "val" / "plant" / source_path.name, image)

    for source_path in test_pos:
        image = cv2.imread(str(source_path))
        if image is not None:
            write_image(target_dir / "test" / "plant" / source_path.name, image)

    train_hard_negative_count = 0
    for source_path in hard_negative_images:
        image = cv2.imread(str(source_path))
        if image is None:
            continue
        for suffix, augmented in hard_negative_variants(image):
            out_name = f"{source_path.stem}_{suffix}.jpg"
            write_image(target_dir / "train" / "no_plant" / out_name, augmented)
            train_hard_negative_count += 1

    # Balance negatives roughly against positives in train, with a modest val/test sample.
    random.shuffle(negative_sources["train"])
    random.shuffle(negative_sources["val"])
    random.shuffle(negative_sources["test"])

    remaining_train_negatives = max(train_positive_count - train_hard_negative_count, 0)
    train_neg = negative_sources["train"][:remaining_train_negatives]
    val_neg = negative_sources["val"][: max(10, len(val_pos) * 4)]
    test_neg = negative_sources["test"][: max(10, len(test_pos) * 8)]

    for split, items in (("train", train_neg), ("val", val_neg), ("test", test_neg)):
        for source_path in items:
            image = cv2.imread(str(source_path))
            if image is not None:
                write_image(target_dir / split / "no_plant" / source_path.name, image)

    print(f"Built target-specific dataset at: {target_dir}")
    print(f"Positive source images used: {len(positive_images)}")
    print(f"Hard negative source images used: {len(hard_negative_images)}")
    for split in ("train", "val", "test"):
        plant_count = len(list((target_dir / split / "plant").glob("*.*")))
        no_plant_count = len(list((target_dir / split / "no_plant").glob("*.*")))
        print(f"{split}: plant={plant_count}, no_plant={no_plant_count}")


if __name__ == "__main__":
    main()
