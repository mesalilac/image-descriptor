import os
import hashlib
import json
import re
import shutil
import argparse
from blake3 import blake3
from PIL import Image
from tqdm import tqdm
import torch
from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
    CLIPProcessor,
    CLIPModel,
)


# Initialize models (will download on first run)
blip_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")

clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Define your own categories
categories = [
    # Nature & Environment
    "nature",  # Forests, mountains, oceans, etc.
    "sky",  # Clouds, sunset, night, stars
    "animals",  # Wildlife, pets, birds
    # Urban & Man-Made
    "cityscape",  # Skylines, streets, architecture
    "architecture",  # Bridges, ruins, buildings
    # People & Characters
    "portrait",  # Realistic or stylized people
    "anime",  # Anime and manga style characters/scenes
    "fantasy",  # Dragons, elves, mythical creatures
    "sci_fi",  # Mecha, spaceships, cyberpunk, etc.
    # Art & Design
    "abstract",  # Non-representational art
    "digital_art",  # Stylized illustrations, concept art
    "minimal",  # Clean, simple, space-efficient
    # Mood & Style
    "dark",  # Night, low light, moody
    "bright",  # Colorful, vibrant
    "vaporwave",  # Retro-futuristic, glitch, synthwave
    # Technology & Objects
    "technology",  # Gadgets, computers, machinery
    "vehicles",  # Cars, spaceships, motorcycles
]


def sanitize_tags(text):
    text = re.sub(r"[^\w\s]", "", text.lower())
    return list(set(word for word in text.split() if len(word) > 2))


def generate_caption(image):
    inputs = blip_processor(image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    caption = caption.rstrip()

    return caption


def classify_category(image):
    inputs_image = clip_processor(images=image, return_tensors="pt")
    image_features = clip_model.get_image_features(**inputs_image)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    inputs_text = clip_processor(text=categories, return_tensors="pt", padding=True)
    text_features = clip_model.get_text_features(**inputs_text)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    similarity = torch.matmul(image_features, text_features.T)
    # category_idx = similarity.argmax().item()
    # category_idx = max_idx.item()
    max_score, max_idx = torch.max(similarity, dim=1)

    threshold = 0.3
    if max_score.item() < threshold:
        return "uncategorized"
    else:
        # return categories[category_idx]
        return categories[max_idx.item()]


def load_existing_data(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def save_unique_metadata(new_items, json_path):
    # Load current data
    existing = load_existing_data(json_path)

    # Flatten both into list of dicts
    all_items = existing + new_items

    # Build deduplicated list based on SHA-256 signature
    seen = set()
    unique = []
    for item in all_items:
        sig = item.get("signature")
        if sig and sig not in seen:
            seen.add(sig)
            unique.append(item)

    # Save
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(unique, f, indent=4, ensure_ascii=False)


def sanitize_filename(text: str, max_length=60):
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)  # Remove special chars
    text = re.sub(r"[\s_]+", "_", text)  # Convert spaces to underscores
    return text[:max_length]


def generate_filename(caption: str, image_path: str):
    sanitized_caption = sanitize_filename(caption)

    file_extension = os.path.basename(image_path).split(".")[-1]

    return f"{sanitized_caption}.{file_extension}"


def blake3_file(path):
    hasher = blake3()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def process_image(path):
    try:
        image = Image.open(path).convert("RGB")
    except:
        return None

    caption = generate_caption(image)
    category = classify_category(image)
    tags = sanitize_tags(caption)

    return {
        "signature": blake3_file(path),  # Command is `b3sum` on archlinux
        "file_path": path,
        "caption": caption,
        "category": category,
        "tags": tags,
    }


def image_is_cached(data: list[dict], signature: str) -> tuple[bool, dict | None]:
    for item in data:
        if item["signature"] == signature:
            return (True, item)
    return (False, None)


def get_unique_filename(path):
    directory, filename = os.path.split(path)
    name, ext = os.path.splitext(filename)

    candidate = filename
    counter = 1

    while os.path.exists(os.path.join(directory, candidate)):
        candidate = f"{name} copy{counter if counter > 1 else ''}{ext}"
        counter += 1

    return os.path.join(directory, candidate)


def main(folder, metadata_file_path):
    results = []
    exesting_data = load_existing_data(metadata_file_path)

    for file in tqdm(os.listdir(folder)):
        if not file.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            continue
        full_path = os.path.join(folder, file)
        is_cached = image_is_cached(exesting_data, blake3_file(full_path))

        # If you found the file in the target that means you need to move it
        if is_cached[0]:
            item = is_cached[1]
            if item is not None:
                item["file_path"] = full_path  # Update file path
                results.append(item)
                continue

        data = process_image(full_path)
        if data:
            results.append(data)

    save_unique_metadata(results, metadata_file_path)

    if len(results) > 0:
        print("Do you want to move images")
        confirm = input("Confirm? (y/n): ")
        if confirm.lower() == "y":
            for ele in results:
                old_file_path = ele["file_path"]
                caption = ele["caption"]
                category = ele["category"]

                new_file_name = generate_filename(caption, old_file_path)
                new_directory = os.path.join(folder, category)
                new_file_path = os.path.join(new_directory, new_file_name)

                new_unique_file_path = get_unique_filename(new_file_path)

                print(f"'{old_file_path}' => '{new_unique_file_path}'")
                os.makedirs(new_directory, exist_ok=True)
                if os.path.isfile(old_file_path):
                    shutil.move(old_file_path, new_unique_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--folder",
        type=str,
        default="~/Pictures/wallpapers/",
        help="Wallpapers folder",
    )

    args = parser.parse_args()
    folder_path = os.path.expanduser(args.folder)
    metadata_file_path = os.path.join(folder_path, "wallpaper_metadata.json")

    main(folder_path, metadata_file_path)
