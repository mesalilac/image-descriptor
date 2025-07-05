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

category_map = {
    "People": "a photo of one or more people",
    "Animals": "an image of an animal, such as a pet or wildlife",
    "Nature": "a picture of a natural landscape, like mountains, forests, or oceans",
    "City": "a photo of a city, urban scene, or architecture",
    "Objects": "an image of an everyday object, item, or product",
    "Vehicles": "a photo of a vehicle, like a car, truck, or motorcycle",
    "Food": "an image of food, a meal, or drinks",
    "Abstract": "an abstract image or non-representational art",
    "Art": "a piece of art, a painting, or a sculpture",
    "Events": "a photo of an event or celebration, like a party or festival",
    "Documents": "an image of a document, text, or a screenshot",
    "Outdoor": "a general outdoor scene, such as a park or street",
    "Indoor": "a general indoor scene, such as a room or office",
    "Sky": "an image of the sky, clouds, sunset, or stars",
    "Sports": "a photo of a sport or athletic activity",
    "Technology": "an image showing technology, gadgets, or computers",
    "Buildings": "a picture of a building or structure",
    "Water": "a photo of water, like a river, lake, or ocean",
    "Plants": "an image of plants, flowers, or greenery",
    "Other": "an uncategorized image, something that doesn't fit other categories",
}

# Extract the CLIP-friendly phrases for the model
clip_categories = list(category_map.values())
# Create a reverse map to get directory name from CLIP category phrase
reverse_category_map = {v: k for k, v in category_map.items()}

image_paths: dict[str, str] = {}


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

    inputs_text = clip_processor(
        text=clip_categories, return_tensors="pt", padding=True
    )
    text_features = clip_model.get_text_features(**inputs_text)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    similarity = torch.matmul(image_features, text_features.T)
    max_score, max_idx = torch.max(similarity, dim=1)

    # You can still use a threshold if you want a stronger "uncategorized" guard,
    # or rely solely on the "Other" category if its score is highest.
    threshold = 0.1  # Adjust based on your testing

    predicted_clip_phrase = clip_categories[max_idx.item()]

    if predicted_clip_phrase == category_map["Other"] or max_score.item() < threshold:
        return "Other"  # Return the directory name "Other"
    else:
        return reverse_category_map[predicted_clip_phrase]


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

    x, y = image.size

    signature = blake3_file(path)

    image_paths[signature] = path

    return {
        "signature": signature,  # Command is `b3sum` on archlinux
        "size": f"{x}x{y}",
        "caption": caption,
        "category": category,
        "tags": tags,
    }


def image_is_cached(data: list[dict], signature: str) -> tuple[bool, dict | None]:
    for item in data:
        if item["signature"] == signature:
            return (True, item)
    return (False, None)


def get_unique_filename(path, size: str):
    directory, filename = os.path.split(path)
    name, ext = os.path.splitext(filename)

    name = f"{name}_{size}{ext}"
    filename = name

    candidate = filename
    counter = 1

    while os.path.exists(os.path.join(directory, candidate)):
        candidate = f"{name} copy{counter if counter > 1 else ''}{ext}"
        counter += 1

    return os.path.join(directory, candidate)


def main(folder, yes, flat, metadata_file_path):
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
                image_paths[item["signature"]] = full_path
                results.append(item)
                continue

        data = process_image(full_path)
        if data:
            results.append(data)

    save_unique_metadata(results, metadata_file_path)

    if len(results) > 0:
        print("Do you want to move images")

        if yes:
            confirm = "y"
        else:
            confirm = input("Confirm? (y/n): ")

        if confirm.lower() == "y":
            for ele in results:
                signature = ele["signature"]
                caption = ele["caption"]
                category = ele["category"]
                size = ele["size"]

                old_file_path = image_paths[signature]

                new_file_name = generate_filename(caption, old_file_path)
                if flat:
                    new_directory = folder
                else:
                    new_directory = os.path.join(folder, category)

                new_file_path = os.path.join(new_directory, new_file_name)

                new_unique_file_path = get_unique_filename(new_file_path, size)

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
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip confirmation prompt for reorganizing images.",
    )
    parser.add_argument(
        "-f",
        "--flat",
        action="store_true",
        help="Keep images in the root of the target directory, and don't move them into subdirectories",
    )

    args = parser.parse_args()
    folder_path = os.path.expanduser(args.folder)
    metadata_file_path = os.path.join(folder_path, "wallpaper_metadata.json")

    main(folder_path, args.yes, args.flat, metadata_file_path)
