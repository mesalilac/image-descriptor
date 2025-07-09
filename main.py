import os
import re
import json
import shutil
import torch
import argparse
from tqdm import tqdm
from blake3 import blake3
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, pipelines

if torch.cuda.is_available():
    print(f"CUDA available! using GPU '{torch.cuda.get_device_name(0)}'")
    device = torch.device("cuda")
else:
    print("CUDA not available! using CPU")
    device = torch.device("cpu")

print("---------------------------")

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-large"
).to(device)

classifier = pipelines.pipeline(
    "zero-shot-classification", model="facebook/bart-large-mnli", device=0
)


canidate_lables = [
    "Anime",
    "Art",
    "Buildings",
    "Animals",
    "Nature",
    "People",
    "Vechicles",
    "Other",
]


def generate_caption(image_obj: Image) -> str:
    inputs = processor(image_obj, return_tensors="pt").to(device)

    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)

    return caption


def classify_category(caption: str) -> str:
    classification_results = classifier(caption, canidate_lables, multi_lable=False)

    predicted_category = classification_results["labels"][0]

    if isinstance(predicted_category, str):
        return predicted_category
    else:
        return "Other"


def blake3_file(path: str) -> str:
    hasher = blake3()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class Metadata:
    def __init__(self, filename="image_metadata.json"):
        self.filename = filename
        self.data: dict[str, dict[str, str]] = {}
        self.paths: dict[str, str] = {}

        self.__load()

    def put_path(self, signature: str, path: str):
        self.paths[signature] = path

    def get_path(self, signature: str) -> str | None:
        return self.paths.get(signature)

    def data_has_signature(self, signature: str) -> bool:
        return signature in self.data

    def put_metadata(
        self,
        signature: str,
        size: str,
        caption: str,
        category: str,
        tags: str,
    ):
        data = {
            "signature": signature,
            "size": size,
            "caption": caption,
            "category": category,
            "tags": tags,
        }
        self.data[signature] = data

    def save(self):
        with open(self.filename, "w", encoding="utf-8") as f:
            json.dump(list(self.data.values()), f, indent=4, ensure_ascii=False)

    def __load(self) -> None:
        if not os.path.exists(self.filename):
            return
        with open(self.filename, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)

                for item in data:
                    x = {
                        "signature": item["signature"],
                        "size": item["size"],
                        "caption": item["caption"],
                        "category": item["category"],
                        "tags": item["tags"],
                    }
                    self.data[item["signature"]] = x
            except json.JSONDecodeError:
                return


def sanitize_filename(text: str, max_length=60):
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)  # Remove special chars
    text = re.sub(r"[\s_]+", "_", text)  # Convert spaces to underscores
    return text[:max_length]


def generate_filename(signature: str, caption: str, image_path: str):
    sanitized_caption = sanitize_filename(caption)

    file_extension = os.path.basename(image_path).split(".")[-1]

    return f"{sanitized_caption}_{signature[:9]}.{file_extension}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-f",
        "--folder",
        type=str,
        required=True,
        help="Wallpapers folder",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip confirmation prompt for reorganizing images.",
    )
    parser.add_argument(
        "-F",
        "--flat",
        action="store_true",
        default=False,
        help="Keep images in the root of the target directory, and don't move them into subdirectories",
    )
    parser.add_argument(
        "-d",
        "--dry",
        action="store_true",
        help="Dry run don't save anything just show what's going to happen.",
    )

    args = parser.parse_args()
    folder_path = os.path.expanduser(args.folder)
    metadata = Metadata(os.path.join(folder_path, "image_metadata.json"))

    if os.path.exists(folder_path) == False:
        print(f"Folder '{folder_path}' does not exist!")
        exit(1)

    images_list = []

    for file in os.listdir(folder_path):
        if not file.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            continue

        images_list.append(file)

    with tqdm(total=len(images_list), desc="Processing images", unit="file") as pbar:
        for file_path in images_list:
            pbar.set_description(f"Processing {file_path[:12]}...")
            absolute_file_path = os.path.join(folder_path, file_path)

            image_obj = Image.open(absolute_file_path).convert("RGB")

            image_signature = blake3_file(absolute_file_path)
            size = f"{image_obj.size[0]}x{image_obj.size[1]}"

            metadata.put_path(image_signature, absolute_file_path)

            if metadata.data_has_signature(image_signature) == False:
                caption = generate_caption(image_obj)
                category = classify_category(caption)
                new_tags = caption + " " + category + " " + size
                metadata.put_metadata(
                    image_signature, size, caption, category, new_tags
                )

            pbar.update(1)

    if args.dry == False:
        metadata.save()

    print("Do you want to move images")

    if args.yes:
        confirm = "y"
    else:
        confirm = input("Confirm? (y/N): ")

    if confirm == "Y" or confirm == "y" and args.flat == False:
        for signature, curr_path in metadata.paths.items():
            if metadata.data.get(signature):
                caption = metadata.data[signature]["caption"]
                category = metadata.data[signature]["category"]
                new_path = os.path.join(
                    folder_path,
                    category,
                    generate_filename(signature, caption, curr_path),
                )
                print(f"'{curr_path}' => '{new_path}'")
                if args.dry == False:
                    os.makedirs(os.path.dirname(new_path), exist_ok=True)
                    shutil.move(curr_path, new_path)
