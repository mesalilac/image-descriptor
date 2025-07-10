# Usage

Have all the wallpaper images in the root of the target directory.
run the script and it will sort the files into folder categories

main.py [-h] [--folder FOLDER] [-y] [-f] [-d]

| Alias | Full     | Description                                                                              |
| ----- | -------- | ---------------------------------------------------------------------------------------- |
| -h    | --help   | show this help message and exit                                                          |
| -f    | --folder | FOLDER Wallpapers folder                                                                 |
| -y    | --yes    | Skip confirmation prompt for reorganizing images.                                        |
| -F    | --flat   | Keep images in the root of the target directory, and don't move them into subdirectories |
| -d    | --dry    | Dry run don't save anything just show what's going to happen.                            |

# Example

```
- Pictures/
    | - wallpapers/
        | - 1.jpg
        | - 2.jpg
        | - 3.jpg
        | - 4.jpg
        | - 5.jpg
```

```python
python main.py -f ~/Pictures/wallpapers
```

## Results

```
- Pictures/
    | - wallpapers/
        | - image_metadata.json
        | - <Category1>
            | - <new file name>.jpg
            | - <new file name>.jpg
            | - <new file name>.jpg
        | - <Category2>
            | - <new file name>.jpg
            | - <new file name>.jpg
            | - <new file name>.jpg
```

## image_metadata.json

```json
[
    {
        "signature": "blake3 hash",
        "size": "WIDTHxHEIGHT",
        "caption": "image description",
        "category": "image category",
        "tags": "<caption> + <category> + <size>"
    }
]
```
