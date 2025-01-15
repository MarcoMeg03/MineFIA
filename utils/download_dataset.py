# A script to download OpenAI contractor data or BASALT datasets
# using the shared .json files (index file).
#
# Json files are in format:
# {"basedir": <prefix>, "relpaths": [<relpath>, ...]}
#
# The script will download only the MP4 files for the demonstrations.
#

import argparse
import urllib.request
import os
import json

parser = argparse.ArgumentParser(description="Download MP4 files from OpenAI contractor datasets")
parser.add_argument("--json-file", type=str, required=True, help="Path to the index .json file")
parser.add_argument("--output-dir", type=str, required=True, help="Path to the output directory")
parser.add_argument("--num-demos", type=int, default=None, help="Maximum number of demonstrations to download")

def main(args):
    # Legge il file indice (JSON)
    with open(args.json_file, "r") as f:
        data = json.load(f)

    basedir = data["basedir"]
    relpaths = data["relpaths"]
    
    # Limita il numero di dimostrazioni da scaricare
    if args.num_demos is not None:
        relpaths = relpaths[:args.num_demos]

    # Crea la directory di output se non esiste
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Per ogni dimostrazione, scarica solo il file MP4
    for i, relpath in enumerate(relpaths):
        video_url = basedir + relpath
        video_filename = os.path.basename(relpath)
        video_outpath = os.path.join(args.output_dir, video_filename)

        # Controlla se il file esiste già
        if os.path.exists(video_outpath):
            print(f"[{100 * i / len(relpaths):.0f}%] {video_filename} già presente. Skipping.")
            continue

        percent_done = 100 * i / len(relpaths)
        print(f"[{percent_done:.0f}%] Downloading {video_filename}")
        try:
            urllib.request.urlretrieve(video_url, video_outpath)
        except Exception as e:
            print(f"\tError downloading {video_url}: {e}")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
