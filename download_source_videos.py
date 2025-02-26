# EXAMPLE USE:
# time python3 download_source_videos.py 
#   --input_file=/home/918573232/vd_aug_gpt/vd_aug/youdescribe_classic_dataset_cleaned_processed_videos_with_clip_ids_old_mech_2024-10-26.csv
#   --output_dir=/data1/juve/datasets/youdescribe/videos/source
#   -q

import yt_dlp
import csv
import os
import argparse
import glob

parser = argparse.ArgumentParser(description="Download YouTube videos listed in a CSV file.")
parser.add_argument('-i', '--input_file', type=str, required=True, help="Path to the CSV file containing a 'youtube_id' column.")
parser.add_argument('-o', '--output_dir', type=str, required=True, help="Directory where the downloaded videos will be stored.")
parser.add_argument('-q', '--quiet', action="store_true", help="Reduce output from the download process.")
args = parser.parse_args()

YOUTUBE_LINK = "https://www.youtube.com/watch?v="

os.makedirs(args.output_dir, exist_ok=True)

with open(args.input_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        youtube_id = row["youtube_id"].strip()
        if not youtube_id:
            continue

        output_file_name = os.path.join(args.output_dir, youtube_id)

        # check if the video was already downloaded (any extension)
        if glob.glob(output_file_name + ".*"):
            print(f"Video {youtube_id} already exists, skipping.")
            continue

        youtube_url = YOUTUBE_LINK + youtube_id

        ydl_opts = {
            'format': 'bv*+ba/b',
            'outtmpl': output_file_name + ".%(ext)s",
            'quiet': args.quiet
        }

        print(f"Downloading video: {youtube_id}")
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([youtube_url])
        except Exception as e:
            print(f"Error downloading video {youtube_id}: {e}")
