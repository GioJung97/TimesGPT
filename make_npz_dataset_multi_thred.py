import os
import av
import glob
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoImageProcessor
from concurrent.futures import ProcessPoolExecutor, as_completed

# # *Assumption* Already have full videos
# # *Assumption* Handling the 8 frame case first, more frames are coming soon..
# # *Assumption* Handling clips of 10 seconds or less
# # *Assumption* We're only dealing with known english
# # *Suspicion* There are 1825 suspected non-english
# ##     Some were spanish, and some were phonetically how to spell something (e.g. S-P-E-L-L-I-N-G)
# # Input: YD CSV file
# # Input: Directory where the raw videos are
# # Output Json files with (youtube_id + time segment, pixel_values, and captions)
# 
# # GLOBAL time_encoding enabled/disabled
# # Open YD CSV file
# # Init models
# # Make a list of youtube_ids
# # Split list into train, test, val
# # For split in [train, test, val]
# ## Init new output list
# ## For youtube_id in split
# ### (1) Get values from CSV for youtube_id
# * yid = youtube_id,
# * start = audio_clip_start_time*1000,
# * end = audio_clip_end_time*1000,
# * dur = audio_clip_duration*1000 and
# * caption = audio_clip_transcript
# * english_flag = is_predicted_language_english
# ### if dur <= 10,000 AND english_flag
# ####   - calculate boundaries (start & end times)
# ####   - vid = f"{yid}_{start}_{end}"
# ####   - av.open(glob.glob(str(yid)+".*"))
# ####   - Count frames within boundaries
# ####   - if time_encoding -> calculate fps
# ####   - linespace to get frame indices
# ####   - Get frames using the frame indices
# ####   - if time_encoding enabled
# ####       - Calculate time encoding (TBD)
# ####   - Tokenize the captions
# ####   - Add videoID, pixel_values, labels, and time_encoding (if enabled) to dict
# ####   - append dict to output list
# 
# ## Save output list as json

# Open YD CSV file
SOURCE_VIDEO_PATH = "/data1/juve/datasets/youdescribe/videos/source"
# OUTPUT_CLIP_PATH = "/data1/juve/datasets/youdescribe/videos/clips/duration_leq_10"
# PROCESSED_DATA_OUTPUT_PATH = "/data2/juve/dataset/youdescribe/hf_datasets/arrow"
PROCESSED_DATA_OUTPUT_PATH = "/data2/juve/dataset/juve-optimization-2-multi-thread"
NUM_FRAMES = 8
BATCH_SIZE = 500
TIME_ENCODING = False
DATASET_NAME = "YD3_" + str(NUM_FRAMES) + "_frames"
csv_file = "/home/922053012/youdescribe-dataset/dataset/youdescribe_classic_dataset_cleaned_processed_videos_2024-10-26.csv" # 80505 datapoints
output_csv_file = PROCESSED_DATA_OUTPUT_PATH + "/" + DATASET_NAME + "/index.csv"
OPEN_VIDEOS = {}
NUM_WORKERS = 20
CLIP_LENGTH = 10
# rand_seed = 23

# Init models
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
tokenizer.pad_token = tokenizer.eos_token
image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")

def calculate_boundaries(start, end, audio_clip_duration, video_len):
    """
    Returns the start & end time corresponding to the clip,
    ensuring times are within the video boundaries.
    """
    # if start exceeds video_len for some reason
    if start > video_len:
        print("[WARNING] Start exceeds video_len")
        return None, None
    
    is_extended = (end - start) == 0.0
    
    if is_extended:
        # handle extended clips: generate a clip of length audio_clip_duration
        # centered around the start time
        half_duration = audio_clip_duration / 2.0
        new_start = start - half_duration
        new_end = start + half_duration

        # ensure new_start and new_end are within video boundaries
        if new_start < 0.0:
            # adjust new_end to maintain the desired duration
            new_start = 0.0
            new_end = new_start + audio_clip_duration
            if new_end > video_len:
                new_end = video_len
        elif new_end > video_len:
            # adjust new_start to maintain the desired duration
            new_end = video_len
            new_start = new_end - audio_clip_duration
            if new_start < 0.0:
                new_start = 0.0
    else:
        # inline clips: use the entire clip in the general case
        new_start = start
        new_end = end

        # ensure new_start and new_end are within video boundaries
        new_start = max(0.0, new_start)
        new_end = min(video_len, new_end)

    # ensure that start is less than end
    # print(f"[SECONDS] original_start: {start}\toriginal_end: {end}\nnew_start: {int(round(new_start))}\tnew_end: {int(round(new_end))}\nis_extended: {is_extended}\t audio_clip_duration: {audio_clip_duration}\t video_len: {video_len}")
    if new_start >= new_end:
        print(f"DEBUG: Invalid clip duration. Start: {new_start}, End: {new_end}")
        return None, None

    # round start and end times
    # new_start = int(round(new_start))
    # new_end = int(round(new_end))

    return new_start, new_end

def calculate_boundaries_old(start, end, video_len, clip_window_duration, extraction_type="balanced"):
    """
    Extracts clip segment
    returns the start & end time corresponding to our mechanism
    """
    middle = ((end - start) / 2.0) + start

    # case 0: video < 10 secs
    if video_len < 10.0:
        # if video_len < end:
        #     end = video_len
        # return 0.0, end, extraction_type
        return 0.0, video_len
    # case 1: the unused time goes into negative time of the video
    elif start - clip_window_duration < 0.0:
        if extraction_type.lower() == "before":
            start = 0.0
            end = start + clip_window_duration
        elif extraction_type.lower() == "balanced":
            if middle - (clip_window_duration / 2.0) < 0.0:
                start = 0.0
                end = start + clip_window_duration
            else:
                start = middle - (clip_window_duration / 2.0)
                end = start + clip_window_duration
        elif extraction_type.lower() == "after":
            start = middle
            end = start + clip_window_duration
        else:
            print(f"DEBUG: case 1 Something went wrong")
    # case 2: the unused time goes over the length of the video
    elif end + clip_window_duration > video_len:
        if extraction_type.lower() == "after":
            end = video_len
            start = end - clip_window_duration
        elif extraction_type.lower() == "balanced":
            if middle + (clip_window_duration / 2.0) > video_len:
                end = video_len
                start = end - clip_window_duration
            else:
                start = middle - (clip_window_duration / 2.0)
                end = start + clip_window_duration   
        elif extraction_type.lower() == "before":
            end = middle
            start = end - clip_window_duration
        else:
            print(f"DEBUG: case 2 Something went wrong")
    # case 3: balanced amount of time on each side of the clip
    else:
        if extraction_type.lower() == "after":
            if middle + (clip_window_duration / 2.0) > video_len:
                end = video_len
                start = end - clip_window_duration
            else:
                start = middle
                end = start + clip_window_duration
        elif extraction_type.lower() == "balanced":
            start = middle - (clip_window_duration / 2.0)
            end = start + clip_window_duration
        elif extraction_type.lower() == "before":
            if middle - (clip_window_duration / 2.0) < 0.0:
                start = 0
                end = start + clip_window_duration
            else:
                start = middle - clip_window_duration
                end = middle
        else:
            print(f"DEBUG: case 3 Something went wrong")
    # start = int(round(start))
    # end = int(round(end))
    
    assert end - start >= 10.0, f"calculate_boundaries() Assertion failed: end - start not >= 10\nend: {end}\tstart: {start}\nend - start: {end - start}"
    return start, end

def process_n_frames_from_source_video(video_path, start_time_ms, end_time_ms, num_frames, open_videos):
    # open video and store this information in
    # our open_videos dict
    if video_path not in open_videos:
        # print(f"[DEBUG] Recording open video status for {video_path}")
        open_videos[video_path] = av.open(video_path)
    container = open_videos[video_path]

    # container = av.open(video_path)
    video_stream = container.streams.video[0]

    # retrieve fps if possible
    fps = float(video_stream.average_rate) if video_stream.average_rate is not None else None
    # if video_stream.average_rate is not None:
    #     fps = float(video_stream.average_rate)
    # else:
    #     fps = None

    # convert ms to seconds
    start_time_s = start_time_ms / 1000.0
    end_time_s = end_time_ms / 1000.0

    # convert seconds to PTS using time_base
    time_base = video_stream.time_base
    start_pts = int(start_time_s / time_base) if time_base else None
    end_pts = int(end_time_s / time_base) if time_base else None
    # print(f"time_base: {time_base}\tstart_pts: {start_pts}\tend_pts{end_pts}")
    
    # seek to the closest frame before `start_pts`
    container.seek(start_pts, any_frame=False, backward=True, stream=video_stream)

    # collect frames in the interval [start_pts, end_pts]
    frames_in_interval = []
    for packet in container.demux(video_stream):
        for frame in packet.decode():
            # if frame passes all these checks...
            if frame.pts is None:
                # print(f"frame.pts is None => frame.pts: {frame.pts}")
                continue
            if frame.pts < start_pts:
                # print(f"frame.pts < start_pts => frame.pts: {frame.pts}, start_pts: {start_pts}")
                continue
            if frame.pts > end_pts:
                # print(f"frame.pts > end_pts => frame.pts: {frame.pts}, end_pts: {end_pts}")
                break
            
            # append good frames
            rgb_frame = frame.to_ndarray(format="rgb24")
            # pil_img = Image.fromarray(rgb_frame).resize((224, 224), Image.BILINEAR)
            # rgb_frame = np.array(pil_img)
            frames_in_interval.append(rgb_frame)
        # stop decoding once exceeded end_pts
        if frames_in_interval and frame.pts is not None and frame.pts > end_pts:
            # print(f"frames_in_interval: {frames_in_interval}\tframe.pts: {frame.pts}\tend_pts: {end_pts}")
            break

    # if no frames were found in the interval
    if not frames_in_interval:
        print(f"No frames found between {start_time_ms}ms and {end_time_ms}ms in {video_path}.")
        return None, fps

    # evenly sample `num_frames` from frames_in_interval
    frame_count = len(frames_in_interval)
    indices = np.linspace(0, frame_count - 1, num=num_frames, endpoint=True).astype(int)
    selected_frames = [frames_in_interval[i] for i in indices]

    processed_frames = image_processor(selected_frames).pixel_values[0]
    # print(f"fps: {fps}")
    # container.close()

    return processed_frames, fps

def process_n_captions(captions):
    return tokenizer(captions, padding="max_length", truncation=True, return_tensors="pt").input_ids

def process_chunk(df_chunk, source_video_path, output_dir, dataset_name, num_frames, partial_csv_name):
    """
    Each chunk is a subset of the main dataframe with unique youtube_ids. We iterate over each row in this chunk,
    open/process frames and save output .npz.
    """
    # each process has its own open_videos cache
    open_videos = {}
    videoIDS_index = []

    os.makedirs(os.path.join(output_dir, dataset_name), exist_ok=True)

    for i in range(len(df_chunk)):
        print(f"Iteration: {i} / {len(df_chunk)}")
        row = df_chunk.iloc[i]
        yid              = row["youtube_id"]
        audio_clip_start = float(row["audio_clip_start_time"]) * 1000
        audio_clip_end   = float(row["audio_clip_end_time"])   * 1000
        audio_clip_dur   = float(row["audio_clip_duration"])    * 1000
        video_dur        = float(row["video_duration"])         * 1000
        captions         = row["audio_clip_transcript"]
        audio_clip_id = row["audio_clip_id"]

        new_start, new_end = calculate_boundaries(audio_clip_start, audio_clip_end, audio_clip_dur, video_dur)
        if new_start is None or new_end is None:
            continue
        
        youtube_id_video_path = os.path.join(SOURCE_VIDEO_PATH, yid)
        source_video_path = glob.glob(str(youtube_id_video_path) + ".*")

        # skip, if we didn't find a file for this youtube_id
        if not source_video_path:
            print(f"ERROR: Could not find video on disk: {yid}")
            continue

        # skip the .mhtml files        
        if source_video_path[0][-6:] == ".mhtml":
            print(f"[WARNING] {source_video_path[0][-6:]} file format does not work..")
            continue

        frames, fps = process_n_frames_from_source_video(
            source_video_path[0],
            new_start, new_end,
            num_frames,
            open_videos
        )
        if frames is None or fps is None:
            print(f"[WARNING] Skipping clip: {yid} => no frames or fps..")
            continue

        processed_captions = process_n_captions(captions)

        new_start = int(new_start)
        new_end   = int(new_end)
        vid_id  = f"{yid}_{new_start}_{new_end}_{audio_clip_id}"
        output_npz_filename = os.path.join(output_dir, dataset_name, vid_id + ".npz")

        np.savez(output_npz_filename, frames, processed_captions)
        videoIDS_index.append(vid_id)

        if i == len(df_chunk)-1:
            # close video that we opened if we are at the last datapoint
            if source_video_path[0] in open_videos:
                # print(f"[DEBUG] Closing video: {source_video_path[0]}")
                open_videos[source_video_path[0]].close()
                del open_videos[source_video_path[0]]
        else:
            # close the video if we dont have consecutive datapoints with the same youtube_id
            if df_chunk.iloc[i + 1]["youtube_id"] != yid:
                if source_video_path[0] in open_videos:
                    # print(f"[DEBUG] Closing video: {source_video_path[0]}")
                    open_videos[source_video_path[0]].close()
                    del open_videos[source_video_path[0]]
    # close all open videos in this process
    for path, container in open_videos.items():
        container.close()

    # write partial CSV
    with open(partial_csv_name, 'w') as f:
        for vid_id in videoIDS_index:
            f.write(vid_id + "\n")

    return videoIDS_index

########################################################
## MAIN
########################################################
# def main():
    # YD_csv_file = pd.read_csv(csv_file)

    # # Filter data and take a list of unique youtube_ids to split the data on.
    # df_leq_ten_second_clips = YD_csv_file[YD_csv_file["audio_clip_duration"] <= 10].reset_index(drop=True)
    # df_only_english = df_leq_ten_second_clips[df_leq_ten_second_clips["is_predicted_language_english"] == True].reset_index(drop=True)
    # youtube_ids = list(df_only_english["youtube_id"].unique())
    # df = YD_csv_file[YD_csv_file["youtube_id"].isin(youtube_ids)].reset_index()
    
    # # chunking dataset by youtube_ids
    # # np.array_split: splits an array into NUM_WORKERS sub arrays.
    # chunked_youtube_ids = np.array_split(youtube_ids[:4], NUM_WORKERS)

    # # for each chunk of youtube_ids, create a chunk DataFrame
    # chunks = []
    # for id_list in chunked_youtube_ids:
    #     df_chunk = df[df["youtube_id"].isin(id_list)]
    #     chunks.append(df_chunk)

    # os.makedirs(PROCESSED_DATA_OUTPUT_PATH, exist_ok=True)
    # os.makedirs(os.path.join(PROCESSED_DATA_OUTPUT_PATH, DATASET_NAME), exist_ok=True)
    
    # partial_csv_files = []
    # workers = []
    # with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
    #     for i, chunk_df in enumerate(chunks):
    #         partial_csv = os.path.join(PROCESSED_DATA_OUTPUT_PATH, f"partial_{i}.csv")
    #         partial_csv_files.append(partial_csv)

    #         job = executor.submit(
    #             process_chunk,
    #             chunk_df,
    #             SOURCE_VIDEO_PATH,
    #             PROCESSED_DATA_OUTPUT_PATH,
    #             DATASET_NAME,
    #             NUM_FRAMES,
    #             partial_csv
    #         )
    #         workers.append(job)

    # # gather results
    # all_video_ids = []
    # # as_completed(): yields workers as they finish execution
    # # regardless of the order in which they were submitted.
    # for fut in as_completed(workers):
    #     result_videoIDS_index = fut.result()  # list of videoIDS_index processed by that worker
    #     all_video_ids.extend(result_videoIDS_index)

    # # combine partial CSVs into one final CSV
    # with open(output_csv_file, 'w') as out_f:
    #     for partial_file in partial_csv_files:
    #         if os.path.exists(partial_file):
    #             with open(partial_file, 'r') as pcsv:
    #                 for line in pcsv:
    #                     out_f.write(line)
    #             # remove partial CSV
    #             os.remove(partial_file)

if __name__ == "__main__":
    YD_csv_file = pd.read_csv(csv_file)

    # Filter data and take a list of unique youtube_ids to split the data on.
    df_leq_ten_second_clips = YD_csv_file[YD_csv_file["audio_clip_duration"] <= CLIP_LENGTH].reset_index(drop=True)
    df_only_english = df_leq_ten_second_clips[df_leq_ten_second_clips["is_predicted_language_english"] == True].reset_index(drop=True)
    youtube_ids = list(df_only_english["youtube_id"].unique())
    df = YD_csv_file[YD_csv_file["youtube_id"].isin(youtube_ids)].reset_index()
    
    # chunking dataset by youtube_ids
    # np.array_split: splits an array into NUM_WORKERS sub arrays.
    chunked_youtube_ids = np.array_split(youtube_ids[:100], NUM_WORKERS)

    # for each chunk of youtube_ids, create a chunk DataFrame
    chunks = []
    for id_list in chunked_youtube_ids:
        df_chunk = df[df["youtube_id"].isin(id_list)]
        chunks.append(df_chunk)

    os.makedirs(PROCESSED_DATA_OUTPUT_PATH, exist_ok=True)
    os.makedirs(os.path.join(PROCESSED_DATA_OUTPUT_PATH, DATASET_NAME), exist_ok=True)
    
    partial_csv_files = []
    workers = []
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for i, chunk_df in enumerate(chunks):
            partial_csv = os.path.join(PROCESSED_DATA_OUTPUT_PATH, f"partial_{i}.csv")
            partial_csv_files.append(partial_csv)

            job = executor.submit(
                process_chunk,
                chunk_df,
                SOURCE_VIDEO_PATH,
                PROCESSED_DATA_OUTPUT_PATH,
                DATASET_NAME,
                NUM_FRAMES,
                partial_csv
            )
            workers.append(job)

    # gather results
    all_video_ids = []
    # as_completed(): yields workers as they finish execution
    # regardless of the order in which they were submitted.
    for fut in as_completed(workers):
        result_videoIDS_index = fut.result()  # list of videoIDS_index processed by that worker
        all_video_ids.extend(result_videoIDS_index)

    # combine partial CSVs into one final CSV
    with open(output_csv_file, 'w') as out_f:
        for partial_file in partial_csv_files:
            if os.path.exists(partial_file):
                with open(partial_file, 'r') as pcsv:
                    for line in pcsv:
                        out_f.write(line)
                # remove partial CSV
                os.remove(partial_file)