import argparse
import datetime
import glob
import os
import time
from multiprocessing import Pool

import av
import tqdm

def extract_frames_from_video(video_path, dataset_path):
    def extract_frames(video_path):
        video = av.open(video_path)
        for frame in video.decode(0):
            yield frame.to_image()

    # Extract the category name from the video file path
    base_name = os.path.basename(video_path)
    category_name = base_name.split('_')[1]
    sequence_name = os.path.splitext(base_name)[0]
    sequence_path = os.path.join(
        f"{dataset_path}-frames", category_name, sequence_name
    )

    if os.path.exists(sequence_path):
        return

    os.makedirs(sequence_path, exist_ok=True)

    # Extract frames
    for j, frame in enumerate(tqdm.tqdm(extract_frames(video_path), desc=f"{sequence_name}")):
        frame.save(os.path.join(sequence_path, f"{j}.jpg"))


def main(args):
    video_paths = glob.glob(os.path.join(args.dataset_path, "*.avi"))

    with Pool() as pool:
        pool.starmap(extract_frames_from_video, [(video, args.dataset_path) for video in video_paths])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="data/UCF101", help="Path to UCF-101 dataset")
    args = parser.parse_args()
    main(args)
