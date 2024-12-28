import os
import ffmpeg
import torch
import numpy as np
import librosa
from torchvision.io.video import read_video
from torchvision.models.video import r3d_18, R3D_18_Weights
from coco_mulla.utilities.encodec_utils import extract_rvq

device = 'cuda'
sr = 32000

def chunk_video(video_path, output_folder, segment_duration=100, fps=5, audio_rate=32000):
    """
    Chunk a video into segments of fixed duration and specified frame and audio rates.
    The function returns the paths of the created segments.
    """
    probe = ffmpeg.probe(video_path)
    video_duration = float(probe['format']['duration'])
    output_paths = []

    if video_duration <= segment_duration:
        # If the video is shorter than one segment, return empty since we only want segments
        return []

    os.makedirs(output_folder, exist_ok=True)
    total_segments = int(video_duration // segment_duration)
    
    for i in range(total_segments):
        start_time = i * segment_duration
        segment_folder_name = f"segment_{i+1:03d}"
        segment_folder_path = os.path.join(output_folder, segment_folder_name)
        os.makedirs(segment_folder_path, exist_ok=True)

        segment_path = os.path.join(segment_folder_path, "video.mp4")
        (
            ffmpeg
            .input(video_path, ss=start_time, t=segment_duration)
            .output(segment_path, vf=f"fps={fps}", ar=audio_rate)
            .run(overwrite_output=True)
        )
        output_paths.append(segment_path)

    return output_paths


def extract_video_embeddings(folder_path, model, preprocess):
    """
    Extract video embeddings from `video.mp4` in the given folder and save them as `video_emb.npy`.
    """
    video_path = os.path.join(folder_path, "video.mp4")
    if not os.path.isfile(video_path):
        return

    vid, _, info = read_video(video_path, output_format="TCHW")
    fps = info["video_fps"]
    total_frames = vid.shape[0]

    embeddings = []
    # Extract embeddings for each 1-second chunk of frames
    for i in range(0, total_frames, int(fps)):
        vid_cropped = vid[i:i+int(fps)]
        if vid_cropped.shape[0] == 0:
            continue
        batch = preprocess(vid_cropped).unsqueeze(0)
        with torch.no_grad():
            embedding = model(batch)
        embeddings.append(embedding)

    if embeddings:
        combined_embeddings = torch.cat(embeddings, dim=0)
        output_path = os.path.join(folder_path, "video_emb.npy")
        np.save(output_path, combined_embeddings.numpy())


def extract_music_embeddings(folder_path):
    """
    Extract music embeddings using `extract_rvq` from `video.mp4` and save as `music_emb.npy`.
    """
    video_path = os.path.join(folder_path, "video.mp4")
    if not os.path.isfile(video_path):
        return
    
    # Load audio
    wav, _ = librosa.load(video_path, sr=sr, mono=True)
    wav_torch = torch.from_numpy(wav).to(device)[None, None, ...]  # shape: (1,1,T)

    # Extract RVQ-based embeddings
    mix_rvq = extract_rvq(wav_torch, sr=sr)
    output_path = os.path.join(folder_path, "music_emb.npy")
    np.save(output_path, mix_rvq.cpu().numpy())

def process_directory(input_dir, output_dir, max_files=10, segment_duration=100, fps=5, audio_rate=32000):
    """
    Process up to `max_files` video files in the input directory:
    - Chunk them into segments.
    - Extract video and audio embeddings for each segment.
    - Remove the `video.mp4` files after embedding extraction, leaving only `video_emb.npy` and `music_emb.npy`.
    """
    # Load model and set it to embedding mode
    weights = R3D_18_Weights.DEFAULT
    model = r3d_18(weights=weights)
    model.eval()
    model.fc = torch.nn.Identity()  # remove classification head
    preprocess = weights.transforms()

    # Find up to max_files .mp4 files in input_dir
    all_files = [f for f in os.listdir(input_dir) if f.endswith(".mp4")]
    all_files = sorted(all_files)[:max_files]

    for file_name in all_files:
        input_path = os.path.join(input_dir, file_name)
        file_stem = os.path.splitext(file_name)[0]
        output_path = os.path.join(output_dir, file_stem)

        # Chunk the video
        print(f"Chunking {file_name}...")
        segment_paths = chunk_video(
            video_path=input_path,
            output_folder=output_path,
            segment_duration=segment_duration,
            fps=fps,
            audio_rate=audio_rate
        )

        # Extract embeddings for each segment and then remove the video file
        for segment_path in segment_paths:
            folder_path = os.path.dirname(segment_path)
            extract_video_embeddings(folder_path, model, preprocess)
            extract_music_embeddings(folder_path)

        print(f"Finished processing {file_name}.")


if __name__ == "__main__":
    input_directory = "/l/users/fathinah.izzati/coco-mulla-repo/demo/input/tnj/"
    output_directory = "/l/users/fathinah.izzati/coco-mulla-repo/demo/out/"

    process_directory(
        input_dir=input_directory,
        output_dir=output_directory,
        max_files=2,
        segment_duration=10,
        fps=5,
        audio_rate=32000
    )
