import os
import ffmpeg
import torch
import numpy as np
from torchvision.io.video import read_video
from torchvision.models.video import r3d_18, R3D_18_Weights

def chunk_video(video_path, output_folder, segment_duration=10, fps=50, audio_rate=32000):
    """
    Chunk a video into segments of fixed duration and specified frame and audio rates.
    
    Parameters
    ----------
    video_path : str
        The path to the input video.
    output_folder : str
        The base folder where segmented videos will be saved.
    segment_duration : int or float
        Duration of each segment in seconds.
    fps : int
        The output frames per second for the video segments.
    audio_rate : int
        The output audio sampling rate (Hz).

    Returns
    -------
    output_paths : list of str
        A list of paths to the created video segments.
    """
    # Get video duration in seconds
    probe = ffmpeg.probe(video_path)
    video_duration = float(probe['format']['duration'])
    output_paths = []
    
    if video_duration <= segment_duration:
        # If the video is shorter than one segment, just return the original video path
        return [video_path]

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Calculate how many full segments fit into the video
    total_segments = int(video_duration // segment_duration)
    for i in range(total_segments):
        start_time = i * segment_duration
        # Create a folder for each segment
        segment_folder_name = f"segment_{i+1:03d}"
        segment_folder_path = os.path.join(output_folder, segment_folder_name)
        os.makedirs(segment_folder_path, exist_ok=True)

        # Define the path for the segment's video
        segment_path = os.path.join(segment_folder_path, "video.mp4")

        # Use ffmpeg to segment the video
        (
            ffmpeg
            .input(video_path, ss=start_time, t=segment_duration)
            .output(segment_path, vf=f"fps={fps}", ar=audio_rate)
            .run(overwrite_output=True)
        )
        output_paths.append(segment_path)

    return output_paths


def extract_embeddings_from_folder(folder_path, model, preprocess):
    """
    Extract embeddings from a folder containing a segmented video ("video.mp4").

    Parameters
    ----------
    folder_path : str
        Path to the folder containing "video.mp4".
    model : torch.nn.Module
        The R3D model for embedding extraction.
    preprocess : callable
        Preprocessing function from the model weights.

    Returns
    -------
    None
    """
    video_path = os.path.join(folder_path, "video.mp4")
    if not os.path.isfile(video_path):
        return

    # Read the video (TCHW format)
    vid, _, info = read_video(video_path, output_format="TCHW")
    fps = info["video_fps"]
    total_frames = vid.shape[0]

    embeddings = []
    # Extract embeddings for each 1-second segment (assuming fps frames represent ~1 second)
    # If you want per-frame embeddings, you can adjust the indexing below.
    for i in range(0, total_frames, int(fps)):
        vid_cropped = vid[i:i+int(fps)]
        if vid_cropped.shape[0] == 0:
            continue

        batch = preprocess(vid_cropped).unsqueeze(0)
        with torch.no_grad():
            embedding = model(batch)
        embeddings.append(embedding)

    # Combine embeddings into a single tensor and save
    if embeddings:
        combined_embeddings = torch.cat(embeddings, dim=0)
        output_path = os.path.join(folder_path, "video_emb.npy")
        np.save(output_path, combined_embeddings.numpy())


def process_directory(input_dir, output_dir, max_files=10, segment_duration=10, fps=50, audio_rate=32000):
    """
    Process up to `max_files` video files in the input directory, chunk them, and extract embeddings.
    
    Parameters
    ----------
    input_dir : str
        Directory containing the input video files.
    output_dir : str
        Base directory to store the chunked videos and embeddings.
    max_files : int
        Maximum number of video files to process.
    segment_duration : int or float
        Duration of each segment in seconds.
    fps : int
        Frames per second for output video segments.
    audio_rate : int
        Audio sampling rate (Hz).
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

        # Extract embeddings for each segment
        for segment_path in segment_paths:
            folder_path = os.path.dirname(segment_path)
            extract_embeddings_from_folder(folder_path, model, preprocess)
        print(f"Finished processing {file_name}.")


if __name__ == "__main__":
    # Example usage:
    input_directory = "/l/users/fathinah.izzati/coco-mulla-repo/demo/input/tnj/"
    output_directory = "/l/users/fathinah.izzati/coco-mulla-repo/demo/out/"
    
    process_directory(input_dir=input_directory, output_dir=output_directory, max_files=1)
