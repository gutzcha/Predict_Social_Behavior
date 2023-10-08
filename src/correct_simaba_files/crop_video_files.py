import os
import subprocess
import pandas as pd
from const import action_map
from tqdm import tqdm

def crop_and_save_videos_ffmpeg(file_name, timestamp_df):
    # Create a directory to save cropped videos if it doesn't exist
    output_dir = "cropped_videos_ffmpeg"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for index, row in tqdm(timestamp_df.iterrows(), total=len(timestamp_df), desc="Processing"):
        time_start = row['start_time']
        time_end = row['end_time']
        action = row['action']

        output_subfolder = os.path.join(output_dir, action)
        if not os.path.exists(output_subfolder):
            os.makedirs(output_subfolder)

        output_file = os.path.join(output_subfolder, f"{action}_{time_start}_{time_end}.avi")

        # Using FFmpeg to crop and save the video
        ffmpeg_command = [
            'ffmpeg',
            '-i', file_name,
            '-ss', str(time_start),
            '-to', str(time_end),
            '-c:v', 'copy',
            '-an',  # Remove audio
            output_file
        ]
        subprocess.run(ffmpeg_command)

    print("Cropping and saving completed!")


def extract_time_stamps_from_excel_file(excel_path):
    df = pd.read_excel(excel_path)
    df = df[['start_time','end_time', 'action']]
    df['action'] = df['action'].apply(lambda x: action_map[x])
    return df




# Example usage
if __name__ == "__main__":
    # data = {
    #     "time_start": [5, 15, 25],
    #     "time_end": [10, 20, 30],
    #     "action": ["action1", "action2", "action3"]
    # }
    # df = pd.DataFrame(data)
    #
    # video_file = "input_video.mp4"
    # crop_and_save_videos_ffmpeg(video_file, df)
    excel_path = 'cropped_Rat8-probe8-Day2-Free-sniffing_2020-01-13-133802-0000_annot.xlsx'
    df = extract_time_stamps_from_excel_file(excel_path)
    video_file = excel_path.replace('_annot.xlsx','.avi')
    crop_and_save_videos_ffmpeg(video_file, df)