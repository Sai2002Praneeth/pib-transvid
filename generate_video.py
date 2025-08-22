import math
import os
import moviepy.editor as mp
import logging
import numpy
from PIL import Image

if not hasattr(Image, "Resampling"):
    Image.Resampling = type("Resampling", (), {
        "NEAREST" : Image.NEAREST,
        "BOX" : Image.BOX,
        "BILINEAR" : Image.BILINEAR,
        "HAMMING" : Image.HAMMING,
        "BICUBIC" : Image.BICUBIC,
        "LANCZOS" : Image.LANCZOS
    })

if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = Image.Resampling.LANCZOS

def zoom_in_effect(clip, zoom_ratio=0.04):
    def effect(get_frame, t):
        img = Image.fromarray(get_frame(t))
        base_size = img.size

        new_size = [
            math.ceil(img.size[0] * (1 + (zoom_ratio * t))),
            math.ceil(img.size[1] * (1 + (zoom_ratio * t)))
        ]

        # The new dimensions must be even.
        new_size[0] = new_size[0] + (new_size[0] % 2)
        new_size[1] = new_size[1] + (new_size[1] % 2)

        resample = Image.Resampling.LANCZOS

        img = img.resize(new_size, resample)

        x = math.ceil((new_size[0] - base_size[0]) / 2)
        y = math.ceil((new_size[1] - base_size[1]) / 2)

        img = img.crop([
            x, y, new_size[0] - x, new_size[1] - y
        ]).resize(base_size, resample)

        result = numpy.array(img)
        img.close()

        return result

    return clip.fl(effect)

size = (1920, 1080)

def edit_video_with_voiceover(voiceover_path, scrapped_images_folder, static_images_folder):
    # Set up logging
    logging.basicConfig(filename='video_edit_log.txt', level=logging.INFO)

    try:
        # Find voice_over.mp3 in the assets folder
        output_path = 'assets/generated_video.mp4'
        if os.path.exists(output_path):
            return output_path

        # Find Scrapped_Images folder and count total images
        if os.path.exists(scrapped_images_folder) and os.path.isdir(scrapped_images_folder):
            image_files_scrapped = [f for f in os.listdir(scrapped_images_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.gif'))]
            image_folder_scrapped = scrapped_images_folder
        else:
            image_files_scrapped = []

        # Check if there are any images in the static folder
        image_files_static = [f for f in os.listdir(static_images_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.gif'))]
        image_folder_static = static_images_folder

        # Combine image files from both folders
        all_image_files = image_files_scrapped + image_files_static

        if not all_image_files:
            print(f"No images found in both Scrapped_Images and static folders.")
            raise ValueError("No images found in both Scrapped_Images and static folders.")

        total_images = len(all_image_files)

        N = 3
        # Calculate time for each image
        voiceover_clip = mp.AudioFileClip(voiceover_path)
        time_per_image = voiceover_clip.duration / total_images
        time_per_image = time_per_image / N
        # Create clips for each image
        video_clips = []
        for i in range(1, N + 1):
            for image_file in all_image_files:
                image_path = os.path.join(image_folder_scrapped, image_file) if image_file in image_files_scrapped else os.path.join(image_folder_static, image_file)
                image_clip = mp.ImageClip(image_path, duration=time_per_image).resize(size)

                # adding zoom
                image_clip = zoom_in_effect(image_clip, zoom_ratio=0.04)

                video_clips.append(image_clip)

        # Concatenate the clips into one video
        final_video = mp.concatenate_videoclips(video_clips, method="compose")

        # Set the audio to the voiceover
        final_video = final_video.set_audio(voiceover_clip)

        # Save the final video
        final_video.write_videofile(output_path, codec='libx264', fps=24)

        logging.info("Video editing completed successfully.")
        return output_path

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        print(f"Error: {str(e)}")
        return None

if __name__ == "__main__":
    voiceover_path = 'assets/output.mp3'
    scrapped_images_folder = 'Scrapped_Images'
    static_images_folder = 'static'
    edit_video_with_voiceover(voiceover_path=voiceover_path, scrapped_images_folder=scrapped_images_folder, static_images_folder=static_images_folder)
