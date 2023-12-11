from moviepy.editor import VideoFileClip

def convert_video_to_gif(input_video, output_gif, fps=10):
    try:
        # Read the video
        video_clip = VideoFileClip(input_video)

        # Write GIF
        video_clip.write_gif(output_gif, fps=fps)

        print(f"Conversion successful. GIF saved to {output_gif}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert a video to a GIF.")
    parser.add_argument("input_video", help="Input video file path")
    parser.add_argument("output_gif", help="Output GIF file path")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for the GIF (default: 10)")

    args = parser.parse_args()

    convert_video_to_gif(args.input_video, args.output_gif, args.fps)
