"""
Video Caption Extractor
Extracts captions from videos using Whisper model and saves them in multiple formats.

Requirements:
pip install whisper openai-whisper moviepy
"""

import os
import json
from pathlib import Path
import whisper
from moviepy.editor import VideoFileClip

class VideoCaptionExtractor:
    def __init__(self, model_size="base"):
        """
        Initialize the caption extractor.
        
        Args:
            model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
                       'base' is recommended for balance of speed and accuracy
        """
        print(f"Loading Whisper model ({model_size})...")
        self.model = whisper.load_model(model_size)
        print("Model loaded successfully!")
        
    def extract_audio(self, video_path, audio_path):
        """Extract audio from video file."""
        try:
            video = VideoFileClip(str(video_path))
            video.audio.write_audiofile(str(audio_path), verbose=False, logger=None)
            video.close()
            return True
        except Exception as e:
            print(f"Error extracting audio from {video_path}: {e}")
            return False
    
    def transcribe_video(self, video_path):
        """Transcribe video to text with timestamps."""
        temp_audio = "temp_audio.mp3"
        
        try:
            # Extract audio from video
            if not self.extract_audio(video_path, temp_audio):
                return None
            
            # Transcribe audio
            print(f"Transcribing {video_path.name}...")
            result = self.model.transcribe(temp_audio)
            
            # Clean up temp file
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
            
            return result
            
        except Exception as e:
            print(f"Error transcribing {video_path}: {e}")
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
            return None
    
    def save_as_txt(self, result, output_path):
        """Save caption as plain text."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result['text'])
    
    def save_as_srt(self, result, output_path):
        """Save caption as SRT subtitle format."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(result['segments'], 1):
                start = self._format_timestamp(segment['start'])
                end = self._format_timestamp(segment['end'])
                text = segment['text'].strip()
                
                f.write(f"{i}\n")
                f.write(f"{start} --> {end}\n")
                f.write(f"{text}\n\n")
    
    def save_as_json(self, result, output_path):
        """Save caption with full details as JSON."""
        output_data = {
            'text': result['text'],
            'language': result['language'],
            'segments': [
                {
                    'id': seg['id'],
                    'start': seg['start'],
                    'end': seg['end'],
                    'text': seg['text'].strip()
                }
                for seg in result['segments']
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    def _format_timestamp(self, seconds):
        """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    def process_folder(self, input_folder, output_folder, formats=['txt', 'srt', 'json']):
        """
        Process all videos in a folder and extract captions.
        
        Args:
            input_folder: Path to folder containing videos
            output_folder: Path to folder for saving captions
            formats: List of output formats ('txt', 'srt', 'json')
        """
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        
        # Create output folder if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Common video extensions
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
        
        # Find all video files
        video_files = [
            f for f in input_path.iterdir() 
            if f.is_file() and f.suffix.lower() in video_extensions
        ]
        
        if not video_files:
            print(f"No video files found in {input_folder}")
            return
        
        print(f"Found {len(video_files)} video(s) to process\n")
        
        # Process each video
        for i, video_file in enumerate(video_files, 1):
            print(f"[{i}/{len(video_files)}] Processing: {video_file.name}")
            
            result = self.transcribe_video(video_file)
            
            if result:
                base_name = video_file.stem
                
                # Save in requested formats
                if 'txt' in formats:
                    txt_path = output_path / f"{base_name}.txt"
                    self.save_as_txt(result, txt_path)
                    print(f"  ✓ Saved: {txt_path.name}")
                
                if 'srt' in formats:
                    srt_path = output_path / f"{base_name}.srt"
                    self.save_as_srt(result, srt_path)
                    print(f"  ✓ Saved: {srt_path.name}")
                
                if 'json' in formats:
                    json_path = output_path / f"{base_name}.json"
                    self.save_as_json(result, json_path)
                    print(f"  ✓ Saved: {json_path.name}")
                
                print(f"  Language detected: {result['language']}\n")
            else:
                print(f"  ✗ Failed to process {video_file.name}\n")
        
        print("Processing complete!")


def main():
    # Configuration
    INPUT_FOLDER = "videos"          # Folder containing your videos
    OUTPUT_FOLDER = "captions"       # Folder to save captions
    MODEL_SIZE = "base"              # Options: tiny, base, small, medium, large
    FORMATS = ['txt', 'srt', 'json'] # Output formats
    
    # Create extractor and process videos
    extractor = VideoCaptionExtractor(model_size=MODEL_SIZE)
    extractor.process_folder(INPUT_FOLDER, OUTPUT_FOLDER, formats=FORMATS)


if __name__ == "__main__":
    main()