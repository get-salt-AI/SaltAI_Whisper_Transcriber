import json
import os
import tempfile

import numpy as np
from pydub import AudioSegment
from moviepy.editor import VideoFileClip
import openai
from PIL import Image
import torch

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

import folder_paths

OUTPUT = folder_paths.get_output_directory()
INPUT = folder_paths.get_input_directory()
TEMP = folder_paths.get_temp_directory()
MODELS = folder_paths.models_dir

WHISPER_MODELS = os.path.join(MODELS, "whisper")


def pil2tensor(x):
    return torch.from_numpy(np.array(x).astype(np.float32) / 255.0).unsqueeze(0)


class SAIWhisperLoadModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (["openai/whisper-large-v2", "openai/whisper-large-v3", "openai/whisper-base", "openai/whisper-large", "openai/whisper-medium", "openai/whisper-small", "openai/whisper-tiny", "distil-whisper/distil-large-v3", ], ),
            },
            "optional": {
                "device": (["cuda", "cpu"], ),
            },
        }
    
    RETURN_TYPES = ("WHISPER_MODEL",)
    RETURN_NAMES = ("model", "processor")

    FUNCTION = "load_model"
    CATEGORY = "SALT/Whisper"

    def load_model(self, model, device="cuda"):
        whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(model, cache_dir=WHISPER_MODELS, use_safetensors=True).to(device)
        processor = AutoProcessor.from_pretrained(model)
        return ((whisper_model, processor, device), )
    

class SAIWhisperTranscribe:
    def __init__(self):
        self.video_extensions = [
            ".3g2", ".3gp", ".3gp2", ".3gpp", ".amv", ".asf", ".avi", ".divx",
            ".drc", ".dv", ".f4v", ".flv", ".m2v", ".m4p", ".m4v", ".mkv",
            ".mov", ".mp4", ".mpe", ".mpeg", ".mpeg2", ".mpeg4", ".mpg",
            ".mpv", ".mxf", ".nsv", ".ogg", ".ogv", ".qt", ".rm", ".rmvb",
            ".roq", ".svi", ".vob", ".webm", ".wmv", ".yuv"
        ]
        self.audio_extensions = [
            ".mp3", ".wav", ".aac", ".flac", ".ogg", ".m4a", ".wma"
        ]   
                    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "whisper_model": ("WHISPER_MODEL",),
                "file_path": ("STRING", {})
            },
            "optional": {
                "frame_rate": ("FLOAT", {"default": 8, "min": 1, "max": 244}),
                "chunk_type": (["sentence", "word"],),
                "max_new_tokens": ("INT", {"min": 1, "max": 4096, "default": 128}),
            },
        }

    RETURN_TYPES = ("STRING", "DICT", "DICT", "STRING", "IMAGE", "INT", "INT", "INT")
    RETURN_NAMES = ("transcription_text", "transcription_timestamp_dict", "transcription_frame_dict", "prompt_schedule", "images", "transcription_count", "frame_rate", "frame_count")

    FUNCTION = "transcribe"
    CATEGORY = "SALT/Whisper"

    def transcribe(self, whisper_model, file_path, **kwargs):
        model, processor, device = whisper_model

        media_type = self.validate(file_path)
        if not media_type:
            supported_formats = ', '.join(self.video_extensions + self.audio_extensions)
            raise ValueError(f"Unsupported media file format. Please provide a valid video or audio file: {supported_formats}")
        
        path = os.path.join(INPUT, file_path)
        audio_path, derived_fps, frame_count, duration = self.extract_audio(path, kwargs.get('frame_rate', 8))

        raw_text, transcription, transcription_frame, prompt_schedule, images = self.transcribe_audio(
            audio_path, 
            path, 
            derived_fps,
            duration,
            model, 
            processor, 
            kwargs.get("max_new_tokens", 128),
            media_type,
            kwargs.get("chunk_type", "sentence")
        )

        transcription_count = len(transcription_frame)

        return raw_text, transcription, transcription_frame, prompt_schedule, images, transcription_count, derived_fps, frame_count

    def transcribe_audio(self, audio_path, file_path, fps, duration, model, processor, max_new_tokens, media_type="audio", chunk_type="sentence"):
        audio = AudioSegment.from_file(audio_path).set_frame_rate(16000).set_channels(1)
        samples = np.array(audio.get_array_of_samples())
        if audio.sample_width == 2:
            samples = samples.astype(np.float32) / 2**15
        elif audio.sample_width == 4:
            samples = samples.astype(np.float32) / 2**31

        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            feature_extractor=processor.feature_extractor,
            tokenizer=processor.tokenizer,
            return_timestamps=chunk_type,
            max_new_tokens=max_new_tokens,
        )
        result = pipe(samples)
        
        raw_text = result['text'].strip()
        transcription = {}
        transcription_frame = {}
        images = []
        prompt_schedule = ""

        last_end_time = 0
        segment_offset = 0

        for chunk in result['chunks']:
            text = chunk['text']
            start_time, end_time = chunk['timestamp']

            if start_time < last_end_time:
                segment_offset += last_end_time

            adjusted_start_time = start_time + segment_offset
            frame_number = int(adjusted_start_time * fps)

            transcription[round(adjusted_start_time, ndigits=2)] = text.strip()
            transcription_frame[frame_number] = text.strip()
            prompt_schedule += f'"{frame_number}": "{text.strip()}"' + (",\n" if chunk != result['chunks'][-1] else "\n")
            
            if media_type == "video":
                img = self.extract_frame(file_path, adjusted_start_time, duration)
                images.append(pil2tensor(img))
            else:
                img = Image.new('RGB', (512, 512), color='black')
                images.append(pil2tensor(img))

            last_end_time = end_time

        images = torch.cat(images, dim=0)

        return raw_text, transcription, transcription_frame, prompt_schedule, images

    def extract_audio(self, file_path, fps):
        os.makedirs(TEMP, exist_ok=True)
        clip = VideoFileClip(file_path)
        fps = fps or clip.fps
        duration = clip.duration
        frame_count = int(duration * fps)
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3', dir=TEMP)
        clip.audio.write_audiofile(tmp_file.name)
        clip.close()
        return tmp_file.name, fps, frame_count, duration

    def extract_frame(self, file_path, timestamp, video_duration):
        if timestamp > video_duration:
            return Image.new('RGB', (512, 512), color='black')
        with VideoFileClip(file_path) as clip:
            frame = clip.get_frame(timestamp)
        return Image.fromarray(frame)

    def validate(self, file_path):
        if any(file_path.lower().endswith(ext) for ext in self.video_extensions):
            return "video"
        elif any(file_path.lower().endswith(ext) for ext in self.audio_extensions):
            return "audio"
        else:
            return False


class SAIOpenAIAPIWhisper:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {}),
                "openai_key": ("STRING", {}),
            },
            "optional": {
                "model": (["whisper-1"],),
                "mode": (["transcribe", "translate_to_english"],),
                "language": ("STRING", {}),
                "response_format": (["text", "json", "verbose_json", "prompt_schedule"],),
                "temperature": ("FLOAT", {"min": 0.0, "max": 1.0, "default": 0.7}),
                "timestamp_granularities": (["segment", "word"],),
                "max_frames": ("INT", {"default": 0, "min": 0}),
                "seek_seconds": ("FLOAT", {"default": 0.0}),
                "prompt": ("STRING", {"multiline": True, "placeholder": "Optional prompt..."})
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("transcription_result", "audio_path", "frames_count")

    FUNCTION = "whisper_v2"
    CATEGORY = "SALT/Whisper"

    def whisper_v2(self, file_path, openai_key, model="whisper-1", mode="transcribe", language="", response_format="text", temperature=0.7, timestamp_granularities="segment", max_frames=0, seek_seconds=0.0, prompt=""):
        language = language if language else None
        prompt = prompt if prompt else None
        file_path = os.path.join(INPUT, file_path)

        if not os.path.exists(file_path):
            raise ValueError(f"The specified file `{file_path}` does not exist!")
        
        if model not in ("whisper-1"):
            raise ValueError(f"The specified model `{model}` does not exist!")

        if mode not in ("transcribe", "translate_to_english"):
            print(f'The `mode` selected "{mode}" is not valid. Please use either "transcribe", or "translate_to_english"')
            mode = "transcribe"

        openai.api_key = openai_key

        audio_path, fps, total_frames = self.extract_audio(file_path)

        max_frames = max_frames if max_frames != 0 else total_frames

        match mode:
            case "transcribe":
                if response_format == "prompt_schedule":
                    transcription = self.transcribe_audio(file_path, model, prompt, language, "verbose_json", temperature, timestamp_granularities, json_string=False)
                    out = self.prompt_schedule(transcription, fps, max_frames, seek_seconds)
                else:
                    out = self.transcribe_audio(file_path, model, prompt, language, response_format, temperature, timestamp_granularities)
            case "translate_to_english":
                out = self.translate_audio(file_path, model, prompt, response_format, temperature)

        return (out, audio_path, total_frames)

    def transcribe_audio(self, file_path, model="whisper-1", prompt=None, language=None, response_format="json", temperature=0.7, timestamp_granularities="segment", json_string=True):
        with open(file_path, "rb") as audio_file:
            response = openai.audio.transcriptions.create(
                model=model,
                file=audio_file,
                prompt=prompt,
                response_format=response_format,
                language=language,
                temperature=temperature,
                timestamp_granularities=timestamp_granularities
            )
            from pprint import pprint
            pprint(response, indent=4)
            if response_format in ("json", "verbose_json"):
                segments = getattr(response, 'segments', [])
                if json_string:
                    out = json.dumps(segments, ensure_ascii=True, indent=4)
                else:
                    out = segments
            else:
                out = response
            
            return out

    def translate_audio(self, file_path, model="whisper-1", prompt=None, response_format="json", temperature=0.7):
        with open(file_path, "rb") as audio_file:
            response = openai.audio.translations.create(
                model=model,
                file=audio_file,
                prompt=prompt,
                response_format=response_format,
                temperature=temperature,
            )
            from pprint import pprint
            pprint(response, indent=4)
            if response_format in ("json", "verbose_json"):
                segments = getattr(response, 'segments', [])
                out = json.dumps(segments, ensure_ascii=True, indent=4)
            else:
                out = response
            
            return out

    def extract_audio(self, file_path):
        clip = VideoFileClip(file_path)
        fps = clip.fps
        total_frames = int(clip.duration * fps)
        audio_path = os.path.join(OUTPUT, f"{os.path.splitext(os.path.basename(file_path))[0]}.mp3")
        clip.audio.write_audiofile(audio_path)
        clip.close()
        return audio_path, fps, total_frames
    
    def prompt_schedule(self, transcription, fps, max_frames, seek_seconds):
        prompt_schedule = ""
        max_seconds = max_frames / fps if max_frames > 0 else float('inf')
        start_frame = int(seek_seconds * fps)

        if isinstance(transcription, list):
            for idx, segment in enumerate(transcription):
                segment_start = segment.get("start", 0.0)
                if segment_start < seek_seconds or segment_start > max_seconds:
                    continue

                frame_number = int(segment_start * fps) - start_frame
                text = segment.get("text", "")
                if frame_number >= 0:
                    prompt_schedule += f'"{frame_number}": "{text.strip()}"' + (",\n" if idx != len(transcription) else "\n")

        return prompt_schedule

NODE_CLASS_MAPPINGS = {
    "SAIWhisperLoadModel": SAIWhisperLoadModel,
    "SAIWhisperTranscribe": SAIWhisperTranscribe,
    "SAIOpenAIAPIWhisper": SAIOpenAIAPIWhisper,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAIWhisperLoadModel": "Whisper Model Loader",
    "SAIWhisperTranscribe": "Whisper Transcribe",
    "SAIOpenAIAPIWhisper": "Whisper Transcribe (OpenAI API)"
}
