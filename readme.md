# Whisper Transcriber

Transcribe video and audio files with OpenAI's speech recoginition model Whisper

## Installation

- Clone this repository
- Install the requirements `pip install -r requirements.txt` 
	**Note:**, if you're using the portable version of ComfyUI you'll have to run against the environments python: `path\to\ComfyUI\python_embeded\python.exe -m pip install -r requirements.txt`
- Restart ComfyUI if running, and refresh any active tabs. 

## Nodes

### Whisper Model Loader
Load a OpenAI Whisper model

### Whisper Transcribe
Transcribe with `transformers`

### Whisper Transcribe (OpenAI API)
Transcribe with `openai` on Whisper V2 model. Requires a OpenAI access token. 

**Refer to [Salt AI Documentation](https://get-salt-ai.github.io/SaltAI-Web-Docs/) for node information.**