# AI Video Assistant

Turn a YouTube video or a local audio/video file into a transcript, summary,
action items, key decisions, and open questions. The Python application also
supports question answering over the transcript with retrieval-augmented
generation (RAG).

## Features

- Download YouTube audio with `yt-dlp`, with a YouTube captions fallback when
  audio downloads are blocked; local audio/video files are supported in the
  Streamlit and command-line interfaces.
- Convert audio and split it into chunks with FFmpeg and pydub.
- Transcribe downloaded YouTube audio with Sarvam's speech-to-text API.
- Generate a title, summary, action items, decisions, and open questions with
  OpenRouter.
- Build a local Chroma vector store and ask questions about the transcript.
- Use the Streamlit interface (`app.py`), command-line pipeline (`main.py`), or
  the HTML frontend and FastAPI backend (`index.html`, `api.py`).

## Requirements

- Python 3.12 or newer.
- FFmpeg installed and available on `PATH`.
- An OpenRouter API key for summaries, extraction, titles, and RAG chat.
- A Sarvam API key when transcribing YouTube audio.
- Internet access for YouTube downloads, provider APIs, and the first download
  of Whisper and embedding models.

## Installation

### Windows (PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r Requirements.txt
```

### macOS / Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r Requirements.txt
```

Install FFmpeg using your operating system's package manager or the instructions
at [ffmpeg.org](https://ffmpeg.org/download.html), then confirm `ffmpeg` is
available in a new terminal.

## API keys

Create a `.env` file in the project root. Keep real keys private; `.env` is
excluded from Git.

```dotenv
OPENROUTER_API_KEY=your_openrouter_api_key
SARVAM_API_KEY=your_sarvam_api_key
OPENROUTER_MODEL=openrouter/free
```

OpenRouter supplies the chat model; `openrouter/free` is the default router
model and can be changed with `OPENROUTER_MODEL`. Sarvam transcribes downloaded
audio; if YouTube blocks the audio download, available YouTube captions are
used instead. Do not commit API keys or paste them into source files.

## Run the application

### Streamlit interface

```bash
streamlit run app.py
```

Enter a YouTube URL or local file path, choose English or Hinglish, and select
**Analyse**. The first run can take longer while Whisper and the embedding model
are downloaded.

### Command-line interface

```bash
python main.py
```

Follow the prompts to provide a video URL or local file path and select the
transcription language. After processing, the CLI allows questions about the
transcript.

### HTML and API

Install the project dependencies, then start the FastAPI application:

```powershell
python -m pip install -r Requirements.txt
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. The page calls `/api/process` and `/api/ask` on the
same FastAPI application. Set `OPENROUTER_API_KEY` and `SARVAM_API_KEY` in the
environment (or a root `.env` file) before analyzing a video. This API accepts
YouTube URLs; a local path on a visitor's computer is not accessible to a
deployed server.

### Vercel deployment

The repository is configured to deploy both the HTML page and FastAPI backend
as one Vercel project. Vercel uses `api:app` as the Python entrypoint and
`vercel.json` allows up to 300 seconds for video processing. In the Vercel
project settings, add these Environment Variables for Production (and Preview
if needed), then redeploy:

- `OPENROUTER_API_KEY` — title, summary, action items, decisions, open
  questions, and RAG answers.
- `OPENROUTER_MODEL` — model ID or router, default `openrouter/free`.
- `SARVAM_API_KEY` — English and Hinglish video transcription.

Alternatively, after linking the project with the Vercel CLI, add the variables
using `vercel env add OPENROUTER_API_KEY` and `vercel env add SARVAM_API_KEY`,
then deploy with `vercel --prod`. Do not put API keys in the HTML or commit them.
The existing frontend URL is <https://ai-video-analyzer-4454.vercel.app>.

The Vercel API converts YouTube audio into short segments for Sarvam, then
returns analysis to the page. Chat requests send the active transcript to the
API, which retrieves relevant transcript passages for OpenRouter; no server-side
session or vector database is required. Long downloads or provider calls can
still exceed Vercel's function duration limit.

## Configuration

The following environment variables are supported:

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | — | OpenRouter chat completions for summaries and RAG |
| `OPENROUTER_MODEL` | `openrouter/free` | OpenRouter model ID or router |
| `SARVAM_API_KEY` | — | Sarvam transcription for English and Hinglish |
| `WHISPER_MODEL` | `small` | Local Whisper model size |
| `SARVAM_STT_MODEL` | `saaras:v2.5` | Sarvam transcription model |

## Project structure

```text
.
├── app.py                 # Streamlit web application
├── api.py                 # FastAPI backend for Vercel and local HTML UI
├── index.html             # HTML frontend connected to /api/process and /api/ask
├── main.py                # Command-line analysis and RAG chat
├── pyproject.toml         # Vercel Python entrypoint and lightweight dependencies
├── vercel.json            # Vercel function duration configuration
├── Requirements.txt       # Python dependencies
├── core/
│   ├── extractor.py       # Action items, decisions, and questions
│   ├── rag_engine.py      # Transcript question-answering chain
│   ├── summarizer.py      # Title and summary generation
│   ├── transcriber.py     # Whisper and Sarvam transcription
│   └── vector_store.py    # Chroma and embedding setup
└── utils/
    └── audio_processor.py # Download, convert, and chunk media
```

Generated downloads and vector data are stored locally and should not be
committed.

## License

This project is distributed under the Unlicense. See [LICENSE](LICENSE).
