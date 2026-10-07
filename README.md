# AI Video Assistant

Turn a YouTube video or a local audio/video file into a transcript, summary,
action items, key decisions, and open questions. The Python application also
supports question answering over the transcript with retrieval-augmented
generation (RAG).

## Features

- Download YouTube audio with `yt-dlp`, or load a local audio/video file.
- Convert audio and split it into chunks with FFmpeg and pydub.
- Transcribe English locally with Whisper.
- Transcribe Hinglish with Sarvam's speech-to-text translation API.
- Generate a title, summary, action items, decisions, and open questions with
  Mistral.
- Build a local Chroma vector store and ask questions about the transcript.
- Use the Streamlit interface (`app.py`), command-line pipeline (`main.py`), or
  standalone HTML demo (`index.html`).

## Requirements

- Python 3.10 or newer. Python 3.12 is recommended for compatibility with the
  audio dependencies.
- FFmpeg installed and available on `PATH`.
- A Mistral API key for summaries, extraction, titles, and RAG chat.
- A Sarvam API key when transcribing in Hinglish.
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
MISTRAL_API_KEY=your_mistral_api_key
SARVAM_API_KEY=your_sarvam_api_key
```

The Sarvam key is only needed for Hinglish transcription. Do not commit API keys
or paste them into source files.

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

### Standalone HTML interface

Serve the project directory locally and open `index.html`:

```bash
python -m http.server 8000 --bind 127.0.0.1
```

Then visit <http://127.0.0.1:8000/index.html>. This HTML page is a frontend
demo: `API_URL` is empty by default, so it displays sample results and simulated
progress rather than invoking the Python pipeline. Its `/process` and `/ask`
requests require a compatible backend; this project does not currently expose
those HTTP endpoints.

### Vercel deployment

The static HTML frontend is deployed at
<https://ai-video-analyzer-4454.vercel.app>. The Vercel project is connected to
this GitHub repository's `main` branch, so future pushes trigger deployments.
Vercel serves `index.html` from the repository root without a build command.

This is the frontend demo only. The Python/Streamlit pipeline is not hosted by
this static deployment, and the page will continue to show demo data until a
compatible backend is deployed and `API_URL` is configured.

## Configuration

The following environment variables are supported:

| Variable | Default | Purpose |
| --- | --- | --- |
| `MISTRAL_API_KEY` | — | Mistral chat completions for summaries and RAG |
| `SARVAM_API_KEY` | — | Sarvam transcription for Hinglish |
| `WHISPER_MODEL` | `small` | Local Whisper model size |
| `SARVAM_STT_MODEL` | `saaras:v2.5` | Sarvam transcription model |

## Project structure

```text
.
├── app.py                 # Streamlit web application
├── index.html             # Standalone HTML demo frontend
├── main.py                # Command-line analysis and RAG chat
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
