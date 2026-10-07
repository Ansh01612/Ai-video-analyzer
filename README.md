# Ai-video-analyzer
# 🎬 VideoMind — AI Video Assistant

Turn any YouTube video or local audio/video file into a transcript, summary, action items, key decisions, open questions — and chat with it using RAG.

## ✨ Features

- **Input:** YouTube URL or local file path
- **Transcription:** local Whisper (English) with Hinglish support
- **Summary & title** generation via Mistral (LangChain LCEL)
- **Extraction:** action items, key decisions, open questions
- **RAG chat:** ask questions about the video (ChromaDB + HuggingFace embeddings)
- **Web UI:** responsive HTML/CSS interface with light/dark theme

## 🗂 Project Structure

```
.
├── main.py              # Pipeline + CLI entry point
├── app.py               # Streamlit app
├── test.py              # Quick pipeline test script
├── index.html           # Standalone HTML/CSS web UI
├── Requirements.txt     # Python dependencies
├── .env                 # API keys (not committed)
├── core/                # transcriber, summarizer, extractor, rag_engine
└── utils/               # audio_processor
```

## ⚙️ Setup

**Prerequisites:** Python 3.10+ and [FFmpeg](https://ffmpeg.org/download.html) installed and on your PATH.

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r Requirements.txt

# 3. Add your API keys
```

Create a `.env` file in the project root:

```env
MISTRAL_API_KEY=your_mistral_key
SARVAM_API_KEY=your_sarvam_key   # only if using Hinglish
```

## ▶️ Usage

**CLI**

```bash
python main.py
```

Enter a YouTube URL or file path, choose a language (`english` / `hinglish`), then chat with the video.

**Streamlit app**

```bash
streamlit run app.py
```

**Quick test**

```bash
python test.py
```

## 🌐 HTML UI

`index.html` is a standalone front end. It shows demo data until connected to a backend.

1. Open `index.html` in a browser to preview.
2. To connect your backend, set `API_URL` in the script (e.g. `http://localhost:8000`).
3. The backend must expose:

| Endpoint | Request | Response |
|----------|---------|----------|
| `POST /process` | `{ "source": "...", "language": "english" }` | `{ title, summary, action_items, key_decisions, open_questions, transcript }` |
| `POST /ask` | `{ "question": "..." }` | `{ "answer": "..." }` |

## 🧰 Tech Stack

Whisper · yt-dlp · LangChain · Mistral · ChromaDB · Sentence-Transformers · Streamlit

## 📝 Notes

- Never commit your `.env` file (it is covered by `.gitignore`).
- The first run downloads the Whisper and embedding models, so it may take a while.

## 📄 License

Add your license here (e.g. MIT).
