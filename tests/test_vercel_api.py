import unittest
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi import HTTPException
import requests
from yt_dlp.utils import DownloadError

from fastapi.testclient import TestClient

from index import (
    AskRequest,
    ProcessRequest,
    app,
    _download_youtube_captions,
    _retrieve_context,
    _validated_youtube_url,
    _openrouter_chat,
    _parse_webvtt,
    _transcribe_audio,
    ask_question,
    process_video,
)


class VercelApiTests(unittest.TestCase):
    def test_webvtt_parser_extracts_and_deduplicates_caption_lines(self):
        content = (
            "WEBVTT\n\n"
            "00:00:00.100 --> 00:00:02.000\nHello &amp; welcome.\n\n"
            "00:00:02.000 --> 00:00:04.000\nHello & welcome.\n\n"
            "00:00:04.000 --> 00:00:06.000\nNext <b>important</b> point.\n"
        )

        self.assertEqual(
            _parse_webvtt(content),
            "Hello & welcome. Next important point.",
        )

    def test_caption_fallback_uses_android_vr_client_for_listing_and_download(self):
        video_url = "https://youtu.be/video-id"
        with tempfile.TemporaryDirectory() as directory:
            metadata_downloader = MagicMock()
            metadata_downloader.__enter__.return_value = metadata_downloader
            metadata_downloader.extract_info.return_value = {
                "subtitles": {"en": [{"ext": "vtt"}]},
                "automatic_captions": {},
            }

            caption_downloader = MagicMock()
            caption_downloader.__enter__.return_value = caption_downloader

            def write_caption_file(urls):
                self.assertEqual(urls, [video_url])
                (Path(directory) / "captions.en.vtt").write_text(
                    "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nCaption text.\n",
                    encoding="utf-8",
                )
                return 0

            caption_downloader.download.side_effect = write_caption_file

            with patch(
                "index._youtube_downloader",
                side_effect=[metadata_downloader, caption_downloader],
            ) as create_downloader:
                transcript = _download_youtube_captions(
                    video_url, Path(directory), "english"
                )

        self.assertEqual(transcript, "Caption text.")
        self.assertEqual(create_downloader.call_count, 2)
        for call in create_downloader.call_args_list:
            self.assertEqual(
                call.args[0]["extractor_args"]["youtube"]["player_client"],
                ["android_vr"],
            )

    def test_openrouter_rate_limit_returns_actionable_message(self):
        response = requests.Response()
        response.status_code = 429
        http_error = requests.HTTPError(response=response)

        with (
            patch("index._required_key", return_value="test-key"),
            patch("index.requests.post") as post,
        ):
            post.return_value.raise_for_status.side_effect = http_error
            with self.assertRaises(HTTPException) as error:
                _openrouter_chat("system", "user")

        self.assertEqual(error.exception.status_code, 503)
        self.assertIn("OpenRouter rate limit", error.exception.detail)

    def test_openrouter_uses_compatible_endpoint_and_configured_model(self):
        response = requests.Response()
        response.status_code = 200
        response._content = b'{"choices":[{"message":{"content":"OK"}}]}'

        with (
            patch("index._required_key", return_value="test-key"),
            patch("index.requests.post", return_value=response) as post,
        ):
            result = _openrouter_chat("system", "user")

        self.assertEqual(result, "OK")
        self.assertEqual(
            post.call_args.args[0],
            "https://openrouter.ai/api/v1/chat/completions",
        )
        self.assertEqual(
            post.call_args.kwargs["headers"]["Authorization"],
            "Bearer test-key",
        )
        self.assertEqual(post.call_args.kwargs["json"]["model"], "openrouter/free")

    def test_sarvam_rate_limit_returns_actionable_message(self):
        response = requests.Response()
        response.status_code = 429

        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "audio.wav"
            audio_path.write_bytes(b"test")
            with (
                patch.dict(os.environ, {"SARVAM_API_KEY": "test-key"}),
                patch("index.requests.post", return_value=response),
                self.assertRaises(HTTPException) as error,
            ):
                _transcribe_audio([audio_path], "english")

        self.assertEqual(error.exception.status_code, 502)
        self.assertIn("Sarvam rate limit or account quota", error.exception.detail)

    def test_frontend_and_health_endpoint_are_served_by_the_api_app(self):
        client = TestClient(app)

        page = client.get("/")
        health = client.get("/api/health")

        self.assertEqual(page.status_code, 200)
        self.assertIn('const API_URL = "/api"', page.text)
        self.assertEqual(health.json(), {"status": "ok"})

    def test_process_route_rejects_visitor_local_paths(self):
        response = TestClient(app).post(
            "/api/process",
            json={"source": "C:\\Videos\\meeting.mp4", "language": "english"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Local file paths", response.json()["detail"])

    def test_accepts_youtube_video_urls(self):
        self.assertEqual(
            _validated_youtube_url("https://youtu.be/video-id"),
            "https://youtu.be/video-id",
        )

    def test_rejects_local_paths_and_non_youtube_hosts(self):
        for source in ("C:\\Videos\\meeting.mp4", "https://example.com/video"):
            with self.subTest(source=source), self.assertRaises(HTTPException) as error:
                _validated_youtube_url(source)
            self.assertEqual(error.exception.status_code, 400)

    def test_retrieves_relevant_transcript_passage(self):
        transcript = (
            "The design team discussed the color palette and typography. "
            "The launch schedule is set for September after the final QA review. "
            "The finance team will revisit the annual budget next quarter."
        )

        context = _retrieve_context(transcript, "When is the launch schedule?")

        self.assertIn("September", context)
        self.assertNotIn("annual budget", context)

    def test_returns_empty_context_when_no_terms_match(self):
        self.assertEqual(
            _retrieve_context("The team reviewed the meeting.", "Who owns quantum physics?"),
            "",
        )

    def test_process_endpoint_returns_frontend_result_shape(self):
        transcript = "The launch is planned for September after QA."
        with (
            patch("index._download_and_chunk_audio", return_value=[]) as download,
            patch("index._transcribe_audio", return_value=transcript),
            patch("index._required_key", return_value="test-key"),
            patch(
                "index._openrouter_chat",
                side_effect=["Partial summary", "Summary", "Title", "Actions", "Decisions", "Questions"],
            ),
        ):
            result = process_video(
                ProcessRequest(source="https://youtu.be/video-id", language="english")
            )

        download.assert_called_once()
        self.assertEqual(
            set(result),
            {
                "title",
                "summary",
                "action_items",
                "key_decisions",
                "open_questions",
                "transcript",
            },
        )
        self.assertEqual(result["transcript"], transcript)

    def test_analyze_transcript_endpoint_returns_frontend_result_shape(self):
        transcript = "The launch is planned for September after QA."
        with (
            patch("index._required_key", return_value="test-key"),
            patch(
                "index._openrouter_chat",
                side_effect=["Partial summary", "Summary", "Title", "Actions", "Decisions", "Questions"],
            ),
        ):
            response = TestClient(app).post(
                "/api/analyze-transcript",
                json={"transcript": f"  {transcript}  "},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            set(response.json()),
            {
                "title",
                "summary",
                "action_items",
                "key_decisions",
                "open_questions",
                "transcript",
            },
        )
        self.assertEqual(response.json()["transcript"], transcript)

    def test_analyze_transcript_endpoint_rejects_blank_text(self):
        with patch("index._required_key", return_value="test-key"):
            response = TestClient(app).post(
                "/api/analyze-transcript",
                json={"transcript": "  \n  "},
            )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"], "Paste a transcript to analyze.")

    def test_process_uses_youtube_captions_if_audio_download_is_blocked(self):
        transcript = "The video transcript from YouTube captions."
        audio_error = HTTPException(
            status_code=502,
            detail="YouTube audio download failed: HTTP Error 403",
        )
        with (
            patch("index._download_and_chunk_audio", side_effect=audio_error),
            patch("index._download_youtube_captions", return_value=transcript) as captions,
            patch("index._transcribe_audio") as transcribe,
            patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True),
            patch(
                "index._openrouter_chat",
                side_effect=["Partial summary", "Summary", "Title", "Actions", "Decisions", "Questions"],
            ),
        ):
            result = process_video(
                ProcessRequest(source="https://youtu.be/video-id", language="english")
            )

        captions.assert_called_once()
        transcribe.assert_not_called()
        self.assertEqual(result["transcript"], transcript)

    def test_process_uses_youtube_captions_when_audio_exceeds_size_limit(self):
        transcript = "Transcript from captions."
        audio_error = HTTPException(
            status_code=413,
            detail="The downloaded video audio exceeds the 100 MB limit.",
        )
        with (
            patch("index._download_and_chunk_audio", side_effect=audio_error),
            patch("index._download_youtube_captions", return_value=transcript) as captions,
            patch("index._transcribe_audio") as transcribe,
            patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True),
            patch(
                "index._openrouter_chat",
                side_effect=["Partial summary", "Summary", "Title", "Actions", "Decisions", "Questions"],
            ),
        ):
            result = process_video(
                ProcessRequest(source="https://youtu.be/video-id", language="english")
            )

        captions.assert_called_once()
        transcribe.assert_not_called()
        self.assertEqual(result["transcript"], transcript)

    def test_caption_listing_bot_check_error_is_explained(self):
        downloader = MagicMock()
        downloader.__enter__.return_value = downloader
        downloader.extract_info.side_effect = DownloadError(
            "Sign in to confirm you're not a bot"
        )
        with (
            patch("index._youtube_downloader", return_value=downloader),
            tempfile.TemporaryDirectory() as directory,
            self.assertRaises(HTTPException) as error,
        ):
            _download_youtube_captions(
                "https://youtu.be/video-id", Path(directory), "english"
            )

        self.assertIn("blocking caption access from this server", error.exception.detail)

    def test_ask_endpoint_uses_retrieved_context(self):
        request = AskRequest(question="When is launch?", transcript="Video transcript.")
        with (
            patch("index._retrieve_context", return_value="Launch is scheduled for September.") as retrieve,
            patch("index._openrouter_chat", return_value="September.") as chat,
        ):
            result = ask_question(request)

        retrieve.assert_called_once_with(request.transcript, request.question)
        self.assertEqual(result, {"answer": "September."})
        self.assertIn("September", chat.call_args.args[0])

    def test_ask_endpoint_does_not_call_model_without_relevant_context(self):
        request = AskRequest(question="Who owns quantum physics?", transcript="A planning meeting.")
        with (
            patch("index._retrieve_context", return_value=""),
            patch("index._openrouter_chat") as chat,
        ):
            result = ask_question(request)

        chat.assert_not_called()
        self.assertEqual(
            result,
            {"answer": "I could not find this information in the video transcript."},
        )


if __name__ == "__main__":
    unittest.main()
