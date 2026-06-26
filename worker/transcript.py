import json
import asyncio
from urllib.parse import urlparse, parse_qs
import urllib.request
import yt_dlp


def log(event: str, **kwargs):
    print(json.dumps({"event": event, **kwargs}, ensure_ascii=False), flush=True)


def extract_video_id(url: str) -> str:
    parsed = urlparse(url)
    if parsed.hostname in ("youtu.be",):
        return parsed.path.lstrip("/")
    if parsed.hostname in ("www.youtube.com", "youtube.com"):
        video_id = parse_qs(parsed.query).get("v", [None])[0]
        if video_id:
            return video_id
    raise ValueError(f"Could not extract video ID from URL: {url}")


def fetch_video_info(url: str) -> tuple[str, str]:
    """Returns (title, language_code)."""
    try:
        ydl_opts = {
            "quiet": True,
            "skip_download": True,
            "no_warnings": True,
            "ignore_errors": True,
            "extract_flat": "in_playlist",
            "format": None,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if info is None:
                log("yt_dlp_no_info", url=url)
                return "Unknown Title", "en"
            title = info.get("title") or "Unknown Title"
            lang = info.get("language") or info.get("default_audio_language") or "en"
            lang = lang.split("-")[0].lower()
            log("yt_dlp_info_fetched", url=url, title=title, lang=lang)
            return title, lang
    except Exception as e:
        log("yt_dlp_error", url=url, error=str(e))
        return "Unknown Title", "en"


def fetch_transcript(video_id: str, lang: str) -> str:
    log("transcript_fetch_start", video_id=video_id, lang=lang)
    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        # Match preferred language variants dynamically using wildcards
        ydl_opts = {
            "skip_download": True,
            "subtitleslangs": ["orig", f"{lang}.*", "ru.*", "en.*", ".*-orig", ".*"],
            "quiet": True,
            "no_warnings": True,
            "ignore_errors": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

        if not info:
            raise RuntimeError(f"Failed to extract info via yt-dlp for video {video_id}")

        auto = info.get("automatic_captions", {})
        subtitles = info.get("subtitles", {})

        # Priority fallback check to find any valid native language track matching our target
        caps = None
        for key in ["orig", f"{lang}-orig", lang, "ru-orig", "ru", "en-orig", "en"]:
            if key in subtitles:
                caps = subtitles[key]
                break
            if key in auto:
                caps = auto[key]
                break

        # Fall back to any 'orig' string match if specific codes were missing
        if not caps:
            all_tracks = {**auto, **subtitles}
            orig_key = next((k for k in all_tracks.keys() if "orig" in k), None)
            if orig_key:
                caps = all_tracks[orig_key]
            else:
                caps = next(iter(subtitles.values()), None) or next(iter(auto.values()), None)

        if not caps:
            raise RuntimeError(f"No captions found for video {video_id}")

        cap_url = next((f["url"] for f in caps if f.get("ext") == "json3"), None)
        if not cap_url:
            raise RuntimeError(f"No json3 caption format for video {video_id}")

        req = urllib.request.Request(
            cap_url,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )

        with urllib.request.urlopen(req) as r:
            data = json.loads(r.read())

        text = " ".join(
            seg.get("utf8", "") for e in data.get("events", []) for seg in e.get("segs", []) if seg.get("utf8")
        ).replace("\n", " ")

        if not text.strip():
            raise RuntimeError(f"Extracted json3 text data payload evaluated as empty for {video_id}")

        log("transcript_fetch_success", video_id=video_id, chars=len(text))
        return text
    except RuntimeError:
        raise
    except Exception as e:
        log("transcript_error", video_id=video_id, error=str(e))
        raise RuntimeError(f"Could not fetch transcript for video {video_id}: {e}")


async def build_transcript_context(url: str) -> tuple[str, str, str]:
    """Returns (title, lang, transcript)."""
    loop = asyncio.get_event_loop()

    video_id = extract_video_id(url)
    title, lang = await loop.run_in_executor(None, fetch_video_info, url)
    transcript = await loop.run_in_executor(None, fetch_transcript, video_id, lang)

    return title, lang, transcript
