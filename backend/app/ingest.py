# backend/app/ingest.py
import json
import argparse
from urllib.parse import urlparse
from pathlib import Path
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
from app.chunking import chunk_transcript

DATA_PATH = Path(__file__).parent.parent / "data" / "transcripts.json"

#validation
def validate_url(url:str) -> None:
    host = urlparse(url).netloc.lower()
    if not host or not any(h in host for h in ("youtube.com", "youtu.be")):
        raise ValueError("That is not a YouTube link.")
    
def list_playlist_videos(playlist_url: str) -> list[dict]:
    validate_url(playlist_url)
    opts = {"extract_flat": True, "quiet": True}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(playlist_url, download=False)
    except yt_dlp.utils.DownloadError as e:
        raise ValueError(f"Link is invalid, private or unavailable: {e}") from e
    if not info:
        raise ValueError("Nothing found at that link.")
    entries = info["entries"] if "entries" in info else [info]   # NEW: playlist OR single video
    videos = [{"video_id": e["id"], "title": e.get("title") or "Untitled"}
              for e in entries if e and e.get("id")]
    
    if not videos:
        raise ValueError("The playlist has no accessible videos.")
    return videos

def get_transcript(video_id: str) -> list[dict]:
    try:
        api = YouTubeTranscriptApi()
        transcript = api.fetch(video_id)   # returns a FetchedTranscript object (iterable)
        return [{"text": s.text, "start": s.start, "duration": s.duration} for s in transcript]
    except Exception as e:
        print(f"  ✗ primary failed for {video_id}: {e} — trying fallback")
        try:
            return fallback_ytdlp_captions(video_id)
        except Exception as e2:
            print (f" fallback failed too: {e2}")
            return []

def fallback_ytdlp_captions(video_id: str) -> list[dict]:
    # Simplified fallback — pulls auto captions via yt-dlp when the transcript API is blocked
    opts = {"skip_download": True, "writeautomaticsub": True, "subtitleslangs": ["en"], "quiet": True}
    url = f"https://www.youtube.com/watch?v={video_id}"
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.extract_info(url, download=False)
    return []  # stub: parse downloaded .vtt if you want full fallback — note this honestly as a known gap

def ingest(playlist_url: str, out_path: Path = DATA_PATH):
    videos = list_playlist_videos(playlist_url)
    all_chunks = []
    for v in videos:
        entries = get_transcript(v["video_id"])
        if not entries:
            print(f"Skipping {v['title']} — no transcript")
            continue
        yt_url = f"https://www.youtube.com/watch?v={v['video_id']}"
        chunks = chunk_transcript(entries, v["video_id"], v["title"], yt_url)
        all_chunks.extend(chunks)
        print(f"Ingested {v['title']}: {len(chunks)} chunks")

    if not all_chunks:                                    # NEW: never overwrite good data with nothing
        raise ValueError("No transcripts could be fetched for any video. Data file left untouched.")
    
    out_path.parent.mkdir(exist_ok=True)
    tmp = out_path.with_suffix(".tmp")                    # NEW: write to a temp file, then swap in
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)
    tmp.replace(out_path)
    print(f"Wrote {len(all_chunks)} chunks to {out_path}. Skipped {len(skipped)} video(s).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--playlist", required=True)
    parser.add_argument("--out", default=str(DATA_PATH))
    args = parser.parse_args()
    try:
        ingest(args.playlist)
    except ValueError as e:
        print(f"Error: {e}")
        raise SystemExit(1)