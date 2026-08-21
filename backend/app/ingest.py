# backend/app/ingest.py
import json
import argparse
from pathlib import Path
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
from app.chunking import chunk_transcript

DATA_PATH = Path(__file__).parent.parent / "data" / "transcripts.json"

def list_playlist_videos(playlist_url: str) -> list[dict]:
    opts = {"extract_flat": True, "quiet": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(playlist_url, download=False)
    return [{"video_id": e["id"], "title": e.get("title", "Untitled")} for e in info["entries"]]

def get_transcript(video_id: str) -> list[dict]:
    try:
        api = YouTubeTranscriptApi()
        transcript = api.fetch(video_id)   # returns a FetchedTranscript object (iterable)
        return [{"text": s.text, "start": s.start, "duration": s.duration} for s in transcript]
    except Exception as e:
        print(f"  ✗ primary failed for {video_id}: {e} — trying fallback")
        return fallback_ytdlp_captions(video_id)

def fallback_ytdlp_captions(video_id: str) -> list[dict]:
    # Simplified fallback — pulls auto captions via yt-dlp when the transcript API is blocked
    opts = {"skip_download": True, "writeautomaticsub": True, "subtitleslangs": ["en"], "quiet": True}
    url = f"https://www.youtube.com/watch?v={video_id}"
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.extract_info(url, download=False)
    return []  # stub: parse downloaded .vtt if you want full fallback — note this honestly as a known gap

def ingest(playlist_url: str):
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

    DATA_PATH.parent.mkdir(exist_ok=True)
    with open(DATA_PATH, "w") as f:
        json.dump(all_chunks, f, indent=2)
    print(f"Wrote {len(all_chunks)} total chunks to {DATA_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--playlist", required=True)
    args = parser.parse_args()
    ingest(args.playlist)