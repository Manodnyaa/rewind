from app.config import settings

def chunk_transcript(entries: list[dict], video_id: str, title: str, youtube_url: str) -> list[dict]:
    """
    entries: [{"text": str, "start":float, "duration":float }, ...]
    Groups entries into overlapping fixed-duration windows.
    """
    if not entries:
        return []
    window = settings.chunk_window_seconds
    overlap = settings.chunk_overlap_seconds
    step = window - overlap

    max_time = entries[-1]["start"] + entries[-1].get("duration",0)
    chunks = []
    window_start = 0.0

    while window_start<max_time:
        window_end = window_start + window
        texts_in_window = [
            e["text"] for e in entries
            if window_start <= e["start"] < window_end
        ]
        if texts_in_window:
            chunks.append({
                "video_id": video_id,
                "title":title,
                "text":" ".join(texts_in_window).strip(),
                "start_seconds": window_start,
                "end_seconds": window_end,
                "youtube_url": youtube_url,
            })
        window_start += step

    return chunks
