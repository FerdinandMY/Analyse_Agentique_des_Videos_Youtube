from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


@dataclass(frozen=True)
class YouTubeAPI:
    """
    Minimal wrapper around YouTube Data API v3.

    This is a scaffolding class: it does not cover all endpoints yet.
    """

    api_key: str

    def fetch_video_comments(
        self,
        video_id: str,
        *,
        max_results: int = 50,
        max_pages: int = 1,
    ) -> List[Dict[str, Any]]:
        # Placeholder: implement the real API calls when integrating the agents.
        # Kept as a function to make it easy to mock in tests.
        url = "https://www.googleapis.com/youtube/v3/commentThreads"
        params: Dict[str, Any] = {
            "part": "snippet",
            "videoId": video_id,
            "maxResults": max_results,
            "key": self.api_key,
            "textFormat": "plainText",
        }

        all_items: List[Dict[str, Any]] = []
        next_page_token: Optional[str] = None
        pages = 0

        while pages < max_pages:
            if next_page_token:
                params["pageToken"] = next_page_token
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            items = data.get("items", [])
            all_items.extend(items)
            next_page_token = data.get("nextPageToken")
            pages += 1
            if not next_page_token:
                break

        return all_items