from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import subprocess

import IPython.display
import numpy as np
from pytimeparse.timeparse import timeparse
import librosa
import soundfile as sf


def extract_youtube_code(url: str) -> str:
    """
    Input: a YouTube URL, such as 'https://www.youtube.com/watch?v=DShJmzVyaH4'
    Output: a code, such as 'DShJmzVyaH4' (without quotes)
    """
    return parse_qs(urlparse(url).query)['v'][0]


def download_youtube_audio(
    url: str,
    output_path: str | Path,
    keep_video: bool = False,
):
    """
    Downloads and saves WAV audio file from YouTube to the specified path.

    Path should have .wav extension, or it is added automatically.

    If keep_video=True, keeps downloaded .webm and .mp4 files also.

    Requires `pip install yt-dlp` (is included in requirements.txt).
    """
    args = [
        'yt-dlp',
        url,
        '-o',
        str(output_path),
        '--extract-audio',
        '--audio-format',
        'wav',
        '--audio-quality',
        '5',
    ]
    if keep_video:
        args.append('-k')
    
    subprocess.run(args)


def parse_time(t: str | float) -> float:
    """
    Time may be provided as float (seconds) or string:
    
    "1:23" (treated as mm:ss)
    "1:23:45" (treated as hh:mm:ss)
    "2:40:56,500"
    "2:40:56.500"
    """
    return (
        timeparse(t.replace(',', '.'))
        if isinstance(t, str)
        else t
    )


@dataclass
class Audio:
    array: np.ndarray
    rate: int

    @classmethod
    def load(cls, path: str | Path) -> Audio:
        return Audio(*librosa.load(path))

    def save(self, path: str | Path):
        Path(path).resolve().parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, self.array, self.rate, format='wav')
    
    def display(self):
        IPython.display.display(IPython.display.Audio(self.array, rate=self.rate))
    
    @property
    def total_seconds(self) -> float:
        return len(self.array) / self.rate

    def locate(self, t: float | None) -> int | None:
        """
        Converts time to index in self.array.
        """
        return (
            round(t * self.rate)
            if t is not None
            else None
        )

    def trim(
        self,
        start: float | None = None,
        end: float | None = None,
    ) -> Audio:
        """
        Returns trimmed copy of the audio. See .locate() method for time formats.

        Example: `audio.trim('12:56,345', None)`

        Raises ValueError if time is not in range.
        """
        return Audio(
            self.array[self.locate(start):self.locate(end)],
            self.rate
        )