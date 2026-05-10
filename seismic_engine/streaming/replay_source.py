"""
Replay data source and ring buffer for simulated real-time streaming.

Feeds pre-recorded waveform data into a circular buffer at real-time pace,
simulating a SeedLink connection for demo purposes without network dependency.
"""

import time
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np


class RingBuffer:
    """
    Circular buffer for 3-component seismic data.

    Maintains a fixed-length window of the most recent samples,
    supporting O(1) append and O(1) read of the current window.
    """

    def __init__(self, capacity_samples: int = 6000, n_channels: int = 3):
        self.capacity = capacity_samples
        self.n_channels = n_channels
        self._buffer = np.zeros((n_channels, capacity_samples), dtype=np.float32)
        self._write_pos = 0
        self._total_written = 0

    @property
    def is_full(self) -> bool:
        return self._total_written >= self.capacity

    @property
    def fill_level(self) -> float:
        return min(1.0, self._total_written / self.capacity)

    @property
    def total_samples_received(self) -> int:
        return self._total_written

    def append(self, data: np.ndarray) -> None:
        """
        Append new samples to the ring buffer.

        data: (n_channels, n_new_samples) array
        """
        n_new = data.shape[1]

        if n_new >= self.capacity:
            self._buffer[:] = data[:, -self.capacity:]
            self._write_pos = 0
            self._total_written += n_new
            return

        end_pos = self._write_pos + n_new
        if end_pos <= self.capacity:
            self._buffer[:, self._write_pos:end_pos] = data
        else:
            first_chunk = self.capacity - self._write_pos
            self._buffer[:, self._write_pos:] = data[:, :first_chunk]
            self._buffer[:, :n_new - first_chunk] = data[:, first_chunk:]

        self._write_pos = end_pos % self.capacity
        self._total_written += n_new

    def get_window(self, window_samples: Optional[int] = None) -> np.ndarray:
        """
        Get the most recent `window_samples` from the buffer.

        Returns: (n_channels, window_samples) array in chronological order.
        """
        if window_samples is None:
            window_samples = self.capacity

        available = min(self._total_written, self.capacity)
        n = min(window_samples, available)

        if n == 0:
            return np.zeros((self.n_channels, window_samples), dtype=np.float32)

        end = self._write_pos
        start = (end - n) % self.capacity

        if start < end:
            result = self._buffer[:, start:end].copy()
        else:
            result = np.concatenate([
                self._buffer[:, start:],
                self._buffer[:, :end],
            ], axis=1)

        if result.shape[1] < window_samples:
            pad = np.zeros((self.n_channels, window_samples - result.shape[1]), dtype=np.float32)
            result = np.concatenate([pad, result], axis=1)

        return result

    def get_latest(self, n_samples: int) -> np.ndarray:
        """Get the N most recent samples (for plotting the visible window)."""
        return self.get_window(n_samples)

    def reset(self) -> None:
        self._buffer[:] = 0
        self._write_pos = 0
        self._total_written = 0


class ReplaySource:
    """
    Simulates real-time data streaming from pre-recorded waveforms.

    Feeds data into a RingBuffer at a configurable pace, yielding
    chunks that simulate arriving packets from a SeedLink connection.
    """

    def __init__(
        self,
        data: np.ndarray,
        sample_rate: float = 100.0,
        chunk_samples: int = 100,
        station: str = "KO.KMRS",
    ):
        """
        Args:
            data: (3, N) full waveform to replay
            sample_rate: samples per second
            chunk_samples: samples per "packet" (1 second at 100 Hz)
            station: station identifier for display
        """
        self.data = data.astype(np.float32)
        self.sample_rate = sample_rate
        self.chunk_samples = chunk_samples
        self.station = station
        self.total_samples = data.shape[1]
        self._position = 0

    @property
    def duration_seconds(self) -> float:
        return self.total_samples / self.sample_rate

    @property
    def elapsed_seconds(self) -> float:
        return self._position / self.sample_rate

    @property
    def progress(self) -> float:
        return self._position / self.total_samples if self.total_samples > 0 else 0

    @property
    def is_exhausted(self) -> bool:
        return self._position >= self.total_samples

    def reset(self) -> None:
        self._position = 0

    def next_chunk(self) -> Optional[np.ndarray]:
        """
        Get the next chunk of data (simulates one packet arrival).

        Returns: (3, chunk_samples) array, or None if exhausted.
        """
        if self._position >= self.total_samples:
            return None

        end = min(self._position + self.chunk_samples, self.total_samples)
        chunk = self.data[:, self._position:end]
        self._position = end
        return chunk

    def stream_chunks(self, real_time: bool = False) -> Generator[np.ndarray, None, None]:
        """
        Generator yielding chunks. If real_time=True, sleeps between chunks.
        """
        chunk_duration = self.chunk_samples / self.sample_rate

        while not self.is_exhausted:
            chunk = self.next_chunk()
            if chunk is None:
                break
            if real_time:
                time.sleep(chunk_duration)
            yield chunk

    @classmethod
    def from_hdf5(
        cls,
        hdf5_path: Path,
        trace_name: str,
        sample_rate: float = 100.0,
        chunk_samples: int = 100,
        station: str = "KO.KMRS",
    ) -> "ReplaySource":
        """Load a trace from HDF5 and create a replay source."""
        import h5py
        with h5py.File(str(hdf5_path), "r") as hf:
            data = hf["data"][trace_name][:]
        return cls(data, sample_rate, chunk_samples, station)

    @classmethod
    def from_mseed(
        cls,
        mseed_path: Path,
        sample_rate: float = 100.0,
        chunk_samples: int = 100,
    ) -> "ReplaySource":
        """Load a MiniSEED file and create a replay source."""
        from obspy import read
        st = read(str(mseed_path))
        st.detrend("demean").filter("bandpass", freqmin=1.0, freqmax=45.0)
        st.resample(sample_rate)

        station = f"{st[0].stats.network}.{st[0].stats.station}"
        channels = []
        for tr in sorted(st, key=lambda t: t.stats.channel):
            channels.append(tr.data.astype(np.float32))

        while len(channels) < 3:
            channels.append(np.zeros_like(channels[0]))

        n_min = min(len(ch) for ch in channels[:3])
        data = np.array([ch[:n_min] for ch in channels[:3]])

        return cls(data, sample_rate, chunk_samples, station)
