"""Read-only RM75/Cine LeRobot v3 metadata, sharded rows and timed RGB."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch

COLOR_CAMERA = 'observation.images.color'


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contained_path(root: Path, relative: str) -> Path:
    """Resolve a relative source/cache path without escaping its declared root."""
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError('Expected a relative path within the declared root')
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Path escapes its declared root')
    return path


class CineV3Repository:
    """Numeric episode IDs are local to this physical repository root."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        info_path, tasks_path = self.root/'meta/info.json', self.root/'meta/tasks.parquet'
        self.info = json.loads(info_path.read_text(encoding='utf-8'))
        if not str(self.info.get('codebase_version', '')).startswith('v3.') or self.info.get('fps') != 30:
            raise ValueError('Cine requires LeRobot v3 metadata at 30 FPS')
        features = self.info.get('features', {})
        for key in ('action', 'observation.state'):
            if features.get(key, {}).get('shape') != [7]:
                raise ValueError(f'Cine requires seven raw coordinates for {key}')
        video = features.get(COLOR_CAMERA, {})
        video_infos = [video[key] for key in ('info', 'video_info') if key in video]
        if video.get('dtype') != 'video' or not video_infos or any(
            not isinstance(info, dict) or info.get('video.fps') != 30 for info in video_infos
        ):
            raise ValueError('Cine requires one true 30-FPS color video feature')
        shape = video.get('shape')
        if not isinstance(shape, list) or len(shape) != 3 or any(type(v) is not int or v <= 0 for v in shape):
            raise ValueError('Cine color shape must describe positive RGB dimensions')
        if shape[-1] == 3:
            self.rgb_size = tuple(shape[:2])
        elif shape[0] == 3:
            self.rgb_size = tuple(shape[1:])
        else:
            raise ValueError('Cine color video must have exactly three RGB channels')
        names = features['action'].get('names')
        if isinstance(names, list) and names != [f'joint_{i}' for i in range(1, 8)]:
            raise ValueError('Cine action channel order must be joint_1 through joint_7')
        self.identity = {'root': str(self.root), 'info_sha256': _sha256(info_path),
                         'tasks_sha256': _sha256(tasks_path)}
        task_rows = pq.read_table(tasks_path).to_pylist()
        self.tasks: dict[int, str] = {}
        for row in task_rows:
            prompt_keys = [key for key in ('task', 'task_text', '__index_level_0__')
                           if isinstance(row.get(key), str)]
            if len(prompt_keys) != 1 or type(row.get('task_index')) is not int:
                raise ValueError('tasks.parquet requires an integer task_index and one prompt column')
            index = row['task_index']
            if index in self.tasks or not row[prompt_keys[0]]:
                raise ValueError('Duplicate or empty Cine task')
            self.tasks[index] = row[prompt_keys[0]]
        paths = sorted((self.root/'meta/episodes').rglob('*.parquet'))
        if not paths:
            raise ValueError('Missing v3 episode parquet metadata')
        episode_digest = hashlib.sha256()
        for path in paths:
            episode_digest.update(path.relative_to(self.root).as_posix().encode('utf-8') + b'\0')
            episode_digest.update(bytes.fromhex(_sha256(path)))
        self.identity['episodes_sha256'] = episode_digest.hexdigest()
        records = [row for path in paths for row in pq.read_table(path).to_pylist()]
        self.episode_records = tuple(sorted(records, key=lambda row: row['episode_index']))
        self._episodes = {}
        for row in self.episode_records:
            index, length = row['episode_index'], row['length']
            if type(index) is not int or type(length) is not int or length <= 0 or index in self._episodes:
                raise ValueError('Invalid or duplicate episode metadata')
            if row['dataset_to_index'] - row['dataset_from_index'] != length:
                raise ValueError('Episode global row interval disagrees with length')
            self._episodes[index] = row
        if len(records) != self.info.get('total_episodes'):
            raise ValueError('Episode metadata is incomplete')
        data_files = sorted((self.root/'data').rglob('*.parquet'))
        if not data_files:
            raise ValueError('Missing v3 data parquet shards')
        # A predicate scan handles episodes crossing files and row groups without
        # loading unrelated episodes or imposing a v2 per-episode filename.
        self._data = ds.dataset([str(path) for path in data_files], format='parquet')

    def read_rows(self, episode_index: int, start: int = 0, stop: int | None = None) -> list[dict]:
        record = self._episodes[episode_index]
        stop = record['length'] if stop is None else stop
        if type(start) is not int or type(stop) is not int or not 0 <= start < stop <= record['length']:
            raise ValueError('Requested frame interval crosses episode bounds')
        predicate = ((ds.field('episode_index') == episode_index) & (ds.field('frame_index') >= start)
                     & (ds.field('frame_index') < stop))
        rows = sorted(self._data.to_table(filter=predicate).to_pylist(), key=lambda row: row['frame_index'])
        if [row['frame_index'] for row in rows] != list(range(start, stop)):
            raise ValueError('Cine episode rows are missing, duplicated or non-contiguous')
        for row in rows:
            if 'index' in row and row['index'] != record['dataset_from_index'] + row['frame_index']:
                raise ValueError('Episode/global row indices disagree')
            for key in ('action', 'observation.state'):
                value = np.asarray(row[key], dtype=np.float64)
                if value.shape != (7,) or not np.isfinite(value).all():
                    raise ValueError(f'Invalid finite raw7 field {key}')
            if row.get('task_index') not in self.tasks:
                raise ValueError('Unresolved row task_index')
            if not math.isfinite(row['timestamp']) or abs(row['timestamp'] - row['frame_index']/30.) > .001:
                raise ValueError('Cine timestamps do not match the original 30-FPS episode clock')
        return rows

    def video_interval(self, episode_index: int) -> tuple[Path, float, float]:
        """Resolve the physical color file and episode's half-open time interval."""
        record = self._episodes[episode_index]
        prefix = f'videos/{COLOR_CAMERA}/'
        start, end = float(record[prefix+'from_timestamp']), float(record[prefix+'to_timestamp'])
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            raise ValueError('Invalid episode video timestamp bounds')
        path = contained_path(self.root, self.info['video_path'].format(video_key=COLOR_CAMERA,
            chunk_index=record[prefix+'chunk_index'], file_index=record[prefix+'file_index']))
        return path, start, end

    def read_rgb_window(self, episode_index: int, start: int, count: int) -> torch.Tensor:
        """Decode only the needed time span, adding the v3 shared-file offset."""
        import av
        if type(count) is not int or count <= 0:
            raise ValueError('RGB frame count must be positive')
        rows = self.read_rows(episode_index, start, start + count)
        path, offset, end = self.video_interval(episode_index)
        wanted = np.array([offset + row['timestamp'] for row in rows])
        if wanted[-1] >= end + .001:
            raise ValueError('Requested frames exceed episode video range')
        times, frames = [], []
        tolerance = .5/30
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 1
            if abs(float(stream.average_rate) - 30.) > .01:
                raise ValueError('Decoded color video must retain 30 FPS')
            container.seek(int(wanted[0] / float(stream.time_base)), stream=stream, backward=True)
            for frame in container.decode(stream):
                if frame.pts is None:
                    raise ValueError('Video frame lacks a presentation timestamp')
                stamp = float(frame.pts * frame.time_base)
                if stamp < wanted[0] - tolerance:
                    continue
                times.append(stamp)
                rgb = frame.to_ndarray(format='rgb24')
                if rgb.shape != (*self.rgb_size, 3):
                    raise ValueError('Decoded RGB shape disagrees with source metadata')
                frames.append(rgb)
                if stamp >= wanted[-1] + tolerance:
                    break
        if not times:
            raise ValueError('No color frames decoded for the episode interval')
        indices = [int(np.argmin(np.abs(np.asarray(times) - stamp))) for stamp in wanted]
        if any(abs(times[index] - stamp) > tolerance for index, stamp in zip(indices, wanted, strict=True)):
            raise ValueError('Decoded RGB cannot be aligned to the requested timestamps')
        return torch.from_numpy(np.stack([frames[index] for index in indices])).permute(0, 3, 1, 2).contiguous()


def fit_cine_action_statistics(repo: CineV3Repository, episode_indices: list[int]) -> dict:
    """Fit unchanged joint-command statistics on explicitly selected train episodes."""
    if not episode_indices or len(set(episode_indices)) != len(episode_indices):
        raise ValueError('Statistics require a nonempty unique episode selection')
    count, total, square = 0, np.zeros(7, dtype=np.float64), np.zeros(7, dtype=np.float64)
    for index in sorted(episode_indices):
        values = np.asarray([row['action'] for row in repo.read_rows(index)], dtype=np.float64)
        count += len(values)
        total += values.sum(0)
        square += np.square(values).sum(0)
    mean = total/count
    std = np.sqrt(np.maximum(square/count - np.square(mean), 0.)).clip(min=1e-6)
    return {'source': 'train', 'root': str(repo.root), 'episode_indices': sorted(episode_indices),
            'count': count, 'mean': mean.tolist(), 'std': std.tolist()}


def validate_cine_split_sources(train_repo: CineV3Repository, val_repo: CineV3Repository,
                               train_episode_indices: list[int], val_episode_indices: list[int]) -> None:
    """Reject shared physical video intervals; do not hash/copy whole videos.

    Different nonoverlapping intervals in a shared physical file are allowed.
    Independent copies of identical video content require a separate content
    audit and are not detected by this inode/time guard.
    """
    if (train_repo.root == val_repo.root or train_repo.root in val_repo.root.parents
            or val_repo.root in train_repo.root.parents):
        raise ValueError('Cine train/validation cannot use the same physical root')
    for repo, indices in ((train_repo,train_episode_indices),(val_repo,val_episode_indices)):
        if not indices or len(set(indices)) != len(indices) or any(type(index) is not int or index not in repo._episodes for index in indices):
            raise ValueError('Cine split check requires a nonempty valid unique episode selection')
    intervals = {}
    for index in train_episode_indices:
        path, start, end = train_repo.video_interval(index)
        stat = path.stat()
        intervals.setdefault((stat.st_dev,stat.st_ino), []).append((start,end))
    for index in val_episode_indices:
        path, start, end = val_repo.video_interval(index)
        stat = path.stat()
        if any(max(start,a) < min(end,b) - 1e-9 for a,b in intervals.get((stat.st_dev,stat.st_ino), ())):
            raise ValueError('Train/validation share overlapping physical video intervals')
