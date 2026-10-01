# Copyright 2026 Mahmud
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Data association for the semantic map: observations -> persistent object tracks. No ROS.

One call to SemanticMap.update() handles all observations of ONE camera frame:

    1. prune   tentative tracks not seen for `tentative_timeout_s` are deleted
               (a false positive seen a couple of times never gets confirmed)
    2. gate    for every observation, candidate tracks = same label AND within
               the association radius (per-class override possible) in x-y
    3. assign  one-to-one within the frame: an object can only produce one box
               per image, so two chairs seen side by side must not both be merged
               into the same track. 'greedy' = global nearest neighbour (closest
               pair first); 'hungarian' = minimum total distance (scipy).
    4. update  matched track: running mean of position (and of confidence),
               count += 1, last_seen = stamp; confirmed once count >= min_observations
    5. spawn   unmatched observation -> new tentative track

Distances are horizontal (x, y): heights from the box centre row are less
reliable (M4) and two objects at the same x-y but different heights are rare.
"""

from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Observation:
    """One single-frame detection in the map frame."""

    label: str
    position: tuple            # (x, y, z) metres
    confidence: float


@dataclass
class Track:
    """A persistent object hypothesis."""

    track_id: int
    label: str
    position: np.ndarray       # running mean (x, y, z)
    confidence: float          # running mean of detector scores
    count: int
    first_seen: float          # seconds (stamp of the first observation)
    last_seen: float
    confirmed: bool = False

    def absorb(self, obs, stamp, min_observations):
        """Running-mean update with one observation."""
        self.count += 1
        p = np.asarray(obs.position, dtype=float)
        self.position = self.position + (p - self.position) / self.count
        self.confidence += (obs.confidence - self.confidence) / self.count
        self.last_seen = max(self.last_seen, stamp)
        if self.count >= min_observations:
            self.confirmed = True


def parse_class_radius(items):
    """['bed=2.0', 'couch=1.5'] -> {'bed': 2.0, 'couch': 1.5}; [''] or [] -> {}."""
    out = {}
    for item in items:
        if not item.strip():
            continue
        label, _, value = item.rpartition('=')
        if not label:
            raise ValueError(f'class_radius entry {item!r} is not "label=metres"')
        out[label.strip()] = float(value)
    return out


def xy_distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def assign_greedy(cost):
    """Global nearest neighbour: repeatedly take the cheapest remaining (row, col) pair."""
    pairs = sorted((c, i, j) for i, row in enumerate(cost) for j, c in enumerate(row)
                   if math.isfinite(c))
    used_rows, used_cols, out = set(), set(), []
    for c, i, j in pairs:
        if i not in used_rows and j not in used_cols:
            used_rows.add(i)
            used_cols.add(j)
            out.append((i, j))
    return out


def assign_hungarian(cost):
    """Minimum-total-cost one-to-one assignment; infinite (gated-out) pairs never chosen."""
    from scipy.optimize import linear_sum_assignment
    c = np.asarray(cost, dtype=float)
    if c.size == 0:
        return []
    big = 1e6
    rows, cols = linear_sum_assignment(np.where(np.isfinite(c), c, big))
    return [(int(i), int(j)) for i, j in zip(rows, cols) if c[i, j] < big]


class SemanticMap:
    """Set of object tracks plus the association rules that maintain it."""

    def __init__(self, association_radius, min_observations, tentative_timeout_s,
                 method='greedy', class_radius=None):
        if method not in ('greedy', 'hungarian'):
            raise ValueError(f'unknown association method {method!r}')
        self.radius = float(association_radius)
        self.min_observations = int(min_observations)
        self.tentative_timeout_s = float(tentative_timeout_s)
        self.assign = assign_greedy if method == 'greedy' else assign_hungarian
        self.class_radius = dict(class_radius or {})
        self.tracks = {}
        self.next_id = 1

    def radius_for(self, label):
        return self.class_radius.get(label, self.radius)

    def prune(self, now):
        """Delete tentative tracks that have not been seen for tentative_timeout_s."""
        stale = [tid for tid, t in self.tracks.items()
                 if not t.confirmed and now - t.last_seen > self.tentative_timeout_s]
        for tid in stale:
            del self.tracks[tid]
        return len(stale)

    def update(self, observations, stamp):
        """
        Fuse the observations of one frame; return the track id each one went to.

        Labels are associated independently: only same-label tracks are candidates.
        """
        self.prune(stamp)
        result = [None] * len(observations)
        for label in sorted({o.label for o in observations}):
            obs_idx = [i for i, o in enumerate(observations) if o.label == label]
            track_ids = [tid for tid, t in self.tracks.items() if t.label == label]
            gate = self.radius_for(label)
            cost = [[(d if (d := xy_distance(observations[i].position,
                                             self.tracks[tid].position)) <= gate
                      else math.inf) for tid in track_ids] for i in obs_idx]
            for r, c in self.assign(cost) if track_ids else []:
                tid = track_ids[c]
                self.tracks[tid].absorb(observations[obs_idx[r]], stamp,
                                        self.min_observations)
                result[obs_idx[r]] = tid
            for i in obs_idx:
                if result[i] is None:
                    result[i] = self._spawn(observations[i], stamp)
        return result

    def _spawn(self, obs, stamp):
        tid = self.next_id
        self.next_id += 1
        track = Track(track_id=tid, label=obs.label,
                      position=np.asarray(obs.position, dtype=float),
                      confidence=float(obs.confidence), count=1, first_seen=stamp,
                      last_seen=stamp)
        track.confirmed = self.min_observations <= 1
        self.tracks[tid] = track
        return tid

    def confirmed(self, label_filter=''):
        """Return the confirmed tracks (optionally of one label), ordered by id."""
        return [t for _, t in sorted(self.tracks.items())
                if t.confirmed and (not label_filter or t.label == label_filter)]

    # -------------------------------------------------------------- persistence
    def to_dict(self):
        """Return the confirmed objects as plain data (for YAML); tentative ones are not saved."""
        return {'objects': [{
            'id': t.track_id, 'label': t.label,
            'position': [round(float(v), 4) for v in t.position],
            'confidence': round(float(t.confidence), 4), 'observation_count': t.count,
            'first_seen': round(t.first_seen, 3), 'last_seen': round(t.last_seen, 3),
        } for t in self.confirmed()]}

    def load_dict(self, data):
        """Replace the map with saved objects (all confirmed); return how many."""
        tracks = {}    # built aside: a malformed entry leaves the current map untouched
        for o in data.get('objects') or []:
            t = Track(track_id=int(o['id']), label=str(o['label']),
                      position=np.asarray(o['position'], dtype=float),
                      confidence=float(o['confidence']), count=int(o['observation_count']),
                      first_seen=float(o['first_seen']), last_seen=float(o['last_seen']),
                      confirmed=True)
            tracks[t.track_id] = t
        self.tracks = tracks
        self.next_id = max(self.tracks, default=0) + 1
        return len(self.tracks)
