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

"""Unit tests for semantic-map data association."""

import math

import numpy as np
import pytest
from semantic_nav_mapping.association import (assign_greedy, assign_hungarian, Observation,
                                              parse_class_radius, SemanticMap)

INF = math.inf


def obs(label, x, y, z=0.4, conf=0.8):
    return Observation(label, (x, y, z), conf)


def make_map(**kw):
    args = {'association_radius': 0.75, 'min_observations': 3, 'tentative_timeout_s': 10.0}
    args.update(kw)
    return SemanticMap(**args)


# ---------------------------------------------------------------- the three required cases
def test_two_chairs_close_together_stay_separate():
    # Chairs 0.6 m apart (closer than the 0.75 m radius), seen side by side in the
    # same frames with +-5 cm noise. One-to-one assignment within a frame stops both
    # boxes from being merged into one track.
    m = make_map()
    rng = np.random.default_rng(0)
    for k in range(10):
        n = rng.normal(0, 0.05, 4)
        m.update([obs('chair', 1.0 + n[0], 2.0 + n[1]), obs('chair', 1.6 + n[2], 2.0 + n[3])],
                 stamp=float(k))
    chairs = m.confirmed('chair')
    assert len(chairs) == 2
    xs = sorted(c.position[0] for c in chairs)
    assert xs == pytest.approx([1.0, 1.6], abs=0.05)
    assert all(c.count == 10 for c in chairs)


def test_close_chairs_merge_if_never_seen_together():
    # The limitation, documented: seen one at a time, the second chair (0.6 m away)
    # falls inside the first one's gate and is absorbed. A smaller radius fixes this
    # case but splits noisy observations of a single object (see the M5 report).
    m = make_map()
    for k in range(5):
        m.update([obs('chair', 1.0, 2.0)], stamp=float(k))
    for k in range(5, 10):
        m.update([obs('chair', 1.6, 2.0)], stamp=float(k))
    assert len(m.confirmed('chair')) == 1
    m_small = make_map(association_radius=0.4)
    for k in range(5):
        m_small.update([obs('chair', 1.0, 2.0)], stamp=float(k))
    for k in range(5, 10):
        m_small.update([obs('chair', 1.6, 2.0)], stamp=float(k))
    assert len(m_small.confirmed('chair')) == 2


def test_false_positive_seen_once_is_never_confirmed():
    m = make_map()
    m.update([obs('tv', 3.0, 1.0)], stamp=0.0)            # one stray "tv"
    for k in range(1, 6):
        m.update([obs('chair', 0.0, 0.0)], stamp=float(k))
    assert [t.label for t in m.confirmed()] == ['chair']
    assert any(t.label == 'tv' and not t.confirmed for t in m.tracks.values())
    m.update([], stamp=20.0)                              # > tentative_timeout_s later
    assert all(t.label != 'tv' for t in m.tracks.values())  # pruned


def test_same_object_from_different_angles_is_one_object_near_its_centre():
    # A depth camera measures the near surface, so each view is biased ~0.2 m toward
    # the camera. Viewed from 8 directions around the object, all observations land
    # within the gate of the running mean, and the biases cancel.
    m = make_map()
    centre = np.array([2.0, -1.0])
    for k in range(8):
        a = k * math.pi / 4                               # camera direction from object
        p = centre + 0.2 * np.array([math.cos(a), math.sin(a)])
        m.update([obs('potted plant', *p)], stamp=float(k))
    plants = m.confirmed('potted plant')
    assert len(plants) == 1 and plants[0].count == 8
    assert plants[0].position[:2] == pytest.approx(centre, abs=1e-9)


# ---------------------------------------------------------------- rules
def test_running_mean_and_counts():
    m = make_map(min_observations=2)
    m.update([obs('tv', 0.0, 0.0, conf=0.6)], stamp=1.0)
    m.update([obs('tv', 0.3, 0.0, conf=0.8)], stamp=2.0)
    m.update([obs('tv', 0.6, 0.3, conf=1.0)], stamp=3.5)
    (tv,) = m.confirmed()
    assert tv.position[:2] == pytest.approx([0.3, 0.1])
    assert tv.confidence == pytest.approx(0.8)
    assert (tv.count, tv.first_seen, tv.last_seen) == (3, 1.0, 3.5)


def test_confirmation_threshold():
    m = make_map(min_observations=3)
    for k in range(2):
        m.update([obs('sink', 0.0, 0.0)], stamp=float(k))
    assert m.confirmed() == []
    m.update([obs('sink', 0.0, 0.0)], stamp=2.0)
    assert len(m.confirmed()) == 1


def test_min_observations_one_confirms_immediately():
    m = make_map(min_observations=1)
    m.update([obs('bed', 0.0, 0.0)], stamp=0.0)
    assert len(m.confirmed()) == 1


def test_different_labels_never_merge():
    m = make_map(min_observations=1)
    m.update([obs('chair', 0.0, 0.0)], stamp=0.0)
    m.update([obs('couch', 0.1, 0.0)], stamp=1.0)
    assert sorted(t.label for t in m.confirmed()) == ['chair', 'couch']


def test_outside_gate_starts_new_track():
    m = make_map(min_observations=1)
    m.update([obs('chair', 0.0, 0.0)], stamp=0.0)
    m.update([obs('chair', 0.8, 0.0)], stamp=1.0)
    assert len(m.confirmed('chair')) == 2


def test_class_radius_override_for_large_objects():
    # The two ends of a 2.8 m bed are 1.4 m apart: split with the default radius,
    # one object with a bed-sized radius.
    for radius, expected in ((None, 2), (2.0, 1)):
        m = make_map(min_observations=1, class_radius={'bed': radius} if radius else None)
        m.update([obs('bed', 0.0, 0.0)], stamp=0.0)
        m.update([obs('bed', 1.4, 0.0)], stamp=1.0)
        assert len(m.confirmed('bed')) == expected


def test_height_is_ignored_for_gating():
    m = make_map(min_observations=1)
    m.update([obs('tv', 0.0, 0.0, z=0.2)], stamp=0.0)
    m.update([obs('tv', 0.0, 0.0, z=1.5)], stamp=1.0)
    assert len(m.confirmed()) == 1


def test_confirmed_tracks_are_not_pruned():
    m = make_map(min_observations=2, tentative_timeout_s=1.0)
    m.update([obs('toilet', 0.0, 0.0)], stamp=0.0)
    m.update([obs('toilet', 0.0, 0.0)], stamp=0.5)
    m.update([], stamp=1000.0)
    assert len(m.confirmed()) == 1


def test_update_returns_track_ids():
    m = make_map()
    ids1 = m.update([obs('chair', 0.0, 0.0), obs('tv', 5.0, 5.0)], stamp=0.0)
    ids2 = m.update([obs('tv', 5.1, 5.0), obs('chair', 0.1, 0.0)], stamp=1.0)
    assert ids2 == ids1[::-1]


# ---------------------------------------------------------------- assignment algorithms
def test_greedy_vs_hungarian_classic_counterexample():
    # Observation A is 0.1 from track 1 and 0.5 from track 2; observation B is 0.2
    # from track 1 and outside the gate of track 2. Greedy takes the closest pair
    # (A-1) and leaves B with nothing; Hungarian finds A-2, B-1 (total 0.7 < inf).
    cost = [[0.1, 0.5],
            [0.2, INF]]
    assert assign_greedy(cost) == [(0, 0)]
    assert sorted(assign_hungarian(cost)) == [(0, 1), (1, 0)]


def test_assignments_respect_gate_and_empty_input():
    assert assign_greedy([[INF]]) == [] and assign_hungarian([[INF]]) == []
    assert assign_hungarian([]) == []


def test_hungarian_method_in_map():
    m = make_map(min_observations=1, method='hungarian')
    m.update([obs('chair', 0.0, 0.0), obs('chair', 0.6, 0.0)], stamp=0.0)
    m.update([obs('chair', 0.05, 0.0), obs('chair', 0.55, 0.0)], stamp=1.0)
    assert len(m.confirmed()) == 2


def test_unknown_method_rejected():
    with pytest.raises(ValueError):
        make_map(method='magic')


# ---------------------------------------------------------------- persistence
def test_save_load_round_trip_keeps_only_confirmed():
    m = make_map(min_observations=2)
    for k in range(3):
        m.update([obs('fridge', 4.0, -3.0, z=0.9)], stamp=float(k))
    m.update([obs('tv', 0.0, 0.0)], stamp=3.0)            # tentative: not saved
    data = m.to_dict()
    assert [o['label'] for o in data['objects']] == ['fridge']
    m2 = make_map()
    assert m2.load_dict(data) == 1
    (f,) = m2.confirmed()
    assert f.position == pytest.approx([4.0, -3.0, 0.9]) and f.count == 3
    assert m2.next_id == f.track_id + 1
    m2.update([obs('fridge', 4.1, -3.0)], stamp=10.0)     # loaded objects keep growing
    assert m2.confirmed()[0].count == 4


def test_parse_class_radius():
    assert parse_class_radius(['bed=2.0', ' couch = 1.5 ']) == {'bed': 2.0, 'couch': 1.5}
    assert parse_class_radius(['dining table=1.2']) == {'dining table': 1.2}
    assert parse_class_radius(['']) == {} and parse_class_radius([]) == {}
    with pytest.raises(ValueError):
        parse_class_radius(['bed'])


def test_failed_load_keeps_existing_map():
    m = make_map(min_observations=1)
    m.update([obs('tv', 0.0, 0.0)], stamp=0.0)
    with pytest.raises(KeyError):
        m.load_dict({'objects': [{'id': 1, 'label': 'bed', 'position': [0, 0, 0],
                                  'confidence': 0.9, 'observation_count': 5,
                                  'first_seen': 0.0, 'last_seen': 1.0},
                                 {'label': 'chair'}]})       # second entry is broken
    assert [t.label for t in m.confirmed()] == ['tv']
