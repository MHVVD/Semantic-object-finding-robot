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

"""Unit tests for the spoken command grammar."""

import pytest
from semantic_nav_commander.voice_commands import build_grammar, parse_command

NAMES = ['fridge', 'refrigerator', 'dining table', 'tv', 'couch']


def test_grammar_contains_commands_names_and_unknown():
    g = build_grammar(NAMES)
    assert 'go to the fridge' in g and 'take me to the dining table' in g
    assert 'stop' in g and 'couch' in g
    assert g[-1] == '[unk]' and len(g) == len(set(g))


@pytest.mark.parametrize('text,expected', [
    ('go to the fridge', ('goto', 'fridge')),
    ('take me to the dining table', ('goto', 'dining table')),
    ('find tv', ('goto', 'tv')),
    ('Navigate To The Couch', ('goto', 'couch')),
    ('fridge', ('goto', 'fridge')),
    ('stop', ('cancel', None)),
    ('cancel', ('cancel', None)),
    ('go to the [unk]', None),
    ('[unk]', None),
    ('', None),
    ('go to the garage', None),
])
def test_parse_command(text, expected):
    assert parse_command(text, NAMES) == expected


def test_every_grammar_sentence_parses():
    for sentence in build_grammar(NAMES)[:-1]:
        assert parse_command(sentence, NAMES) is not None
