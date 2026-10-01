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
Spoken command grammar and parsing for voice_command. No ROS, no audio.

Accepted sentences (case-insensitive, "the"/"a" optional):
    go to the fridge | take me to the couch | find the toilet |
    navigate to the tv | drive to the bed | fridge
    stop | cancel

The speech recogniser is restricted to exactly these sentences (a Vosk
"grammar"): with ~100 possible sentences instead of an open vocabulary, the
small 40 MB model is far more accurate, and anything else comes back as
"[unk]" instead of a wrong guess.
"""

import re

VERBS = ('go to', 'take me to', 'find', 'navigate to', 'drive to')
STOP_WORDS = ('stop', 'cancel')


def build_grammar(names):
    """Every accepted sentence for the given object names, plus "[unk]"."""
    phrases = set(STOP_WORDS)
    for name in names:
        phrases.add(name)
        for verb in VERBS:
            phrases.add(f'{verb} the {name}')
            phrases.add(f'{verb} {name}')
    return sorted(phrases) + ['[unk]']


def parse_command(text, names):
    """
    Recognised text -> ('goto', name) | ('cancel', None) | None.

    `names` are the spoken names in the grammar (labels and aliases); the
    returned name is still a spoken name -- the commander resolves aliases.
    """
    words = re.sub(r'\[unk\]', ' ', text.lower()).split()
    text = ' '.join(words)
    if not text:
        return None
    if text in STOP_WORDS:
        return ('cancel', None)
    for verb in sorted(VERBS, key=len, reverse=True):
        if text.startswith(verb + ' '):
            text = text[len(verb) + 1:]
            break
    text = re.sub(r'^(the|a|an) ', '', text)
    return ('goto', text) if text in names else None
