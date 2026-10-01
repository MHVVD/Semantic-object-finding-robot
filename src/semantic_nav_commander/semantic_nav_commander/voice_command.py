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
Voice front end: "go to the fridge" -> GoTo service; "stop" -> cancel service.

Role:
    Listen to the microphone (arecord, 16 kHz mono 16-bit PCM), run offline
    speech recognition (Vosk, small English model) restricted to a grammar of
    command sentences built from the object names the robot can be sent to,
    and turn recognised commands into calls to commander_node. Fully offline:
    no audio leaves the machine.

    Audio is read in a background thread (blocking reads); recognised commands
    go through a queue to a timer in the ROS executor, which makes the service
    calls asynchronously -- GoTo only answers on arrival, and "stop" must still
    be heard and sent meanwhile.

Topics / services:
    Publishes   <heard_topic>          std_msgs/String (every recognised sentence)
    Client      <goto_service>         semantic_nav_interfaces/GoTo
    Client      <cancel_service>       std_srvs/Trigger

Parameters (config/params.yaml, section voice_command):
    model_path        (string) Vosk model directory (~ expanded)
    audio_device      (string) ALSA capture device for arecord, e.g. "default"
    input_wav         (string) '' = microphone; else a WAV file to recognise (testing)
    sample_rate       (int)    microphone sample rate (Vosk models: 16000)
    names             (string[]) spoken object names (labels and their aliases)
    goto_service, cancel_service, heard_topic   (string)
    use_sim_time      (bool)
"""

import json
import os
import queue
import subprocess
import threading
import time
import wave

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from semantic_nav_commander.voice_commands import build_grammar, parse_command
from semantic_nav_interfaces.srv import GoTo
from std_msgs.msg import String
from std_srvs.srv import Trigger

PARAMETERS = {
    'model_path': Parameter.Type.STRING,
    'audio_device': Parameter.Type.STRING,
    'input_wav': Parameter.Type.STRING,
    'sample_rate': Parameter.Type.INTEGER,
    'names': Parameter.Type.STRING_ARRAY,
    'goto_service': Parameter.Type.STRING,
    'cancel_service': Parameter.Type.STRING,
    'heard_topic': Parameter.Type.STRING,
}
CHUNK_BYTES = 8000            # 0.25 s of 16 kHz 16-bit mono
SETTLE_S = 0.5                # after the GoTo service appears, before listening


class VoiceCommand(Node):
    """Microphone -> Vosk (grammar) -> GoTo / cancel service calls."""

    def __init__(self):
        super().__init__('voice_command')
        for name, param_type in PARAMETERS.items():
            self.declare_parameter(name, param_type)
        self.p = p = {name: self.get_parameter(name).value for name in PARAMETERS}
        try:
            import vosk
        except ImportError as e:
            raise RuntimeError('Vosk is not installed: pip install --user vosk '
                               '(see README)') from e
        vosk.SetLogLevel(-1)
        model_path = os.path.expanduser(p['model_path'])
        if not os.path.isdir(model_path):
            raise RuntimeError(f'Vosk model not found at {model_path} (see README)')
        self.model = vosk.Model(model_path)
        self.vosk = vosk
        self.names = [n.lower() for n in p['names']]
        self.grammar = json.dumps(build_grammar(self.names))
        self.heard_pub = self.create_publisher(String, p['heard_topic'], 10)
        self.goto_client = self.create_client(GoTo, p['goto_service'])
        self.cancel_client = self.create_client(Trigger, p['cancel_service'])
        self.commands = queue.Queue()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.listen, daemon=True)
        self.started = False
        self.ready_since = None          # monotonic time the GoTo service was first seen
        self.create_timer(0.1, self.dispatch)

    def start_listening(self):
        self.started = True
        self.thread.start()
        src = self.p['input_wav'] or f'microphone ({self.p["audio_device"]})'
        self.get_logger().info(f'listening on {src}; say e.g. "go to the {self.names[0]}" '
                               'or "stop"')

    # -------------------------------------------------------------- audio thread
    def recognizer(self, rate):
        return self.vosk.KaldiRecognizer(self.model, rate, self.grammar)

    def listen(self):
        try:
            if self.p['input_wav']:
                self.listen_wav(os.path.expanduser(self.p['input_wav']))
            else:
                self.listen_mic()
        except Exception as e:  # noqa: B902 - report anything from the audio thread
            self.get_logger().error(f'audio thread stopped: {e}')

    def listen_mic(self):
        rate = self.p['sample_rate']
        proc = subprocess.Popen(
            ['arecord', '-q', '-D', self.p['audio_device'], '-f', 'S16_LE', '-r', str(rate),
             '-c', '1', '-t', 'raw'], stdout=subprocess.PIPE)
        rec = self.recognizer(rate)
        try:
            while not self.stop_event.is_set():
                data = proc.stdout.read(CHUNK_BYTES)
                if not data:
                    raise RuntimeError('arecord produced no audio (device busy or missing?)')
                if rec.AcceptWaveform(data):
                    self.heard(json.loads(rec.Result()).get('text', ''))
        finally:
            proc.terminate()

    def listen_wav(self, path):
        with wave.open(path, 'rb') as w:
            if w.getnchannels() != 1 or w.getsampwidth() != 2:
                raise RuntimeError(f'{path}: need mono 16-bit PCM')
            rec = self.recognizer(w.getframerate())
            while not self.stop_event.is_set():
                data = w.readframes(4000)
                if not data:
                    break
                if rec.AcceptWaveform(data):
                    self.heard(json.loads(rec.Result()).get('text', ''))
            self.heard(json.loads(rec.FinalResult()).get('text', ''))
        self.get_logger().info(f'finished {path}')

    def heard(self, text):
        if not text or text == '[unk]':
            return
        command = parse_command(text, self.names)
        self.get_logger().info(f'heard "{text}" -> {command or "not a command"}')
        self.heard_pub.publish(String(data=text))
        if command:
            self.commands.put(command)

    # -------------------------------------------------------------- ROS side
    def dispatch(self):
        # Start listening only once the GoTo service is discovered, plus a short
        # settle time: a reply sent before discovery has fully completed is lost
        # (same race as in go_to.py), and a WAV file is recognised in < 1 s.
        if not self.started:
            if not self.goto_client.service_is_ready():
                self.get_logger().info('waiting for the GoTo service ...',
                                       throttle_duration_sec=10.0)
                return
            self.ready_since = self.ready_since or time.monotonic()
            if time.monotonic() - self.ready_since >= SETTLE_S:
                self.start_listening()
            return
        while not self.commands.empty():
            kind, name = self.commands.get()
            if kind == 'cancel':
                if self.cancel_client.service_is_ready():
                    self.cancel_client.call_async(Trigger.Request()).add_done_callback(
                        lambda f: self.get_logger().info(f'stop: {f.result().message}'))
                else:
                    self.get_logger().warn('cancel service not available')
            elif self.goto_client.service_is_ready():
                self.get_logger().info(f'sending the robot to the {name}')
                self.goto_client.call_async(GoTo.Request(label=name)).add_done_callback(
                    lambda f, n=name: self.get_logger().info(
                        f'{n}: {"OK" if f.result().success else "FAILED"} - '
                        f'{f.result().message}'))
            else:
                self.get_logger().warn('GoTo service not available -- is commander_node up?')

    def destroy_node(self):
        self.stop_event.set()
        super().destroy_node()


def main(args=None):
    try:
        rclpy.init(args=args)
        node = VoiceCommand()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass


if __name__ == '__main__':
    main()
