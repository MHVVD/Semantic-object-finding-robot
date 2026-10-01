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
Command-line front end: send the robot to an object from the semantic map.

    ros2 run semantic_nav_commander go_to fridge
    ros2 run semantic_nav_commander go_to dining table
    ros2 run semantic_nav_commander go_to --list          # what the robot knows
    ros2 run semantic_nav_commander go_to --cancel        # stop the current goal

Calls commander_node's GoTo service and waits for the answer (which arrives
when the robot has arrived or failed). Ctrl-C while waiting cancels the
navigation. Exit code 0 on success, 1 on failure.
"""

import argparse
from collections import Counter
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from semantic_nav_interfaces.srv import GoTo, ListObjects
from std_srvs.srv import Trigger

DISCOVERY_SETTLE_S = 0.5


def call(node, client, request, timeout=None):
    if not client.wait_for_service(timeout_sec=5.0):
        print(f'service {client.srv_name} is not available -- is the stack running?')
        sys.exit(1)
    # wait_for_service() only proves the server's REQUEST reader is discovered; our
    # RESPONSE reader may not be matched on the server side yet, and a reply sent
    # before that is silently lost (the call then never returns). Give discovery a
    # moment to finish before the one-shot call.
    end = time.monotonic() + DISCOVERY_SETTLE_S
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.05)
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout)
    return future.result()


def main():
    parser = argparse.ArgumentParser(description='Send the robot to an object by name.')
    parser.add_argument('label', nargs='*', help='object name, e.g. fridge, "dining table"')
    parser.add_argument('--list', action='store_true', help='list known objects')
    parser.add_argument('--cancel', action='store_true', help='cancel the current goal')
    parser.add_argument('--service', default='/semantic_nav/go_to')
    parser.add_argument('--list-service', default='/semantic_nav/list_objects')
    parser.add_argument('--cancel-service', default='/semantic_nav/cancel')
    args, ros_args = parser.parse_known_args()
    if not (args.label or args.list or args.cancel):
        parser.error('give an object name, --list or --cancel')

    # No rclpy SIGINT handler: Ctrl-C must raise KeyboardInterrupt here (so we can
    # still call the cancel service) instead of shutting the context down.
    rclpy.init(args=ros_args, signal_handler_options=SignalHandlerOptions.NO)
    node = Node('go_to_cli')
    code = 0
    try:
        if args.list:
            objs = call(node, node.create_client(ListObjects, args.list_service),
                        ListObjects.Request(), 10.0).objects
            for name, n in sorted(Counter(o.label for o in objs).items()):
                print(f'{name:14s} x{n}')
            for o in objs:
                print(f'  #{o.id:<3d} {o.label:13s} ({o.position.x:6.2f}, {o.position.y:6.2f})'
                      f'  seen {o.observation_count} times')
        elif args.cancel:
            res = call(node, node.create_client(Trigger, args.cancel_service), Trigger.Request(),
                       10.0)
            print(res.message)
            code = 0 if res.success else 1
        else:
            label = ' '.join(args.label)
            print(f'going to the {label} ...')
            try:
                res = call(node, node.create_client(GoTo, args.service), GoTo.Request(label=label))
            except KeyboardInterrupt:
                print('\ninterrupted: canceling the goal')
                call(node, node.create_client(Trigger, args.cancel_service), Trigger.Request(),
                     10.0)
                res = None
            if res is None:
                code = 1
            else:
                print(('OK: ' if res.success else 'FAILED: ') + res.message)
                code = 0 if res.success else 1
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    sys.exit(code)


if __name__ == '__main__':
    main()
