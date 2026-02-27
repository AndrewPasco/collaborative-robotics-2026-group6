import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/apasco/me326/collaborative-robotics-2026-group6/install/tidybot_mujoco_bridge'
