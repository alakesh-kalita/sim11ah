#!/bin/bash
# Double-click this file in Finder to launch the Dashboard pre-set to
# Military Zone with the 3D view. Uses the project's own .venv directly
# (not "python3" from PATH) so it works the same from Finder, Dock, or a
# terminal, and always matches whatever VS Code has linked via
# .vscode/settings.json.
#
# nohup + disown + backgrounding: Terminal.app's own "close window when
# the shell exits" preference (Terminal > Settings > Profiles > Shell)
# sends SIGHUP to whatever's still running in that window when it closes
# the tab -- without this, that killed the GUI process mid-startup,
# which looked like a permanently blank/white window since the app never
# got far enough to draw anything.
cd "$(dirname "$0")"
nohup .venv/bin/python3 scripts/launch_military_zone_3d.py \
    > /tmp/sim11ah_military_zone.log 2>&1 &
disown
sleep 1
echo "Launched (PID $!). This terminal window can be closed safely."
