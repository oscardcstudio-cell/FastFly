#!/usr/bin/env bash
# A public https link to the fly brain running on THIS machine (needs its GPU), through a Cloudflare quick tunnel.
# Visitors open <link>/stage and pick their microphone. One shared brain: every visitor drives the same fly.
# The link dies with this script, and changes at every run. Needs: winget install Cloudflare.cloudflared
cd "$(dirname "$0")"
PORT=${PORT:-8010}
CLOUDFLARED=${CLOUDFLARED:-"/c/Program Files (x86)/cloudflared/cloudflared.exe"}
if ! netstat -ano | grep -q "127.0.0.1:$PORT .*LISTEN"; then  # no --audio, no --beatgrid: only the visitors' microphones
  .venv/Scripts/python -u app_server.py --data flywire_v783.bin --port "$PORT" & trap "kill $!" EXIT
fi
"$CLOUDFLARED" tunnel --url "http://127.0.0.1:$PORT" --no-autoupdate 2>&1 | grep --line-buffered -o 'https://[a-z0-9-]*\.trycloudflare\.com' | sed -u 's#$#/stage#'
