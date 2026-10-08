#!/bin/sh
# Start as root only long enough to make the data volume writable, then drop to the app user.
set -e
DATA="${NEEDLE_DATA_DIR:-/data}"
mkdir -p "$DATA"
if [ "$(id -u)" = "0" ]; then
  chown -R needle:needle "$DATA"
  exec setpriv --reuid=needle --regid=needle --init-groups "$0" "$@"
fi
exec "$@"
