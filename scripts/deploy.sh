#!/usr/bin/env bash
set -euo pipefail
expected_sha="${1:?Expected commit SHA required}"
[[ "$expected_sha" =~ ^[0-9a-f]{40}$ ]]
base=/srv/trust.tgo.dev
exec 9>"$base/deploy.lock"
flock -w 600 9
old_release=$(readlink -f "$base/current")
old_runtime=$(readlink -f "$base/venv")
test -L "$base/current"
test -L "$base/venv"
release=$(mktemp -d "$base/releases/${expected_sha:0:12}-XXXXXX")
activated=false
stage_pid=''
stage_socket="$release/stage.sock"
link_to() {
  ln -s "$1" "$2.next"
  mv -Tf "$2.next" "$2"
}
cleanup() {
  result=$?
  trap - EXIT
  if [[ -n "$stage_pid" ]]; then
    kill "$stage_pid" 2>/dev/null || true
    wait "$stage_pid" 2>/dev/null || true
  fi
  if [[ "$result" != 0 && "$activated" == true ]]; then
    link_to "$old_release" "$base/current"
    link_to "$old_runtime" "$base/venv"
    sudo -n systemctl restart trust.service
    sudo -n systemctl is-active --quiet trust.service
    curl --fail --silent --show-error --retry 10 --retry-all-errors --retry-delay 1 \
      --max-time 15 --unix-socket "$base/current/trust.sock" \
      -H 'Host: trust.tgo.dev' -H 'X-Forwarded-Proto: https' http://localhost/ > /dev/null
    printf 'Activation failed; restored %s\n' "$old_release" >&2
  fi
  exit "$result"
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
git clone --quiet https://github.com/Kiguli/TRUST.git "$release"
test "$(git -C "$release" rev-parse origin/main)" = "$expected_sha"
git -C "$release" checkout --quiet --detach "$expected_sha"
# Keep the deployment checkout on main for source inspection.
git -C "$release" checkout --quiet -B main "$expected_sha"
git -C "$release" branch --set-upstream-to=origin/main main
cp -p "$base/current/.env" "$release/.env"
chmod 600 "$release/.env"
ln -s "$base/shared/uploads" "$release/storage/uploads"
/home/jamie/.pyenv/versions/3.12.10/bin/python3.12 -m venv "$release/.venv"
# shellcheck disable=SC1091
source "$release/.venv/bin/activate"
cd "$release"
"$base/tools/poetry/bin/poetry" check --lock
"$base/tools/poetry/bin/poetry" install --only main --no-interaction
export PATH=/home/jamie/.nvm/versions/node/v22.12.0/bin:$PATH
cd vite
npm ci
npm run build
cd "$release"
SENTRY_DSN='' "$release/.venv/bin/gunicorn" --workers 1 --bind "unix:$stage_socket" wsgi:app > "$release/stage.log" 2>&1 &
stage_pid=$!
curl --fail --silent --show-error --retry 10 --retry-all-errors --retry-delay 1 \
  --max-time 15 --unix-socket "$stage_socket" \
  -H 'Host: trust.tgo.dev' -H 'X-Forwarded-Proto: https' http://localhost/ > /dev/null
kill "$stage_pid"
wait "$stage_pid" || true
stage_pid=''
activated=true
link_to "$release" "$base/current"
link_to "$release/.venv" "$base/venv"
sudo -n systemctl restart trust.service
sudo -n systemctl is-active --quiet trust.service
curl --fail --silent --show-error --retry 10 --retry-all-errors --retry-delay 1 \
  --max-time 15 --unix-socket "$base/current/trust.sock" \
  -H 'Host: trust.tgo.dev' -H 'X-Forwarded-Proto: https' http://localhost/ > /dev/null
test "$(git -C "$base/current" rev-parse HEAD)" = "$expected_sha"
printf 'Deployed %s\n' "$expected_sha"
