#!/usr/bin/env bash
# Jetson: run the instrumented arm client against the Desktop policy server.
# Mirrors ~/lerobot/docs/superpowers/scripts/run_async_client.sh (same site env, same
# robot flags) and adds the watchdog / stop-file / report flags of pi0_arm_client.py.
#
#   ASYNC_SERVER_ADDRESS=192.168.10.1:8080 ASYNC_FPS=1 ./run_pi0_arm_client.sh
#   touch /tmp/pi0_arm_stop        # from another shell: stop the arm client (servoStop + stopScript)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SITE="${LEROBOT_SITE_ENV:-$HOME/lerobot/docs/superpowers/scripts/lerobot_site.env}"
# shellcheck source=/dev/null
source "$SITE"
: "${ASYNC_SERVER_ADDRESS:?Set ASYNC_SERVER_ADDRESS to host:port of policy server}"
: "${ASYNC_FPS:=1}"; : "${ASYNC_CHUNK_SIZE_THRESHOLD:=0.6}"; : "${ASYNC_AGGREGATE_FN:=weighted_average}"
: "${ASYNC_ACTION_WINDOW:=}"; : "${PI0_WATCHDOG_S:=60}"; : "${PI0_MAX_RUN_S:=0}"; : "${PI0_RUN_LABEL:=arm}"
: "${PI0_RUN_OUT:=$HOME/pi0_glue/runs/arm_$(date +%Y%m%d_%H%M%S)}"
ARGS=(
    --server_address="$ASYNC_SERVER_ADDRESS"
    --robot.type=ur_follower_joint --robot.robot_ip="$ROBOT_IP" --robot.id="$ROBOT_ID" --robot.use_gripper=true
    --task="$SINGLE_TASK" --client_device=cpu --fps="$ASYNC_FPS"
    --chunk_size_threshold="$ASYNC_CHUNK_SIZE_THRESHOLD" --aggregate_fn_name="$ASYNC_AGGREGATE_FN"
    --watchdog_s="$PI0_WATCHDOG_S" --max_run_s="$PI0_MAX_RUN_S" --out="$PI0_RUN_OUT" --label="$PI0_RUN_LABEL"
)
[ -n "${ASYNC_ACTION_WINDOW:-}" ] && ARGS+=(--action_window="$ASYNC_ACTION_WINDOW")
[ -n "${GRIPPER_CAL_RANGE:-}" ] && ARGS+=(--robot.gripper_calibration_range="$GRIPPER_CAL_RANGE")
[ -n "${ROBOT_CAMERAS:-}" ] && ARGS+=(--robot.cameras="$ROBOT_CAMERAS")
[ -n "${ROBOT_RTDE_FREQUENCY:-}" ] && ARGS+=(--robot.rtde_frequency="$ROBOT_RTDE_FREQUENCY")
[ -n "${ROBOT_SERVO_DT:-}" ] && ARGS+=(--robot.servo_dt="$ROBOT_SERVO_DT")
[ -n "${ROBOT_RTDE_RECEIVE_FREQUENCY:-}" ] && ARGS+=(--robot.rtde_receive_frequency="$ROBOT_RTDE_RECEIVE_FREQUENCY")
echo "+ python $HERE/pi0_arm_client.py ${ARGS[*]} $*" >&2
[ "${DRY_RUN:-false}" = "true" ] && exit 0
exec python "$HERE/pi0_arm_client.py" "${ARGS[@]}" "$@"
