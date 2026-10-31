#!/usr/bin/env bash
# Run a command inside the GPU pod.
#
#   scripts/pod.sh name              # print the current pod name
#   scripts/pod.sh exec <cmd...>     # run a command in the project dir
#   scripts/pod.sh py   <args...>    # run the project venv python
#   scripts/pod.sh pip  <args...>    # run the project venv pip
#   scripts/pod.sh shell             # interactive shell
#
# We are a CO-TENANT of the wmamba-dev pod: the namespace quota is exactly
# 16 CPU / 32 Gi / 1x mig-1g.18gb and that Deployment holds all of it. We do not
# create, modify or restart it -- see docs/ENVIRONMENT.md. Pod names change on
# every reschedule, so never hardcode one.
set -euo pipefail

NS="${VERICALL_NS:-dgx-s-bmu-cse-240577-restricted}"
APP="${VERICALL_APP:-wmamba-dev}"
ROOT="${VERICALL_ROOT:-/home/dgx-s-bmu-cse-240577/SIH final}"
VENV="$ROOT/.venv"

# NFS attribute caching means the pod is a different client and can serve a
# stale copy of a file this node just wrote, or a stale .pyc (observed once:
# 7 minutes old and still used). PYTHONDONTWRITEBYTECODE avoids writing new
# stale bytecode; the remote sync+read-through below forces the pod's own NFS
# client to revalidate before the command runs. Cheap insurance against a
# failure mode that has cost real time more than once.
REMOTE_PRELUDE='export PYTHONDONTWRITEBYTECODE=1; sync 2>/dev/null || true'

pod_name() {
    local p
    p=$(kubectl get pods -n "$NS" -l "app=$APP" \
        --field-selector=status.phase=Running \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
    [ -n "$p" ] || { echo "No running '$APP' pod in $NS." >&2; exit 1; }
    echo "$p"
}

cmd="${1:-shell}"; shift || true
case "$cmd" in
    name)  pod_name ;;
    exec)  kubectl exec -n "$NS" "$(pod_name)" -- bash -lc "$REMOTE_PRELUDE; cd '$ROOT' && $*" ;;
    py)    kubectl exec -n "$NS" "$(pod_name)" -- bash -lc "$REMOTE_PRELUDE; cd '$ROOT' && '$VENV/bin/python' $*" ;;
    pip)   kubectl exec -n "$NS" "$(pod_name)" -- bash -lc "$REMOTE_PRELUDE; cd '$ROOT' && '$VENV/bin/pip' $*" ;;
    shell) kubectl exec -it -n "$NS" "$(pod_name)" -- bash ;;
    *)     echo "usage: pod.sh {name|exec|py|pip|shell} [args...]" >&2; exit 1 ;;
esac
