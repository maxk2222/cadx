# Host/VM resource detection for docker-compose cpu/mem caps.
# Source this file; host_detect_env writes CADX_* values to stdout and
# host_detect_merge_env merges them into a .env file (existing keys win).
#
# Detection prefers the Docker VM budget (docker info) over host
# hardware: on Docker Desktop the compose caps count against the VM,
# not the physical machine — sysctl/nproc would overestimate.

_host_cpus=0
_host_mem_bytes=0

host_detect() {
  # populates _host_cpus / _host_mem_bytes
  local info
  info=$(docker info --format '{{.NCPU}} {{.MemTotal}}' 2>/dev/null) || info=""
  if [ -n "$info" ]; then
    _host_cpus=${info%% *}
    _host_mem_bytes=${info##* }
  fi
  if [ "${_host_cpus:-0}" -lt 1 ] 2>/dev/null; then
    if command -v nproc >/dev/null 2>&1; then
      _host_cpus=$(nproc)
    else
      _host_cpus=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
    fi
  fi
  if [ "${_host_mem_bytes:-0}" -lt 1 ] 2>/dev/null; then
    if [ -r /proc/meminfo ]; then
      _host_mem_bytes=$(awk '/MemTotal/ {print $2 * 1024}' /proc/meminfo)
    else
      _host_mem_bytes=$(sysctl -n hw.memsize 2>/dev/null || echo 8589934592)
    fi
  fi
}

host_detect_env() {
  # echoes the CADX_* resource assignments
  host_detect
  local total=${_host_cpus:-4}
  local mem_mb=$(( ${_host_mem_bytes:-8589934592} / 1048576 ))
  local reserve=$(( total / 4 )); [ "$reserve" -lt 2 ] && reserve=2
  local exec_cpus=$(( total - reserve )); [ "$exec_cpus" -lt 2 ] && exec_cpus=2
  local worker_cpus=$(( total / 4 )); [ "$worker_cpus" -lt 2 ] && worker_cpus=2
  [ "$worker_cpus" -gt 4 ] && worker_cpus=4
  local exec_mem=$(( mem_mb * 6 / 10 ))
  [ "$exec_mem" -lt 4096 ] && exec_mem=4096
  [ "$exec_mem" -gt 32768 ] && exec_mem=32768
  local worker_mem=$(( mem_mb * 35 / 100 ))
  [ "$worker_mem" -lt 4096 ] && worker_mem=4096
  [ "$worker_mem" -gt 16384 ] && worker_mem=16384
  printf 'CADX_EXECUTOR_CPUS=%s\nCADX_EXECUTOR_MEM=%sg\n' "$exec_cpus" "$((exec_mem / 1024))"
  printf 'CADX_WORKER_CPUS=%s\nCADX_WORKER_MEM=%sg\n' "$worker_cpus" "$((worker_mem / 1024))"
}

host_detect_merge_env() {
  # host_detect_merge_env <env-file>: writes keys absent from the file.
  # Existing entries (user overrides) are never touched.
  local file=$1 line key
  touch "$file" 2>/dev/null || return 0
  host_detect_env | while IFS= read -r line; do
    key=${line%%=*}
    grep -q "^${key}=" "$file" 2>/dev/null || printf '%s\n' "$line" >> "$file"
  done
}
