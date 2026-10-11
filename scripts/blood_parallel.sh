#!/bin/bash
# Run blood.py --profile full on the CPU while the GPU is busy with an imaging run, then refresh the site.
# Marks results/blood_full.running while it runs and results/blood_full.done on success, so
# resume_xray_blood.sh waits for / skips it instead of starting a second copy.
#   bash scripts/blood_parallel.sh [repo_root]    (Git Bash on Windows; start it at below-normal priority)
HERE="$(cd "$(dirname "$0")" && (pwd -W 2>/dev/null || pwd))"
cd "${1:-$HERE/..}" || exit 1
L=logs/pipeline.log
if [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe; else PY=.venv/bin/python; fi
log() { echo "$(date '+%F %T') [blood] $*" | tee -a "$L"; }
[ -f results/blood_full.done ] && { log "blood --profile full already done"; exit 0; }
export CUDA_VISIBLE_DEVICES=-1   # keep the GPU for the imaging run
date '+%F %T' > results/blood_full.running
[ -f results/blood.json ] && [ ! -f results/blood.pre_full.backup.json ] && cp results/blood.json results/blood.pre_full.backup.json
for try in 1 2 3; do
  log "blood --profile full start on CPU (attempt $try)"
  # blood.py via blood_run.py, which keeps torch out of the process (torch + sklearn native code crash here)
  "$PY" -u "$HERE/blood_run.py" --repo "$(pwd -W 2>/dev/null || pwd)" --profile full >> logs/blood_full.log 2>&1
  rc=$?; log "blood --profile full exit $rc"
  [ $rc = 0 ] && break
  tail -5 logs/blood_full.log >> "$L"; sleep 60
done
if [ $rc = 0 ]; then
  date '+%F %T' > results/blood_full.done
  (cd src && "../$PY" export_site.py) >> "$L" 2>&1 && log "export_site done (after blood)" || log "FAILED: export_site"
else
  log "FAILED: blood run"
fi
rm -f results/blood_full.running
