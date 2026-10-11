#!/bin/bash
# Queued after the CT paper run: X-ray --profile paper, then blood --profile full, then export_site.
# Waits for scripts/resume_ct.sh to log "[resume] pipeline done - GPU free" (the GPU only fits one job).
# Safe to re-run any time: finished X-ray folds are skipped, an interrupted fold resumes from its last
# per-epoch checkpoint, a finished X-ray run (results/imaging_xray_paper.json) skips training, and a
# finished blood run (marker results/blood_full.done) is skipped.
#   bash scripts/resume_xray_blood.sh [repo_root]    (Git Bash on Windows; default: this script's repo)
HERE="$(cd "$(dirname "$0")" && (pwd -W 2>/dev/null || pwd))"
cd "${1:-$HERE/..}" || exit 1
L=logs/pipeline.log
if [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe; else PY=.venv/bin/python; fi
log() { echo "$(date '+%F %T') [xb] $*" | tee -a "$L"; }

# 1. wait for CT to free the GPU
if [ ! -f results/imaging_ct_paper.json ] || ! grep -q "\[resume\] pipeline done - GPU free" "$L"; then
  log "waiting for CT paper run to finish"
  until grep -q "\[resume\] pipeline done - GPU free" "$L"; do
    grep -q "\[resume\] FAILED" "$L" && [ ! -f results/imaging_ct_paper.json ] && { log "FAILED: CT run failed, not starting X-ray"; exit 1; }
    sleep 60
  done
fi
"$PY" -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)" || { log "FAILED: CUDA not available (run python setup_env.py)"; exit 1; }

# 2. X-ray, exact paper protocol
if [ ! -f results/imaging_xray_paper.json ]; then
  for try in 1 2 3 4 5; do
    log "X-ray --profile paper start (attempt $try)"
    # paper_run.py = imaging.py --profile paper with a checkpoint every epoch (fsync'd, previous kept as .bak)
    "$PY" -u "$HERE/paper_run.py" --repo "$(pwd -W 2>/dev/null || pwd)" --modality xray >> logs/xray_paper.log 2>&1
    rc=$?; log "X-ray --profile paper exit $rc"
    [ $rc = 0 ] && break
    tail -5 logs/xray_paper.log >> "$L"; sleep 60
  done
  [ $rc = 0 ] || { log "FAILED: X-ray run"; exit 1; }
fi
(cd src && "../$PY" export_site.py) >> "$L" 2>&1 && log "export_site done (after X-ray)" || log "FAILED: export_site"

# 3. blood, full profile (overwrites results/blood.json, so keep the previous one)
# scripts/blood_parallel.sh may already be running it on the CPU: wait for that instead of starting a second one
if [ -f results/blood_full.running ] && [ ! -f results/blood_full.done ]; then
  log "blood --profile full already running in parallel - waiting for it"
  while [ -f results/blood_full.running ] && [ ! -f results/blood_full.done ]; do sleep 60; done
fi
if [ ! -f results/blood_full.done ]; then
  [ -f results/blood.json ] && [ ! -f results/blood.pre_full.backup.json ] && cp results/blood.json results/blood.pre_full.backup.json
  for try in 1 2 3; do
    log "blood --profile full start (attempt $try)"
    # blood.py via blood_run.py, which keeps torch out of the process (torch + sklearn native code crash here)
    "$PY" -u "$HERE/blood_run.py" --repo "$(pwd -W 2>/dev/null || pwd)" --profile full >> logs/blood_full.log 2>&1
    rc=$?; log "blood --profile full exit $rc"
    [ $rc = 0 ] && break
    tail -5 logs/blood_full.log >> "$L"; sleep 60
  done
  [ $rc = 0 ] && date '+%F %T' > results/blood_full.done || log "FAILED: blood run (continuing to export_site)"
fi

# 4. refresh the site data
(cd src && "../$PY" export_site.py) >> "$L" 2>&1 && log "export_site done" || log "FAILED: export_site"
log "pipeline done - GPU free"
