#!/bin/bash
# Resume / finish the CT paper-profile run, refresh the site, then announce the GPU is free.
# Safe to re-run any time: finished folds are skipped, an interrupted fold resumes from its last
# 10-epoch checkpoint, and a finished run (results/imaging_ct_paper.json) skips training entirely.
#   bash scripts/resume_ct.sh            (from the repo root, Git Bash on Windows)
cd "$(dirname "$0")/.." || exit 1
L=logs/pipeline.log
if [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe; else PY=.venv/bin/python; fi
log() { echo "$(date '+%F %T') [resume] $*" | tee -a "$L"; }

"$PY" -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)" || { log "FAILED: CUDA not available (run python setup_env.py)"; exit 1; }
if [ ! -f results/imaging_ct_paper.json ]; then
  for try in 1 2 3 4 5; do
    log "CT --profile paper start (attempt $try)"
    (cd src && "../$PY" -u imaging.py --modality ct --profile paper) >> logs/ct_paper.log 2>&1
    rc=$?; log "CT --profile paper exit $rc"
    [ $rc = 0 ] && break
    tail -5 logs/ct_paper.log >> "$L"; sleep 60
  done
  [ $rc = 0 ] || { log "FAILED: CT run"; exit 1; }
fi
(cd src && "../$PY" export_site.py) >> "$L" 2>&1 && log "export_site done" || log "FAILED: export_site"
log "pipeline done - GPU free"
