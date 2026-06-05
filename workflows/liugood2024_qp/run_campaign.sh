#!/usr/bin/env bash
# Memory-aware Liu & Good 2024 QP reproduction campaign.
#
# A single big-species worker peaks ~4.5 GB (the full coverage matrix), so a fixed
# large --jobs OOMs a ~24 GB machine. Instead we tier by catalog size: small catalogs
# run with high parallelism, large catalogs run serially.
#   small  (<140 MB catalog, ~<2 GB/worker)   -> --jobs 6
#   medium (140-269 MB,      ~2-3 GB/worker)  -> --jobs 2
#   large  (>=270 MB,        ~3-4.5 GB/worker)-> --jobs 1 (serial)
# Each tier's verification rows are saved, then merged into verification_summary.csv.
#
# Requires: CPHMM_QP_DATA_DIR, CPHMM_QP_REFERENCE_DIR, CPHMM_QP_GROUND_TRUTH.
# Optional: CPHMM_PY (python interpreter; default `python3`).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${CPHMM_PY:-python3}"
RES=workflows/liugood2024_qp/results
mkdir -p "$RES"
rm -f "$RES"/*.csv

SMALL="Alistipes_shahii_62199 Oscillibacter_sp_60799 Alistipes_finegoldii_56071 \
Dialister_invisus_61905 Eubacterium_siraeum_57634 Bacteroides_ovatus_58035 \
Bacteroides_eggerthii_54457 Phascolarctobacterium_sp_59817 Bacteroides_coprocola_61586 \
Bacteroides_finegoldii_57739 Bacteroidales_bacterium_58650 Alistipes_sp_60764"
MEDIUM="Parabacteroides_distasonis_56985 Bacteroides_uniformis_57318 Bacteroides_caccae_53434 \
Parabacteroides_merdae_56972 Alistipes_onderdonkii_55464 Bacteroides_fragilis_54507 \
Akkermansia_muciniphila_55290 Bacteroides_massiliensis_44749"
LARGE="Bacteroides_vulgatus_57955 Bacteroides_stercoris_56735 Ruminococcus_bromii_62047 \
Eubacterium_rectale_56927 Alistipes_putredinis_61533 Bacteroides_thetaiotaomicron_56941 \
Barnesiella_intestinihominis_62208 Bacteroides_cellulosilyticus_58046 Ruminococcus_bicirculans_59300"

run_tier () { # $1=label  $2=jobs  $3..=species
  local label="$1" jobs="$2"; shift 2
  echo "=== tier ${label} (jobs=${jobs}) starting $(date) ==="
  "$PY" workflows/liugood2024_qp/reproduce.py --species "$@" --jobs "$jobs" \
      --transfer-length iterative
  cp "$RES/verification_summary.csv" "$RES/_summary_${label}.csv"
}

run_tier small 6 $SMALL
run_tier medium 2 $MEDIUM
run_tier large 1 $LARGE

# Merge the three tier summaries into one.
"$PY" - <<'PYEOF'
import pandas as pd, os
res = "workflows/liugood2024_qp/results"
parts = [pd.read_csv(f"{res}/_summary_{t}.csv")
         for t in ("small", "medium", "large") if os.path.exists(f"{res}/_summary_{t}.csv")]
allsum = pd.concat(parts, ignore_index=True)
allsum.to_csv(f"{res}/verification_summary.csv", index=False)
print(f"\n=== merged verification_summary: {len(allsum)} species ===")
print(allsum.to_string(index=False))
PYEOF
echo "=== campaign done $(date) ==="
