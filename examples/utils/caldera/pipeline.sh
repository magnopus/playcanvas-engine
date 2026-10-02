#!/bin/zsh
# convert (if needed) -> bake -> drop the raw render GLB. Collision GLBs are copied next to the bake.
set -u
setopt pipefail
# CALDERA: Caldera checkout, GLTF_TOOLS: gltf-tools checkout, PY: python with usd-core + numpy
PY=${PY:-python3}
HERE=${0:A:h}
export CALDERA_RAW=${CALDERA_RAW:-$PWD/caldera-raw}
RAW=$CALDERA_RAW
OUT=$HERE/../../assets/meshlets/caldera
TOOLS=${GLTF_TOOLS:?set GLTF_TOOLS to the gltf-tools checkout}
mkdir -p $OUT
for r in "$@"; do
    echo "== $r $(date +%T)"
    free_gb=$(df -g ${RAW:h} | awk 'NR==2 {print $4}')
    if (( free_gb < 3 )); then echo "STOPPING: only ${free_gb} GB free before $r"; break; fi
    # the previous bake goes first: disk is the constraint, and it is being replaced
    [[ -z "${NOBAKE:-}" ]] && rm -rf $OUT/caldera_$r.glb $OUT/caldera_${r}_meshlets $OUT/caldera_${r}_textures
    if [[ ! -f $RAW/caldera_$r.glb || -n "${RECONVERT:-}" ]]; then
        $PY -u $HERE/usd2glb.py --out $RAW $r 2>&1 | grep -v "^ *[0-9]* meshes" || { echo "CONVERT FAILED $r"; continue; }
    fi
    cp $RAW/caldera_${r}_collision.glb $OUT/
    if [[ -z "${NOBAKE:-}" ]]; then
        (cd $TOOLS && /usr/bin/time -l gltf-transform streamed-meshlets $RAW/caldera_$r.glb $OUT/caldera_$r.glb \
            --strip-source-geometry --no-tangents --config ./gltf-tools-plugin.js 2>&1 \
            | grep -E "streamed .* page|Processed|real|peak memory|rror|failed|mesh=.* prim=[0-9]+:") || echo "BAKE FAILED $r"
        [[ -f $OUT/caldera_$r.glb ]] && $PY $HERE/manifest.py 2>/dev/null
        [[ -f $OUT/caldera_$r.glb && -z "${KEEPRAW:-}" && $r != terrain ]] && rm -f $RAW/caldera_$r.glb
    fi
    df -h ~ | tail -1 | awk '{print "  disk free", $4}'
done
echo "PIPELINE DONE $(date +%T)"
