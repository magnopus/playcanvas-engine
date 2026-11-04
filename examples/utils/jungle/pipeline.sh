#!/bin/zsh
# convert (if needed) -> bake -> manifest for the JungleRuins groups.
#   pipeline.sh terrain static trees plants sky canopy treetest
# JUNGLE_RAW: export folder (raw GLBs + resized textures), GLTF_TOOLS: gltf-tools checkout,
# PY: python with usd-core + numpy + Pillow, JUNGLE_USD: the scene (defaults to ~/Downloads)
# RECONVERT=1 re-exports, EXPORT_ARGS passes extra usd2glb.py options (e.g. --forest-radius 1500),
# BAKE_ARGS extra streamed-meshlets options (default --srgb-encoding uastc: ETC1S colour is blocky
# on the close-up stone)
# 'canopy' bakes the leaf maps (canopy.py) into the output folder; CANOPY_ARGS passes its options,
# keep its --forest-radius in step with the trees export. 'sky' converts the dome HDR (sky.py) into
# the skybox, lighting HDR and sun direction (SKY_ARGS, default --to-sun = the 45 deg euler 45, 30, 0
# sun); run it before 'canopy', which bakes along that sun
set -u
setopt pipefail
PY=${PY:-python3}
HERE=${0:A:h}
RAW=${JUNGLE_RAW:-$PWD/jungle-raw}
OUT=${JUNGLE_OUT:-$HERE/../../assets/meshlets/jungle}
TOOLS=${GLTF_TOOLS:?set GLTF_TOOLS to the gltf-tools checkout}
mkdir -p $RAW $OUT
for g in "$@"; do
    echo "== $g $(date +%T)"
    if [[ $g == sky ]]; then
        $PY -u $HERE/sky.py --out $OUT ${=SKY_ARGS:---to-sun 0.35355,0.70711,0.61237} || echo "SKY FAILED"
        continue
    fi
    if [[ $g == treetest ]]; then
        # the isolated tree test scene (meshlet-impostor-test example), baked beside the scene, not in it
        $PY -u $HERE/protos.py --out $RAW --test-scene 2>&1 | grep -E "test scene|prototypes" || { echo "TREETEST EXPORT FAILED"; continue; }
        rm -rf $OUT/treetest && mkdir -p $OUT/treetest
        (cd $TOOLS && /usr/bin/time -l gltf-transform streamed-meshlets $RAW/jungle_treetest.glb $OUT/treetest/jungle_treetest.glb \
            --strip-source-geometry --no-tangents ${=BAKE_ARGS:---srgb-encoding uastc} --config ./gltf-tools-plugin.js 2>&1 \
            | grep -E "streamed .* page|Processed|real|peak memory|rror|failed") || echo "BAKE FAILED treetest"
        continue
    fi
    if [[ $g == canopy ]]; then
        $PY -u $HERE/canopy.py --out $OUT ${=CANOPY_ARGS:-} 2>&1 | grep -v "^  /root/\|^  splatted" || echo "CANOPY FAILED"
        continue
    fi
    if [[ ! -f $RAW/jungle_$g.glb || -n "${RECONVERT:-}" ]]; then
        $PY -u $HERE/usd2glb.py --out $RAW --groups $g ${=EXPORT_ARGS:-} 2>&1 | grep -v "^  /root/" \
            || { echo "CONVERT FAILED $g"; continue; }
    fi
    rm -rf $OUT/jungle_$g.glb $OUT/jungle_${g}_meshlets $OUT/jungle_${g}_textures
    (cd $TOOLS && /usr/bin/time -l gltf-transform streamed-meshlets $RAW/jungle_$g.glb $OUT/jungle_$g.glb \
        --strip-source-geometry --no-tangents ${=BAKE_ARGS:---srgb-encoding uastc} --config ./gltf-tools-plugin.js 2>&1 \
        | grep -E "streamed .* page|Processed|real|peak memory|rror|failed") || echo "BAKE FAILED $g"
    cp $RAW/jungle_$g.json $OUT/
    df -h ~ | tail -1 | awk '{print "  disk free", $4}'
done
$PY $HERE/manifest.py $OUT
echo "PIPELINE DONE $(date +%T)"
