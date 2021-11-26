#!/bin/bash

set -eu

DIRNAME="$(dirname $0)"
MODEL="$(realpath $1)"
IDENT="$2"
OUTPUT="$3"
pushd "$DIRNAME"
VAPORETTO_MODEL_PATH="$MODEL" wasm-pack build --release --target no-modules
popd
encoded_wasm=$(base64 < "${DIRNAME}/pkg/vaporetto_wasm_bg.wasm")
cat \
    <(sed "s/wasm_bindgen/__vaporetto_${IDENT}_wbg/g" < "${DIRNAME}/pkg/vaporetto_wasm.js") \
    <(echo "async function vaporetto_${IDENT}(){await __vaporetto_${IDENT}_wbg(fetch('data:application/wasm;base64,${encoded_wasm}'));return __vaporetto_${IDENT}_wbg.Vaporetto;}") \
    > "$OUTPUT"
