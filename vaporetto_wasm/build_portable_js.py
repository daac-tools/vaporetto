#!/usr/bin/env python3

import argparse
import base64
import os
import subprocess


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='A path to the model file')
    parser.add_argument(
        '--identifier', required=True, help='An identifier that is used to the function name'
    )
    parser.add_argument('--output', required=True, help='A path to the generated file')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()

    working_dir = os.path.dirname(os.path.abspath('__file__'))

    # Builds a wasm with a model file.
    model_path = os.path.abspath(args.model)
    env = os.environ.copy()
    env['VAPORETTO_MODEL_PATH'] = model_path
    subprocess.run(
        ['wasm-pack', 'build', '--release', '--target', 'no-modules'],
        cwd=working_dir,
        env=env,
    )

    # Converts the wasm to the base64 string.
    wasm_path = os.path.join(working_dir, 'pkg/vaporetto_wasm_bg.wasm')
    with open(wasm_path, 'rb') as fp:
        wasm_data = fp.read()
    wasm_data_b64 = base64.b64encode(wasm_data).decode()

    # Reads the glue js file.
    js_path = os.path.join(working_dir, 'pkg/vaporetto_wasm.js')
    with open(js_path, 'rt') as fp:
        js_data = fp.read()

    # Generates a unified js file.
    with open(args.output, 'wt') as fp:
        print(
            js_data.replace('wasm_bindgen', f'__vaporetto_{args.identifier}_wbg'),
            file=fp,
        )
        print(f'async function vaporetto_{args.identifier}(){{', file=fp)
        print(f'    const data = "data:application/wasm;base64,{wasm_data_b64}";', file=fp)
        print(f'    await __vaporetto_{args.identifier}_wbg(fetch(data));', file=fp)
        print(f'    return __vaporetto_{args.identifier}_wbg.Vaporetto;', file=fp)
        print('}', file=fp)
