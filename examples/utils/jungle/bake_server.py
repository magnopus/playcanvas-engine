"""
Receives the atlases the meshlet-impostor-bake example renders and writes them as PNGs, with the
transparent texels' colour dilated into the opaque edge of each view tile (so bilinear sampling
at a leaf's edge never picks up black) and a manifest.

  python bake_server.py --out <dir> [--port 5600]

POST /atlas?name=<file>.png&w=<px>&h=<px>&grid=<views per side>   body: raw RGBA8, rows top to bottom
POST /manifest                                                      body: JSON, written to jungle_impostors.json
                                                                    (+ impostor_{albedo,normal}_<layer>.ktx2, one UASTC file per layer,
                                                                    and with --raw jungle_impostors_{albedo,normal}.rgba, raw RGBA8)
"""
import argparse
import json
import os
import subprocess
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

DILATE_ROUNDS = 8
# the shader samples the first few mips only (it clamps its LOD to 3): a longer chain would only
# add block-compressed levels smaller than a block
KTX_LEVELS = 6
KTX_ARGS = {
    'albedo': ['--format', 'R8G8B8A8_SRGB', '--assign-tf', 'srgb', '--assign-primaries', 'bt709'],
    'normal': ['--format', 'R8G8B8A8_UNORM', '--assign-tf', 'linear', '--assign-primaries', 'none']
}


def dilate(tile):
    """Fills fully transparent texels with the mean colour of their covered 4-neighbours, repeatedly."""
    rgb = tile[..., :3].astype(np.float32)
    covered = tile[..., 3] > 0
    for _ in range(DILATE_ROUNDS):
        if covered.all():
            break
        acc = np.zeros_like(rgb)
        n = np.zeros(covered.shape, np.float32)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            src_c = np.zeros_like(covered)
            src_rgb = np.zeros_like(rgb)
            ys, yd = (slice(dy, None), slice(0, -dy)) if dy > 0 else (slice(0, dy), slice(-dy, None)) if dy < 0 else (slice(None), slice(None))
            xs, xd = (slice(dx, None), slice(0, -dx)) if dx > 0 else (slice(0, dx), slice(-dx, None)) if dx < 0 else (slice(None), slice(None))
            src_c[ys, xs] = covered[yd, xd]
            src_rgb[ys, xs] = rgb[yd, xd]
            acc += src_rgb * src_c[..., None]
            n += src_c
        fill = ~covered & (n > 0)
        rgb[fill] = acc[fill] / n[fill][:, None]
        covered = covered | fill
    out = tile.copy()
    out[..., :3] = np.clip(rgb + 0.5, 0, 255).astype(np.uint8)
    return out


class Handler(BaseHTTPRequestHandler):
    out = '.'
    raw = False

    def _cors(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_POST(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        body = self.rfile.read(int(self.headers['Content-Length']))
        if url.path == '/atlas':
            w, h, grid = int(q['w']), int(q['h']), int(q['grid'])
            img = np.frombuffer(body, np.uint8).reshape(h, w, 4).copy()
            tw, th = w // grid, h // grid
            for j in range(grid):
                for i in range(grid):
                    sl = (slice(j * th, (j + 1) * th), slice(i * tw, (i + 1) * tw))
                    img[sl] = dilate(img[sl])
            name = os.path.basename(q['name'])
            Image.fromarray(img, 'RGBA').save(os.path.join(self.out, name))
            print(f'  wrote {name} {w}x{h}', flush=True)
        elif url.path == '/manifest':
            manifest = json.loads(body)
            self.write_ktx2(manifest)
            manifest['ktx2'] = True
            with open(os.path.join(self.out, 'jungle_impostors.json'), 'w') as f:
                f.write(json.dumps(manifest, indent=1))
            print('  wrote jungle_impostors.json', flush=True)
            if self.raw:
                self.write_raw(manifest)
        self.send_response(200)
        self._cors()
        self.end_headers()
        self.wfile.write(b'ok')

    def write_ktx2(self, manifest):
        """One UASTC KTX2 per layer and map, with mips: the example transcodes them to the device's
        block format (about a quarter of the memory of RGBA8)."""
        for p in manifest['prototypes']:
            for kind, args in KTX_ARGS.items():
                src = os.path.join(self.out, f'impostor_{kind}_{p["layer"]}.png')
                dst = os.path.join(self.out, f'impostor_{kind}_{p["layer"]}.ktx2')
                subprocess.run(['ktx', 'create', *args, '--encode', 'uastc', '--uastc-quality', '2', '--zstd', '18',
                                '--generate-mipmap', '--levels', str(KTX_LEVELS), '--mipmap-wrap', 'clamp', src, dst],
                               check=True)
            print(f'  encoded layer {p["layer"]} {p["name"]}', flush=True)

    def write_raw(self, manifest):
        """Concatenates every layer's PNG into one raw RGBA8 file per map: the example uploads them
        straight into a texture array (a canvas round trip would drop the colour under alpha 0)."""
        layers = max(p['layer'] for p in manifest['prototypes']) + 1
        width, height = manifest['atlasWidth'], manifest['atlasHeight']
        for kind in ('albedo', 'normal'):
            blob = np.zeros((layers, height, width, 4), np.uint8)
            for p in manifest['prototypes']:
                blob[p['layer']] = np.asarray(Image.open(os.path.join(self.out, f'impostor_{kind}_{p["layer"]}.png')).convert('RGBA'))
            name = f'jungle_impostors_{kind}.rgba'
            blob.tofile(os.path.join(self.out, name))
            print(f'  wrote {name} ({blob.nbytes >> 20} MiB)', flush=True)

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--port', type=int, default=5600)
    ap.add_argument('--raw', action='store_true', help='also write the raw RGBA8 atlases')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    Handler.out = args.out
    Handler.raw = args.raw
    print(f'listening on {args.port}, writing to {args.out}', flush=True)
    HTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
