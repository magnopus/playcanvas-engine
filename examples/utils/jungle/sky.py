"""
Converts the JungleRuins dome light (a latlong HDR) into the example's sky: a skybox HDR, a
clamped environment HDR for image-based lighting and a directional 'sun' extracted from the
clamped energy, so the canopy maps and the direct light share one direction.

The reference render is lit by the DomeLight alone. kloppenheim_06 has no hard sun disk: the
key light is a bright cloud glow ~22 degrees above the horizon, about a third of the dome's
energy. Radiance above --clamp is moved out of the environment into the directional light,
whose direction is the energy-weighted mean of what was moved.

Orientation: the USD stage is Z-up with poleAxis 'scene', so the dome's pole is +Z and
longitude 0 points to -Y (UsdLux DomeLight_1 docs). Converted to the engine's Y-up frame
(glTF x, z, -y) that is the engine's equirect mapping mirrored horizontally.

--to-sun overrides the sun direction (the HDR's glow sits low, ~22 degrees, which reads as dusk);
the dome is then turned about the vertical so the glow keeps the sun's azimuth, and the extracted
energy and colour are kept.

Writes jungle_sky.hdr (skybox), jungle_sky_env.hdr (lighting) and jungle_sky.json:
{ toSun, sunColor, sunIntensity, clamp } - toSun in engine space, sunIntensity in the engine's
light units (irradiance / pi, so it balances an environment at intensity 1).
"""
import argparse
import json
import math
import os
import subprocess

import numpy as np

LUMA = np.array([0.2126, 0.7152, 0.0722])


def read_hdr(path):
    """Decodes a Radiance HDR to float RGB (rows top to bottom) through ffmpeg."""
    probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                            'stream=width,height', '-of', 'csv=p=0', path],
                           check=True, capture_output=True, text=True).stdout.strip().split(',')
    w, h = int(probe[0]), int(probe[1])
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-pix_fmt', 'gbrpf32le', '-f', 'rawvideo', '-'],
                         check=True, capture_output=True).stdout
    g, b, r = np.frombuffer(raw, dtype=np.float32).reshape(3, h, w)
    return np.stack([r, g, b], -1)


def downsample(img, w):
    f = img.shape[1] // w
    return img.reshape(img.shape[0] // f, f, w, f, 3).mean((1, 3)) if f > 1 else img


def write_hdr(path, img):
    """Flat (non-RLE) RGBE, which the engine's HDR parser accepts."""
    h, w, _ = img.shape
    m = img.max(-1)
    e = np.where(m > 1e-32, np.floor(np.log2(np.maximum(m, 1e-32))) + 1, 0)
    scale = np.where(m > 1e-32, 256.0 / np.exp2(e), 0)
    rgbe = np.empty((h, w, 4), dtype=np.uint8)
    rgbe[..., :3] = np.clip(img * scale[..., None], 0, 255).astype(np.uint8)
    rgbe[..., 3] = np.where(m > 1e-32, e + 128, 0).astype(np.uint8)
    with open(path, 'wb') as f:
        f.write(f'#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y {h} +X {w}\n'.encode())
        f.write(rgbe.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hdr', default=os.path.expanduser('~/Downloads/JungleRuins/USD/kloppenheim_06_puresky_8k.hdr'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--clamp', type=float, default=1.5, help='radiance above this moves into the sun')
    ap.add_argument('--sky-size', type=int, default=4096)
    ap.add_argument('--env-size', type=int, default=512)
    ap.add_argument('--to-sun', type=lambda v: [float(c) for c in v.split(',')],
                    help='engine-space direction to the sun, overriding the extracted one; the example\'s '
                         'original light, euler 45, 30, 0, is 0.35355,0.70711,0.61237')
    args = ap.parse_args()

    # mirror into the engine's equirect convention (see the module docs)
    src = read_hdr(args.hdr)[:, ::-1]
    sky = downsample(src, args.sky_size)
    work = downsample(src, 2048).astype(np.float64)
    h, w, _ = work.shape

    lat = (0.5 - (np.arange(h) + 0.5) / h) * np.pi
    lon = ((np.arange(w) + 0.5) / w * 2 - 1) * np.pi
    LAT, LON = np.meshgrid(lat, lon, indexing='ij')
    dirs = np.stack([np.cos(LAT) * np.sin(LON), np.sin(LAT), np.cos(LAT) * np.cos(LON)], -1)
    dw = np.cos(LAT) * (np.pi / h) * (2 * np.pi / w)

    lum = work @ LUMA
    keep = np.minimum(1.0, args.clamp / np.maximum(lum, 1e-9))
    env = work * keep[..., None]
    moved = (work - env) * dw[..., None]
    E = moved.sum((0, 1))
    weight = (moved @ LUMA)[..., None]
    to_sun = (dirs * weight).sum((0, 1))
    spread = np.linalg.norm(to_sun) / weight.sum()
    to_sun /= np.linalg.norm(to_sun)

    if args.to_sun:
        target = np.asarray(args.to_sun, dtype=np.float64)
        target /= np.linalg.norm(target)
        # turn the dome so the glow's azimuth follows the sun; u = 0.5 + longitude / 2pi
        turn = (math.atan2(target[0], target[2]) - math.atan2(to_sun[0], to_sun[2])) / (2 * math.pi)
        sky = np.roll(sky, round(turn * sky.shape[1]), axis=1)
        env = np.roll(env, round(turn * env.shape[1]), axis=1)
        print(f'  sun overridden to {target.round(4)}, dome turned {turn * 360:.1f} deg', flush=True)
        to_sun = target

    up = np.clip(dirs[..., 1], 0, None)
    E_env = (env * (dw * up)[..., None]).sum((0, 1))
    print(f'  sun dir {to_sun.round(4)} elevation {np.degrees(np.arcsin(to_sun[1])):.1f} deg, '
          f'lobe concentration {spread:.3f}', flush=True)
    print(f'  sun irradiance {E.round(3)} (normal), sky on horizontal {E_env.round(3)}', flush=True)

    write_hdr(os.path.join(args.out, 'jungle_sky.hdr'), sky.astype(np.float32))
    write_hdr(os.path.join(args.out, 'jungle_sky_env.hdr'), downsample(env, args.env_size).astype(np.float32))
    meta = {
        'toSun': [round(float(c), 5) for c in to_sun],
        'sunColor': [round(float(c), 4) for c in E / E.max()],
        'sunIntensity': round(float(E.max() / np.pi), 4),
        'clamp': args.clamp
    }
    with open(os.path.join(args.out, 'jungle_sky.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    print(f'  {meta}', flush=True)


if __name__ == '__main__':
    main()
