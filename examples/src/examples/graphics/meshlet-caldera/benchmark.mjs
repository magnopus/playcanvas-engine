// Benchmark and GPU timing overlay for the Caldera example, used only with `?benchmark=1` or
// `?gpuStats=1`. The benchmark is driven by `npm run bench:caldera`
// (examples/utils/caldera/benchmark.mjs), which reads the `Benchmark ` / `BENCHMARK_RESULT ` /
// `BENCHMARK_ERROR ` console lines this module prints.
//
// It flies a fixed low-altitude loop through the main regions and reports frame-time statistics
// per pass and per leg. The `forward` pass looks along the route (streaming load of a moving
// view), `mountain` runs the same route looking at the central mountain (culling and shadow load
// of a fixed far target), and `jump` cuts between regions and measures how frames recover.
//
// ?benchSpeed=<m/s> (60), ?benchClearance=<m above the sampled surface> (12),
// ?benchPasses=forward,mountain,jump, ?benchSettle=<max seconds to wait for streaming before a pass> (45),
// ?benchDwell=<seconds spent at each stop of the jump pass> (8)

import { Vec3 } from 'playcanvas';

/**
 * @import { AppBase, Entity, GraphicsDevice } from 'playcanvas'
 * @import { CameraControls } from 'playcanvas/scripts/esm/camera-controls.mjs'
 */

// Route through the region centres: glTF x / z metres, and the surface height sampled from above
// every ~80 m. WAYPOINTS index into PATH, so the camera passes through each region centre.
const PATH = [
    [428, 1093, 21.1],
    [426, 1011, 53.8],
    [425, 928, 103],
    [423, 846, 128.6],
    [421, 764, 140.1],
    [419, 681, 139.6],
    [418, 599, 128.9],
    [416, 516, 98.7],
    [414, 434, 77.4],
    [494, 426, 87.2],
    [574, 417, 93.4],
    [654, 409, 120.8],
    [734, 400, 131.7],
    [813, 392, 110.9],
    [893, 383, 94.8],
    [973, 375, 42.3],
    [1053, 366, 18.3],
    [1061, 288, 16.1],
    [1068, 210, 13.9],
    [1076, 131, 10.4],
    [1083, 53, 9.4],
    [1091, -25, 15.9],
    [1098, -103, 19.9],
    [1106, -181, 11.8],
    [1113, -260, 11.3],
    [1121, -338, 9.4],
    [1128, -416, 7.1],
    [1072, -477, 17.4],
    [1017, -538, 31.4],
    [961, -599, 32.7],
    [906, -660, 42.4],
    [850, -721, 46],
    [795, -782, 56.9],
    [739, -842, 72.2],
    [684, -903, 64.7],
    [628, -964, 59],
    [573, -1025, 48],
    [517, -1086, 16.7],
    [462, -1147, 10.7],
    [406, -1208, 10.6],
    [326, -1208, 23],
    [245, -1209, 23.4],
    [165, -1209, 22.2],
    [84, -1209, 20.8],
    [4, -1210, 26.4],
    [-76, -1210, 28.7],
    [-157, -1210, 25.9],
    [-237, -1211, 21.8],
    [-318, -1211, 18.4],
    [-398, -1211, 29.3],
    [-479, -1212, 8.3],
    [-559, -1212, 10.4],
    [-536, -1138, 8.3],
    [-514, -1063, 18.5],
    [-491, -989, 18.3],
    [-468, -914, 27.1],
    [-445, -840, 36],
    [-423, -766, 53.9],
    [-400, -691, 79.4],
    [-377, -617, 84.3],
    [-355, -542, 83.5],
    [-332, -468, 62.8],
    [-409, -480, 90.2],
    [-485, -493, 86.9],
    [-562, -505, 74.9],
    [-638, -517, 58.4],
    [-715, -530, 48.2],
    [-791, -542, 24],
    [-868, -554, 25.6],
    [-944, -567, 44.6],
    [-1021, -579, 34.4],
    [-1042, -489, 46.1],
    [-1063, -400, 48.2],
    [-1084, -310, 38.2],
    [-1028, -248, 40.9],
    [-973, -187, 47.2],
    [-917, -125, 71.7],
    [-861, -63, 70.8],
    [-806, -2, 80.2],
    [-750, 60, 79.6],
    [-694, 122, 42.6],
    [-639, 183, 18.5],
    [-583, 245, 17.2],
    [-527, 307, 22.5],
    [-472, 368, 22.5],
    [-416, 430, 22.6],
    [-351, 481, 22.5],
    [-286, 532, 36.6],
    [-221, 583, 43.1],
    [-156, 634, 44.1],
    [-91, 685, 50.2],
    [-26, 736, 64.8],
    [38, 787, 75],
    [103, 838, 95.9],
    [168, 889, 127.1],
    [233, 940, 102.7],
    [298, 991, 91.9],
    [363, 1042, 59.9],
    [428, 1093, 21.1]
];
const WAYPOINTS = [
    [0, 'capital'],
    [8, 'agricultural_center'],
    [16, 'subpen'],
    [26, 'beachhead'],
    [39, 'docks'],
    [51, 'arsenal'],
    [61, 'phosphate_mine'],
    [70, 'ruins'],
    [73, 'village'],
    [85, 'airfield'],
    [98, 'capital']
];

const LOOK_AHEAD = 60;
// each cut of the jump pass lands on the far side of the island from the last
const JUMP_ORDER = [
    'capital',
    'docks',
    'airfield',
    'beachhead',
    'ruins',
    'subpen',
    'arsenal',
    'village',
    'agricultural_center',
    'phosphate_mine'
];
const HITCH_MS = [33.4, 50, 100];

/**
 * @param {number[]} values - Frame times in ms.
 * @returns {object} Summary statistics.
 */
const summarize = (values) => {
    const sorted = Float64Array.from(values).sort();
    const n = sorted.length;
    const at = (q) => (n ? sorted[Math.min(n - 1, Math.floor(q * n))] : 0);
    const sum = values.reduce((a, b) => a + b, 0);
    const round = (v) => Math.round(v * 100) / 100;
    return {
        frames: n,
        avgMs: round(n ? sum / n : 0),
        p50Ms: round(at(0.5)),
        p95Ms: round(at(0.95)),
        p99Ms: round(at(0.99)),
        maxMs: round(n ? sorted[n - 1] : 0),
        over33: values.filter((v) => v > HITCH_MS[0]).length,
        over50: values.filter((v) => v > HITCH_MS[1]).length,
        over100: values.filter((v) => v > HITCH_MS[2]).length
    };
};

/**
 * Runs the benchmark passes and prints the result as a `BENCHMARK_RESULT <json>` console line.
 *
 * @param {object} options - Options.
 * @param {AppBase} options.app - The application.
 * @param {Entity} options.camera - The camera entity.
 * @param {CameraControls} options.controls - The camera controls, disabled while it runs.
 * @param {URLSearchParams} options.params - The example's URL parameters.
 * @param {{ regions: { name: string, min: number[], max: number[] }[] }} options.manifest - The
 * Caldera scene manifest.
 * @param {string} options.deviceType - The graphics device type, for the report.
 */
export const runBenchmark = async ({ app, camera, controls, params, manifest, deviceType }) => {
    // ?dagPixel=<px> sets the LOD error target (lower is finer: more pages to stream)
    if (params.get('dagPixel')) app.systems.meshlet.dagPixelThreshold = Number(params.get('dagPixel'));
    const speed = Number(params.get('benchSpeed')) || 60;
    const clearance = Number(params.get('benchClearance') ?? 12);
    const settleSeconds = Number(params.get('benchSettle')) || 45;
    let passes = (params.get('benchPasses') ?? 'forward,mountain,jump').split(',');

    // the mountain pass aims at the top of the `caldera` region, which a partial bake may lack
    const caldera = manifest.regions.find((r) => r.name === 'caldera');
    const mountain = caldera
        ? new Vec3((caldera.min[0] + caldera.max[0]) / 2, caldera.max[1], (caldera.min[2] + caldera.max[2]) / 2)
        : null;
    if (!mountain && passes.includes('mountain')) {
        console.warn('Benchmark: the caldera region is not baked, skipping the mountain pass');
        passes = passes.filter((p) => p !== 'mountain');
    }

    // camera height: the highest surface within two samples either side, so a hill or roof just
    // ahead does not swallow the camera between samples
    const points = PATH.map(([x, z], i) => {
        let top = 0;
        for (let k = Math.max(0, i - 2); k <= Math.min(PATH.length - 1, i + 2); k++) top = Math.max(top, PATH[k][2]);
        return new Vec3(x, top + clearance, z);
    });
    const cumulative = [0];
    for (let i = 1; i < points.length; i++) cumulative.push(cumulative[i - 1] + points[i].distance(points[i - 1]));
    const length = cumulative[cumulative.length - 1];

    const pathIndex = (d) => {
        let i = 1;
        while (i < points.length - 1 && cumulative[i] < d) i++;
        return i;
    };
    const sampleAt = (d, out) => {
        d = Math.min(Math.max(d, 0), length);
        const i = pathIndex(d);
        const span = cumulative[i] - cumulative[i - 1];
        return out.lerp(points[i - 1], points[i], span > 0 ? (d - cumulative[i - 1]) / span : 0);
    };
    const legOf = (d) => {
        const i = pathIndex(d);
        let leg = 0;
        while (leg < WAYPOINTS.length - 2 && i > WAYPOINTS[leg + 1][0]) leg++;
        return leg;
    };

    const gpuProfiler = app.graphicsDevice.gpuProfiler;
    if (gpuProfiler) {
        gpuProfiler.enabled = true;
        gpuProfiler.passDetail = true;
    }
    const directors = () => app.systems.meshlet.directors.filter((d) => d.world.finalized);
    const streamCounters = () =>
        directors().reduce(
            (acc, d) => {
                acc.fetched += d.residency?.fetchedPages ?? 0;
                acc.evicted += d.residency?.evictedPages ?? 0;
                acc.unwanted += d.residency?.installedUnwanted ?? 0;
                acc.dropped += d.residency?.droppedPages ?? 0;
                return acc;
            },
            { fetched: 0, evicted: 0, unwanted: 0, dropped: 0 }
        );
    // pages the cut wants that are not resident: while there are any, the view shows a coarser
    // level than it should, which is the pop-in felt when flying
    const missingNow = () => directors().reduce((n, d) => n + (d.residency?.lastMissingWanted ?? 0), 0);
    const nextFrame = () =>
        new Promise((resolve) => {
            requestAnimationFrame(() => resolve(performance.now()));
        });
    const sleep = (ms) =>
        new Promise((resolve) => {
            setTimeout(resolve, ms);
        });

    controls.enabled = false;
    const position = new Vec3();
    const target = new Vec3();
    // ?dropUnwanted=0 keeps fetching queued page runs the view has moved past, to compare
    const keepUnwanted = params.get('dropUnwanted') === '0';
    const place = (d, mode) => {
        if (keepUnwanted) {
            for (const dir of directors()) if (dir.residency) dir.residency.dropUnwanted = false;
        }
        sampleAt(d, position);
        if (mode === 'mountain') target.copy(mountain);
        else sampleAt(d + LOOK_AHEAD, target);
        camera.setPosition(position);
        camera.lookAt(target);
    };

    // the passes and the settle wait are inherently sequential, one frame at a time
    /* eslint-disable no-await-in-loop */

    // wait for the world to exist and stream in around a view
    const settle = async (d, mode, minSeconds) => {
        const start = performance.now();
        for (;;) {
            place(d, mode);
            await nextFrame();
            const elapsed = (performance.now() - start) / 1000;
            // meshlet.playable also covers regions still loading, whose join-in rebuild stalls
            const ready =
                app.systems.meshlet.playable &&
                directors().length > 0 &&
                directors().every((dir) => dir.residency?.playable && dir.residency.demandSettled);
            if (elapsed >= minSeconds && ready) return true;
            if (elapsed > settleSeconds) return false;
        }
    };

    // cut straight to each region in turn and record how the frames recover: streaming a view in
    // from nothing is what a teleport or a fast flight into new ground costs
    const dwellSeconds = Number(params.get('benchDwell')) || 8;
    const runJumps = async () => {
        await settle(0, 'forward', 3);
        const order = JUMP_ORDER.filter((name) => WAYPOINTS.some((w) => w[1] === name));
        const all = [];
        const legs = {};
        const streamStart = streamCounters();
        const t0 = performance.now();
        for (const name of order) {
            const d = cumulative[WAYPOINTS.find((w) => w[1] === name)[0]];
            const frames = [];
            const timesOver20 = [];
            const pagesBefore = streamCounters().fetched;
            let pages = pagesBefore;
            let lastPageMs = 0;
            let settledMs = null;
            place(d, 'forward');
            const start = performance.now();
            let last = start;
            for (;;) {
                const now = await nextFrame();
                if (now - start > dwellSeconds * 1000) break;
                place(d, 'forward');
                const dt = now - last;
                last = now;
                frames.push(dt);
                if (dt > 20) timesOver20.push(now - start);
                const fetched = streamCounters().fetched;
                if (fetched !== pages) {
                    pages = fetched;
                    lastPageMs = now - start;
                }
                // not before 300 ms: the demand readback lags the cut, and the previous view's
                // settled state would read as this one's
                if (settledMs === null && now - start > 300 &&
                    directors().length > 0 && directors().every((dir) => dir.residency?.playable && dir.residency.demandSettled)) {
                    settledMs = now - start;
                }
            }
            all.push(...frames);
            // lastSlowMs: time after the cut of the last frame over 20 ms; lastPageMs: of the last
            // page request (nothing more asked for after it, if it is well before the dwell's
            // end); settledMs: when the cut was complete and nothing was in flight
            legs[`jump to ${name}`] = {
                ...summarize(frames),
                lastSlowMs: Math.round(timesOver20.length ? timesOver20[timesOver20.length - 1] : 0),
                lastPageMs: Math.round(lastPageMs),
                settledMs: settledMs === null ? null : Math.round(settledMs),
                pages: pages - pagesBefore
            };
        }
        const streamEnd = streamCounters();
        const stats = summarize(all);
        console.log(
            `Benchmark jump: avg ${stats.avgMs} ms, p99 ${stats.p99Ms} ms, max ${stats.maxMs} ms, >50 ms: ${stats.over50}, >100 ms: ${stats.over100}`
        );
        return {
            settledBeforeStart: true,
            seconds: Math.round((performance.now() - t0) / 100) / 10,
            ...stats,
            gpuAvgMs: null,
            pagesFetched: streamEnd.fetched - streamStart.fetched,
            pagesEvicted: streamEnd.evicted - streamStart.evicted,
                pagesUnwanted: streamEnd.unwanted - streamStart.unwanted,
                pagesDropped: streamEnd.dropped - streamStart.dropped,
            legs
        };
    };

    const results = {
        speed,
        clearance,
        pathLengthM: Math.round(length),
        device: deviceType,
        pool: app.systems.meshlet.poolBytes / (1024 * 1024),
        userAgent: navigator.userAgent,
        passes: {}
    };

    try {
        for (const mode of passes) {
            if (mode === 'jump') {
                results.passes.jump = await runJumps();
                continue;
            }
            const settled = await settle(0, mode, 3);
            const frames = [];
            const legFrames = WAYPOINTS.slice(1).map(() => []);
            const gpuSamples = [];
            const missingSamples = [];
            const gpuPassSums = new Map();
            const streamStart = streamCounters();
            const t0 = performance.now();
            let last = t0;
            for (;;) {
                const now = await nextFrame();
                const d = ((now - t0) / 1000) * speed;
                if (d >= length) break;
                place(d, mode);
                const dt = now - last;
                last = now;
                frames.push(dt);
                missingSamples.push(missingNow());
                legFrames[legOf(d)].push(dt);
                const gpuMs = gpuProfiler?.frameTime;
                if (gpuMs) {
                    gpuSamples.push(gpuMs);
                    gpuProfiler.passTimings.forEach((ms, name) =>
                        gpuPassSums.set(name, (gpuPassSums.get(name) ?? 0) + ms)
                    );
                }
            }
            const streamEnd = streamCounters();
            const passResult = {
                settledBeforeStart: settled,
                seconds: Math.round((performance.now() - t0) / 100) / 10,
                ...summarize(frames),
                gpuAvgMs: gpuSamples.length
                    ? Math.round((gpuSamples.reduce((a, b) => a + b, 0) / gpuSamples.length) * 100) / 100
                    : null,
                // average GPU ms per frame of the heaviest passes
                gpuPasses: Object.fromEntries(
                    [...gpuPassSums.entries()]
                        .sort((a, b) => b[1] - a[1])
                        .slice(0, 14)
                        .map(([name, sum]) => [name, Math.round((sum / Math.max(gpuSamples.length, 1)) * 100) / 100])
                ),
                pagesFetched: streamEnd.fetched - streamStart.fetched,
                pagesEvicted: streamEnd.evicted - streamStart.evicted,
                pagesUnwanted: streamEnd.unwanted - streamStart.unwanted,
                pagesDropped: streamEnd.dropped - streamStart.dropped,
                missingAvg: Math.round(missingSamples.reduce((a, b) => a + b, 0) / Math.max(missingSamples.length, 1)),
                missingP95: [...missingSamples].sort((a, b) => a - b)[Math.floor(missingSamples.length * 0.95)] ?? 0,
                missingFramesPct: Math.round((100 * missingSamples.filter(v => v > 0).length) / Math.max(missingSamples.length, 1)),
                legs: {}
            };
            legFrames.forEach((values, k) => {
                passResult.legs[`${WAYPOINTS[k][1]} -> ${WAYPOINTS[k + 1][1]}`] = summarize(values);
            });
            results.passes[mode] = passResult;
            console.log(
                `Benchmark ${mode}: avg ${passResult.avgMs} ms, p99 ${passResult.p99Ms} ms, max ${passResult.maxMs} ms, >50 ms: ${passResult.over50}, >100 ms: ${passResult.over100}`
            );
            await sleep(250);
        }
        console.log(`BENCHMARK_RESULT ${JSON.stringify(results)}`);
    } catch (err) {
        console.error(`BENCHMARK_ERROR ${err?.stack ?? err}`);
    } finally {
        // hand the camera back to the controls where the benchmark left it
        controls.reset(target, position);
        controls.enabled = true;
    }

    /* eslint-enable no-await-in-loop */
};

/**
 * Shows an on-screen table of GPU time per render pass, averaged over two seconds, for devices
 * without remote debugging. Needs the 'timestamp-query' WebGPU feature.
 *
 * @param {AppBase} app - The application.
 * @param {GraphicsDevice} device - The graphics device.
 */
export const showGpuStats = (app, device) => {
    const panel = document.createElement('pre');
    Object.assign(panel.style, {
        position: 'absolute',
        left: '8px',
        bottom: '8px',
        margin: '0',
        padding: '6px 8px',
        zIndex: '10',
        font: '11px/1.3 monospace',
        color: '#fff',
        background: 'rgba(0, 0, 0, 0.65)',
        pointerEvents: 'none'
    });
    document.body.appendChild(panel);
    app.on('destroy', () => panel.remove());

    const profiler = device.gpuProfiler;
    if (!device.supportsTimestampQuery || !profiler) {
        panel.textContent = 'GPU timings unavailable: no timestamp-query support on this device';
        return;
    }
    profiler.enabled = true;
    profiler.passDetail = true;

    const sums = new Map();
    let frames = 0;
    let gpuTotal = 0;
    let started = performance.now();
    app.on('frameend', () => {
        const frameTime = profiler.frameTime;
        if (frameTime) {
            frames++;
            gpuTotal += frameTime;
            profiler.passTimings.forEach((ms, name) => sums.set(name, (sums.get(name) ?? 0) + ms));
        }
        const now = performance.now();
        if (now - started < 2000 || !frames) return;
        const avg = (ms) => (ms / frames).toFixed(2).padStart(6);
        let camera = 0,
            shadow = 0;
        sums.forEach((ms, name) => {
            if (/Shadow/.test(name)) shadow += ms;
            else if (/^Meshlet/.test(name)) camera += ms;
        });
        const rows = [...sums.entries()]
            .sort((a, b) => b[1] - a[1])
            .slice(0, 18)
            .map(([name, ms]) => `${avg(ms)}  ${name}`);
        panel.textContent = [
            `GPU ${avg(gpuTotal)} ms/frame (${frames} frames)`,
            `meshlet camera ${avg(camera)}  shadows ${avg(shadow)}`,
            '',
            ...rows
        ].join('\n');
        sums.clear();
        frames = 0;
        gpuTotal = 0;
        started = now;
    });
};

/**
 * Starts recording the load timeline. Call before the scene's containers are created, so every
 * asset load, world rebuild and long task from then on is seen. Times are `performance.now()`,
 * which is milliseconds since navigation start.
 *
 * @param {AppBase} app - The application.
 * @param {URLSearchParams} [params] - The example's URL parameters: `cacheReads=<n>` sets the cap
 * on concurrent byte cache reads, to tune it; `renderEvery=<n>` draws every nth frame until playable.
 * @returns {object} The probe, to pass to {@link runLoadBenchmark}.
 */
export const startLoadProbe = (app, params = new URLSearchParams()) => {
    const probe = {
        startedAt: performance.now(),
        containerLoads: [],
        rebuilds: [],
        longTasks: [],
        frames: 0
    };

    app.assets.on('load', (asset) => {
        if (asset.type === 'container') probe.containerLoads.push(performance.now());
    });

    const system = app.systems.meshlet;
    const cacheReads = Number(params.get('cacheReads'));
    const scheduler = system.group?.fetchScheduler;
    if (cacheReads > 0 && scheduler) scheduler.maxConcurrentCacheReads = cacheReads;
    const rebuild = system._rebuild;
    system._rebuild = function (...args) {
        const start = performance.now();
        const result = rebuild.apply(this, args);
        const ms = performance.now() - start;
        // a call that only asked for saved tables or record restores and returned is not a rebuild
        if (ms >= 5) probe.rebuilds.push({ start, ms });
        return result;
    };

    try {
        new PerformanceObserver((list) => {
            for (const entry of list.getEntries()) probe.longTasks.push({ start: entry.startTime, ms: entry.duration });
        }).observe({ type: 'longtask', buffered: true });
    } catch {
        // long task timing is unavailable
    }
    // slow frames: wall time between frames, the CPU time inside the frame, and the GPU time
    const gpuProfiler = app.graphicsDevice.gpuProfiler;
    if (gpuProfiler && params.get('benchGpu') !== '0') gpuProfiler.enabled = true;
    probe.slowFrames = [];
    let frameStart = 0;
    let lastEnd = 0;
    app.on('frameupdate', () => {
        frameStart = performance.now();
    });
    app.on('frameend', () => {
        probe.frames++;
        const now = performance.now();
        const dt = lastEnd ? now - lastEnd : 0;
        lastEnd = now;
        if (dt > 100 && probe.slowFrames.length < 400) {
            probe.slowFrames.push({
                atMs: Math.round(now),
                dtMs: Math.round(dt),
                cpuMs: Math.round(now - frameStart),
                gpuMs: gpuProfiler?.frameTime ? Math.round(gpuProfiler.frameTime * 10) / 10 : null
            });
        }
    });

    // renderEvery=<n>: draw only every nth frame until playable, to see whether rendering during
    // the load slows it. Streaming runs inside the render phase, so it cannot stop altogether.
    const renderEvery = Number(params.get('renderEvery')) || 1;
    if (renderEvery > 1) {
        app.autoRender = false;
        let tick = 0;
        const onUpdate = () => {
            app.renderNextFrame = (tick++ % renderEvery) === 0;
        };
        app.on('update', onUpdate);
        system.once('playable', () => {
            app.off('update', onUpdate);
            app.autoRender = true;
        });
    }

    // streaming state every half second, to see what limits page throughput
    probe.samples = [];
    probe.sampler = setInterval(() => {
        let inFlight = 0, queued = 0, runs = 0, arrived = 0, resident = 0, missing = 0;
        for (const dir of system.directors) {
            const r = dir.residency;
            if (!r) continue;
            inFlight += r.inFlight.size;
            queued += r._queue.length;
            runs += r._activeRuns;
            arrived += r._arrived.length - r._arrivedHead;
            resident += r.residentPages;
            missing += r.lastMissingWanted ?? 0;
        }
        probe.samples.push({
            atMs: Math.round(performance.now()),
            frames: probe.frames,
            inFlight, queued, runs, arrived, resident, missing,
            cacheHits: app.loader.byteCache?.hits ?? 0
        });
    }, 500);
    return probe;
};

/**
 * Waits for the Caldera to become playable and settled, then prints the load timeline as a
 * `BENCHMARK_RESULT <json>` console line with a `load` section.
 *
 * @param {object} options - Options.
 * @param {AppBase} options.app - The application.
 * @param {object} options.probe - The probe from {@link startLoadProbe}.
 * @param {URLSearchParams} options.params - The example's URL parameters.
 * @param {string} options.deviceType - The graphics device type, for the report.
 */
export const runLoadBenchmark = async ({ app, probe, params, deviceType }) => {
    const system = app.systems.meshlet;
    const settleLimitMs = (Number(params.get('benchSettle')) || 120) * 1000;
    const round = (v) => Math.round(v * 10) / 10;
    const directors = () => system.directors.filter((d) => d.world.finalized);
    const nextFrame = () =>
        new Promise((resolve) => {
            requestAnimationFrame(() => resolve(performance.now()));
        });

    try {
        const containerTotal = app.assets.list().filter((a) => a.type === 'container').length;
        const readyAt = performance.now();
        await system.whenPlayable();
        const playableAt = performance.now();

        // playable shows the scene; settled also needs the streaming demand to be met. Pages
        // quiet is the last time a page was requested, once none has been for quietMs
        const quietMs = (Number(params.get('benchQuiet')) || 3) * 1000;
        const pagesRequested = () => directors().reduce((acc, d) => acc + (d.residency?.fetchedPages ?? 0), 0);
        let settledAt = null;
        let quietAt = null;
        let lastPages = pagesRequested();
        let lastChangeAt = playableAt;
        const timeline = [];
        let nextSample = playableAt;
        for (;;) {
            const now = await nextFrame(); // eslint-disable-line no-await-in-loop
            const pages = pagesRequested();
            if (pages !== lastPages) {
                lastPages = pages;
                lastChangeAt = now;
            }
            if (now >= nextSample) {
                timeline.push({ atMs: round(now), pages });
                nextSample = now + 2000;
            }
            if (settledAt === null && directors().length > 0 && directors().every((d) => d.residency?.playable && d.residency.demandSettled)) {
                settledAt = now;
            }
            if (quietAt === null && now - lastChangeAt >= quietMs) quietAt = lastChangeAt;
            if (settledAt !== null && quietAt !== null) break;
            if (now - playableAt > settleLimitMs) break;
        }
        const endAt = performance.now();
        clearInterval(probe.sampler);

        const byteCache = app.loader.byteCache ?? null;
        // the same kind of reads again with the load over: slow only while loading means the
        // main thread or GPU is the limit, slow here too means the cache path is
        let idleRead = null;
        const readCache = app.loader.byteCache;
        if (readCache?._entries) {
            const MB = 1048576;
            const urls = [...readCache._entries.values()].filter(e => !e.whole && e.size > 16 * MB).map(e => e.key);
            if (urls.length) {
                const total = 120, length = 768 * 1024;
                let next = 0, bytes = 0, ms = 0;
                const t0 = performance.now();
                await Promise.all(Array.from({ length: 8 }, async () => {
                    while (next < total) {
                        const i = next++;
                        const url = urls[i % urls.length];
                        const size = readCache._entries.get(url).size;
                        const offset = Math.floor(((i * 2654435761) % (size - length)) / 262144) * 262144;
                        const t = performance.now();
                        const data = await readCache.getRange(url, offset, length); // eslint-disable-line no-await-in-loop
                        ms += performance.now() - t;
                        bytes += data?.byteLength ?? 0;
                    }
                }));
                const sec = (performance.now() - t0) / 1000;
                idleRead = { reads: total, avgMs: round(ms / total), MBps: round(bytes / MB / sec) };
            }
        }

        // a populate run must finish writing, or the next run is only partly warm
        if (byteCache?.whenStored) await byteCache.whenStored();
        if (byteCache?.flush) await byteCache.flush();

        const loads = probe.containerLoads;
        const blocking = (from, to) => {
            const tasks = probe.longTasks.filter((t) => t.start >= from && t.start < to);
            return {
                count: tasks.length,
                blockingMs: round(tasks.reduce((sum, t) => sum + Math.max(0, t.ms - 50), 0)),
                longestMs: round(tasks.reduce((max, t) => Math.max(max, t.ms), 0))
            };
        };

        const resources = performance.getEntriesByType('resource');
        const sum = (key) => resources.reduce((acc, r) => acc + (r[key] || 0), 0);
        const schedulers = new Set(directors().map((d) => d.world.fetchScheduler).filter(Boolean));
        const scheduler = [...schedulers].reduce(
            (acc, s) => {
                acc.requests += s.sent;
                acc.coalesced += s.coalesced;
                acc.dropped += s.dropped;
                acc.cacheReads += s.cacheReads;
                acc.cacheReadMB += s.cacheReadBytes / 1048576;
                acc.cacheReadAvgMs += s.cacheReadMs;
                acc.cacheWaitAvgMs += s.cacheWaitMs;
                return acc;
            },
            { requests: 0, coalesced: 0, dropped: 0, cacheReads: 0, cacheReadMB: 0, cacheReadAvgMs: 0, cacheWaitAvgMs: 0 }
        );
        if (scheduler.cacheReads) {
            scheduler.cacheReadAvgMs = round(scheduler.cacheReadAvgMs / scheduler.cacheReads);
            scheduler.cacheWaitAvgMs = round(scheduler.cacheWaitAvgMs / scheduler.cacheReads);
            scheduler.cacheReadMB = round(scheduler.cacheReadMB);
        }
        const pagesFetched = directors().reduce((acc, d) => acc + (d.residency?.fetchedPages ?? 0), 0);
        const rebuildMs = probe.rebuilds.reduce((acc, r) => acc + r.ms, 0);

        const load = {
            device: deviceType,
            pool: Number(params.get('pool')) || 2048,
            cache: params.get('cache') === '1',
            milestonesMs: {
                scriptReady: round(probe.startedAt),
                sceneCreated: round(readyAt),
                firstContainer: loads.length ? round(Math.min(...loads)) : null,
                allContainers: loads.length >= containerTotal ? round(Math.max(...loads)) : null,
                playable: round(playableAt),
                settled: settledAt === null ? null : round(settledAt),
                pagesQuiet: quietAt === null ? null : round(quietAt)
            },
            pageTimeline: timeline,
            idleRead,
            cutTableStore: system._cutTables ? { ...system._cutTables.stats, minMeshlets: system._cutTables.minMeshlets } : null,
            cutTables: directors().reduce((acc, d) => {
                acc.cutMB += (d.world.cut?.groups?.byteSize ?? 0) / 1048576;
                acc.topologyMB += (d.world.cut?.selectionTopology?.byteSize ?? 0) / 1048576;
                acc.recordsMB += (d.world.meshletDataBuffer?.byteSize ?? 0) / 1048576;
                acc.meshlets += d.world.totalMeshlets ?? 0;
                acc.worlds++;
                return acc;
            }, { cutMB: 0, topologyMB: 0, recordsMB: 0, meshlets: 0, worlds: 0 }),
            samples: probe.samples,
            slowFrames: probe.slowFrames,
            containers: { loaded: loads.length, total: containerTotal },
            rebuilds: {
                count: probe.rebuilds.length,
                totalMs: round(rebuildMs),
                beforePlayable: probe.rebuilds.filter((r) => r.start < playableAt).length,
                longestMs: round(probe.rebuilds.reduce((max, r) => Math.max(max, r.ms), 0)),
                list: probe.rebuilds.map(r => ({ atMs: round(r.start), ms: round(r.ms) })),
                containerLoadsMs: loads.map(round)
            },
            mainThread: {
                untilPlayable: blocking(0, playableAt),
                untilSettled: blocking(0, endAt),
                top: [...probe.longTasks]
                    .sort((a, b) => b.ms - a.ms)
                    .slice(0, 8)
                    .map((t) => ({ atMs: round(t.start), ms: round(t.ms) }))
            },
            frames: probe.frames,
            network: {
                requests: resources.length,
                transferMB: round(sum('transferSize') / 1048576),
                bodyMB: round(sum('encodedBodySize') / 1048576),
                scheduler,
                pagesFetched
            },
            byteCache: byteCache ?
                {
                    hits: byteCache.hits,
                    readMB: round(byteCache.bytesRead / 1048576),
                    writtenMB: round(byteCache.bytesWritten / 1048576),
                    entries: byteCache.entryCount
                } :
                null,
            jsHeapMB: performance.memory ? round(performance.memory.usedJSHeapSize / 1048576) : null
        };
        console.log(`BENCHMARK_RESULT ${JSON.stringify({ load })}`);
    } catch (err) {
        console.error(`BENCHMARK_ERROR ${err?.stack ?? err}`);
    }
};
