/**
 * Runs the Caldera meshlet benchmark (`?benchmark=1` in graphics/meshlet-caldera) in Chrome and
 * prints frame-time statistics per pass and per leg of the route.
 *
 * The camera flies a fixed low-altitude loop through the main regions twice: once looking along
 * the direction of travel, once looking at the mountain. The route and the result format are in
 * examples/src/examples/graphics/meshlet-caldera/benchmark.mjs, which reports back through
 * `BENCHMARK_RESULT` / `BENCHMARK_ERROR` console lines.
 *
 *   npm run bench:caldera -- [options]
 *
 * Options:
 *   --port <n>          Examples dev server port (default $EXAMPLES_PORT or 5555). A server is
 *                       started when nothing answers there.
 *   --url <url>         Full example URL instead of the local dev server.
 *   --pool <MiB>        Meshlet page pool (default 4096, the whole island).
 *   --speed <m/s>       Camera speed (default 60).
 *   --passes <list>     Comma separated: forward, mountain, jump (default all three). The jump pass
 *                       cuts between regions and reports `lastSlowMs`, the ms after the cut of
 *                       the last frame over 20 ms.
 *   --regions <list>    Only load these regions (a faster, shorter smoke test; the route still
 *                       covers the whole island).
 *   --cache             Keep fetched bytes in the OPFS cache, as `?cache=1` (a warm run).
 *   --shadows           Sun casts four 2048 px cascades (shadowDistance 1500), meshlets included.
 *   --headless          Run without a window. Frame pacing differs from a real window.
 *   --engine <type>     dev (default: engine source, with debug checks), prf, rel or dbg: serve that
 *                       build from build/playcanvas.<type>.mjs (run `npm run build:<type>:esm`
 *                       first). Only applies when this script starts the dev server itself.
 *   --profile           Record a JS CPU profile of the run and print the heaviest functions.
 *   --trace <file>      Record a Chrome timeline trace (open it in DevTools or Perfetto). Large: use
 *                       with a single pass.
 *   --out <file>        Also write the JSON result here.
 *   --timeout <s>       Give up after this many seconds (default 1800).
 *   --param k=v         Extra example URL parameter, repeatable. Useful ones: dagPixel=<px> (LOD
 *                       error target, lower is finer), benchDwell=<s> (seconds each jump is
 *                       watched, default 8), cacheReads=<n>.
 *   --example <name>    caldera (default) or zorah. Zorah supports --load only.
 *   --load              Measure loading instead of flying the route: milestones (containers loaded,
 *                       playable, settled), world rebuilds, main-thread long tasks, requests and
 *                       byte-cache use. Combine with --profile for a CPU breakdown.
 *   --repeat <n>        Run n times and print the median (default 1). Each run is a new browser.
 *   --user-data <dir>   Chrome profile directory kept between runs, so an OPFS cache (--cache)
 *                       filled by one run is warm for the next. Without it every run is cold.
 *   --throttle <Mbit>   Limit download bandwidth (with --latency <ms>, default 0), to stand in for
 *                       a remote host: localhost makes the network look free.
 *
 * Environment: CHROME_PATH overrides the Chrome executable (default: the installed Google Chrome).
 */
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { launch } from 'puppeteer';

const examplesDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');

const args = process.argv.slice(2);
const flags = new Map();
const extraParams = [];
for (let i = 0; i < args.length; i++) {
    if (!args[i].startsWith('--')) continue;
    const key = args[i].slice(2);
    const next = args[i + 1];
    if (next === undefined || next.startsWith('--')) {
        flags.set(key, true);
    } else {
        if (key === 'param') extraParams.push(next);
        else flags.set(key, next);
        i++;
    }
}

const engine = String(flags.get('engine') ?? 'dev');
const enginePath = engine === 'dev' ? null : path.resolve(examplesDir, `../build/playcanvas.${engine}.mjs`);

const port = Number(flags.get('port') ?? process.env.EXAMPLES_PORT ?? 5555);
const timeoutMs = Number(flags.get('timeout') ?? 1800) * 1000;

const query = new URLSearchParams({
    deviceType: 'webgpu',
    benchmark: flags.get('load') ? 'load' : '1'
});
const example = String(flags.get('example') ?? 'caldera');
// Zorah sizes its own pool unless told to; Caldera's whole island needs 4 GiB
if (flags.get('pool') || example === 'caldera') query.set('pool', String(flags.get('pool') ?? 4096));
if (flags.get('speed')) query.set('benchSpeed', String(flags.get('speed')));
if (flags.get('passes')) query.set('benchPasses', String(flags.get('passes')));
if (flags.get('regions')) query.set('regions', String(flags.get('regions')));
if (flags.get('cache')) query.set('cache', '1');
if (flags.get('shadows')) query.set('shadows', '1');
for (const kv of extraParams) {
    const [k, ...v] = kv.split('=');
    query.set(k, v.join('='));
}

const baseUrl = flags.get('url') ?? `http://localhost:${port}/iframe/graphics_meshlet-${example}.html`;
const url = `${baseUrl}${baseUrl.includes('?') ? '&' : '?'}${query}`;

/**
 * @param {string} target - URL to probe.
 * @returns {Promise<boolean>} True when the server answers.
 */
const reachable = async (target) => {
    try {
        const response = await fetch(target, { method: 'HEAD' });
        return response.ok;
    } catch {
        return false;
    }
};

/** @type {import('node:child_process').ChildProcess|null} */
let server = null;

const stopServer = () => {
    // npm starts vite as a child: signal the whole process group
    if (server?.pid) process.kill(-server.pid, 'SIGTERM');
};

const ensureServer = async () => {
    if (flags.get('url') || await reachable(baseUrl)) {
        if (enginePath) console.log('A server is already running: --engine has no effect on it.');
        return;
    }
    if (enginePath && !fs.existsSync(enginePath)) {
        throw new Error(`${enginePath} does not exist; run \`npm run build:${engine}:esm\` first`);
    }
    console.log(`Starting the examples dev server on port ${port} (engine: ${engine})...`);
    server = spawn('npm', ['run', 'develop'], {
        cwd: examplesDir,
        env: {
            ...process.env,
            EXAMPLES_PORT: String(port),
            ...(enginePath ? { ENGINE_PATH: path.relative(examplesDir, enginePath) } : {})
        },
        stdio: 'ignore',
        detached: true
    });
    const deadline = Date.now() + 180000;
    while (Date.now() < deadline) {
        if (await reachable(baseUrl)) return; // eslint-disable-line no-await-in-loop
        await new Promise((resolve) => {  // eslint-disable-line no-await-in-loop
            setTimeout(resolve, 1000);
        });
    }
    throw new Error('The examples dev server did not start in time');
};

/**
 * @param {object} stats - A summary from the example.
 * @returns {string} One table row of the numbers.
 */
const row = stats => [
    String(stats.frames).padStart(6),
    stats.avgMs.toFixed(1).padStart(7),
    stats.p95Ms.toFixed(1).padStart(7),
    stats.p99Ms.toFixed(1).padStart(7),
    stats.maxMs.toFixed(0).padStart(7),
    String(stats.over33).padStart(6),
    String(stats.over50).padStart(6),
    String(stats.over100).padStart(7),
    stats.lastSlowMs === undefined ? '' : `${String(stats.lastSlowMs).padStart(10)}`,
    stats.lastPageMs === undefined ? '' : `${String(stats.lastPageMs).padStart(10)} ${String(stats.settledMs ?? 'n/a').padStart(9)} ${String(stats.pages).padStart(7)}`
].join(' ');

const header = `${'frames'.padStart(6)} ${'avg'.padStart(7)} ${'p95'.padStart(7)} ${'p99'.padStart(7)} ${'max'.padStart(7)} ${'>33ms'.padStart(6)} ${'>50ms'.padStart(6)} ${'>100ms'.padStart(7)}`;
const jumpHeader = `${header} ${'lastSlowMs'.padStart(10)} ${'lastPageMs'.padStart(10)} ${'settledMs'.padStart(9)} ${'pages'.padStart(7)}`;

/**
 * @param {object} results - The example's results.
 */
const printResults = (results) => {
    console.log(`\nCaldera benchmark: ${results.pathLengthM} m route at ${results.speed} m/s, pool ${results.pool} MiB, ${results.device}`);
    for (const [mode, pass] of Object.entries(results.passes)) {
        console.log(`\n== ${mode} pass (${pass.seconds}s${pass.settledBeforeStart ? '' : ', start did not fully settle'}) ==`);
        console.log(`${'overall'.padEnd(34)} ${header}`);
        console.log(`${''.padEnd(34)} ${row(pass)}`);
        console.log(`GPU avg ${pass.gpuAvgMs ?? 'n/a'} ms, pages fetched ${pass.pagesFetched}, evicted ${pass.pagesEvicted}, installed but already unwanted ${pass.pagesUnwanted}, dropped unsent ${pass.pagesDropped}`);
        if (pass.missingAvg !== undefined) {
            console.log(`Pages wanted but not resident per frame: avg ${pass.missingAvg}, p95 ${pass.missingP95}, frames with any ${pass.missingFramesPct}%`);
        }
        if (pass.gpuPasses) {
            console.log(`GPU ms per frame by pass: ${Object.entries(pass.gpuPasses).map(([name, ms]) => `${name} ${ms}`).join(', ')}`);
        }
        console.log(`${(mode === 'jump' ? 'stop' : 'leg').padEnd(34)} ${mode === 'jump' ? jumpHeader : header}`);
        for (const [leg, stats] of Object.entries(pass.legs)) {
            console.log(`${leg.padEnd(34)} ${row(stats)}`);
        }
    }
};

/**
 * Prints the functions with the most self time in a CDP CPU profile.
 *
 * @param {object} profile - The CPU profile.
 */
const printProfile = (profile) => {
    const byId = new Map(profile.nodes.map(node => [node.id, node]));
    const self = new Map();
    let total = 0;
    profile.samples.forEach((id, i) => {
        const dt = profile.timeDeltas[i] / 1000;
        const { functionName, url: file, lineNumber } = byId.get(id).callFrame;
        const key = `${functionName || '(anonymous)'}  ${file.split('/').pop()}:${lineNumber + 1}`;
        self.set(key, (self.get(key) ?? 0) + dt);
        total += dt;
    });
    const busy = total - (self.get('(idle)  :0') ?? 0);
    console.log(`\nCPU profile: ${Math.round(busy)} ms busy of ${Math.round(total)} ms. Self time by function:`);
    [...self.entries()].filter(([key]) => !key.startsWith('(idle)')).sort((a, b) => b[1] - a[1]).slice(0, 30).forEach(([key, ms]) => {
        console.log(`${String(Math.round(ms)).padStart(7)} ms  ${key}`);
    });
    if (flags.get('out')) {
        const file = `${String(flags.get('out')).replace(/\.json$/, '')}.cpuprofile`;
        fs.writeFileSync(file, JSON.stringify(profile));
        console.log(`Wrote ${file}`);
    }
};

const runOnce = async () => {
    const browser = await launch({
        ...(process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : { channel: 'chrome' }),
        headless: !!flags.get('headless'),
        ...(flags.get('user-data') ? { userDataDir: path.resolve(String(flags.get('user-data'))) } : {}),
        defaultViewport: { width: 1920, height: 1080 },
        args: [
            '--enable-unsafe-webgpu',
            '--enable-features=WebGPUDeveloperFeatures',
            '--ignore-gpu-blocklist',
            // a throttled background tab would skew frame times
            '--disable-background-timer-throttling',
            '--disable-renderer-backgrounding',
            '--disable-backgrounding-occluded-windows'
        ]
    });

    try {
        const page = await browser.newPage();
        const finished = new Promise((resolve, reject) => {
            const timer = setTimeout(() => reject(new Error('Benchmark timed out')), timeoutMs);
            page.on('console', (message) => {
                const text = message.text();
                if (text.startsWith('BENCHMARK_RESULT ')) {
                    clearTimeout(timer);
                    resolve(JSON.parse(text.slice('BENCHMARK_RESULT '.length)));
                } else if (text.startsWith('BENCHMARK_ERROR ')) {
                    clearTimeout(timer);
                    reject(new Error(text.slice('BENCHMARK_ERROR '.length)));
                } else if (text.startsWith('Benchmark ')) {
                    console.log(text);
                }
            });
        });
        page.on('pageerror', (err) => {
            console.error(`page error: ${err.message}`);
        });

        if (flags.get('throttle')) {
            const net = await page.createCDPSession();
            await net.send('Network.enable');
            await net.send('Network.emulateNetworkConditions', {
                offline: false,
                latency: Number(flags.get('latency') ?? 0),
                downloadThroughput: Number(flags.get('throttle')) * 125000,
                uploadThroughput: -1
            });
        }

        console.log(`Opening ${url}`);
        const client = flags.get('profile') ? await page.createCDPSession() : null;
        if (client) {
            await client.send('Profiler.enable');
            await client.send('Profiler.setSamplingInterval', { interval: 250 });
            await client.send('Profiler.start');
        }
        if (flags.get('trace')) {
            await page.tracing.start({
                path: String(flags.get('trace')),
                categories: ['devtools.timeline', 'disabled-by-default-devtools.timeline', 'v8', 'blink.user_timing']
            });
        }
        await page.goto(url, { waitUntil: 'domcontentloaded' });
        const result = await finished;
        if (flags.get('trace')) await page.tracing.stop();
        if (client) printProfile((await client.send('Profiler.stop')).profile);

        return result;
    } finally {
        await browser.close();
    }
};

/**
 * @param {number[]} values - Numbers.
 * @returns {number} The median.
 */
const median = (values) => {
    const sorted = [...values].sort((a, b) => a - b);
    const mid = sorted.length >> 1;
    return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
};

/**
 * Prints one load run.
 *
 * @param {object} load - The example's `load` section.
 */
const printLoad = (load) => {
    const m = load.milestonesMs;
    const ms = v => (v === null ? 'n/a' : `${(v / 1000).toFixed(2)} s`);
    console.log(`\n${example} load: pool ${load.pool} MiB, ${load.device}, cache ${load.cache ? 'on' : 'off'}`);
    console.log(`  script ready      ${ms(m.scriptReady)}   (imports + device init, before any content)`);
    console.log(`  scene created     ${ms(m.sceneCreated)}`);
    console.log(`  first container   ${ms(m.firstContainer)}`);
    console.log(`  all containers    ${ms(m.allContainers)}   (${load.containers.loaded}/${load.containers.total})`);
    console.log(`  playable          ${ms(m.playable)}`);
    console.log(`  settled           ${ms(m.settled)}   (playable floor met, demand settled)`);
    console.log(`  pages quiet       ${ms(m.pagesQuiet)}   (last page request)`);
    console.log(`  pages requested   ${load.pageTimeline.map(t => `${(t.atMs / 1000).toFixed(0)}s:${t.pages}`).join(' ')}`);
    const r = load.rebuilds;
    console.log(`  rebuilds          ${r.count} (${r.beforePlayable} before playable), ${r.totalMs} ms total, longest ${r.longestMs} ms`);
    const a = load.mainThread.untilPlayable;
    const b = load.mainThread.untilSettled;
    console.log(`  long tasks        until playable: ${a.count}, ${a.blockingMs} ms over 50 ms, longest ${a.longestMs} ms; until settled: ${b.count}, ${b.blockingMs} ms, longest ${b.longestMs} ms`);
    console.log(`  longest tasks     ${load.mainThread.top.map(t => `${t.ms}ms@${(t.atMs / 1000).toFixed(1)}s`).join(', ')}`);
    if (load.samples?.length) {
        console.log('  streaming         t(s)  frames  inFlight  queued  runs  arrived  resident  missing  cacheHits');
        load.samples.filter((_, i) => i % 2 === 1).forEach((x) => {
            console.log(`                    ${(x.atMs / 1000).toFixed(1).padStart(5)} ${String(x.frames).padStart(7)} ${String(x.inFlight).padStart(9)} ${String(x.queued).padStart(7)} ${String(x.runs).padStart(5)} ${String(x.arrived).padStart(8)} ${String(x.resident).padStart(9)} ${String(x.missing).padStart(8)} ${String(x.cacheHits).padStart(10)}`);
        });
    }
    if (load.slowFrames?.length) {
        console.log(`  slow frames (> 100 ms): ${load.slowFrames.length}; t(s) wall cpu gpu (ms)`);
        load.slowFrames.slice(0, 60).forEach((f) => {
            console.log(`                    ${(f.atMs / 1000).toFixed(1).padStart(5)} ${String(f.dtMs).padStart(6)} ${String(f.cpuMs).padStart(5)} ${String(f.gpuMs ?? 'n/a').padStart(6)}`);
        });
    }
    if (load.idleRead) console.log(`  idle cache reads  ${load.idleRead.reads} reads of 768 KB after the load, 8 concurrent: ${load.idleRead.avgMs} ms each, ${load.idleRead.MBps} MB/s`);
    if (load.cutTables) {
        const c = load.cutTables;
        console.log(`  GPU tables        ${c.worlds} world(s), ${c.meshlets} meshlets: records ${c.recordsMB.toFixed(0)} MB, cut groups ${c.cutMB.toFixed(0)} MB, topology ${c.topologyMB.toFixed(0)} MB`);
    }
    const n = load.network;
    console.log(`  network           ${n.requests} requests, ${n.bodyMB} MB body, ${n.transferMB} MB on the wire; scheduler ${JSON.stringify(n.scheduler)}, pages fetched ${n.pagesFetched}`);
    if (load.byteCache) {
        console.log(`  byte cache        ${load.byteCache.hits} hits, ${load.byteCache.readMB} MB read, ${load.byteCache.writtenMB} MB written, ${load.byteCache.entries} entries`);
    }
    console.log(`  frames ${load.frames}, JS heap ${load.jsHeapMB ?? 'n/a'} MB`);
};

/**
 * Prints the median of each milestone across runs.
 *
 * @param {object[]} loads - The runs' `load` sections.
 */
const printLoadSummary = (loads) => {
    const keys = Object.keys(loads[0].milestonesMs);
    console.log(`\n== Median of ${loads.length} runs (s) ==`);
    console.log(['run', ...keys, 'rebuilds ms', 'blocked ms'].map(k => k.padStart(14)).join(''));
    loads.forEach((load, i) => {
        const cells = keys.map(k => (load.milestonesMs[k] === null ? 'n/a' : (load.milestonesMs[k] / 1000).toFixed(2)));
        console.log([String(i + 1), ...cells, String(load.rebuilds.totalMs), String(load.mainThread.untilSettled.blockingMs)].map(c => c.padStart(14)).join(''));
    });
    if (loads.length > 1) {
        const cells = keys.map((k) => {
            const values = loads.map(l => l.milestonesMs[k]).filter(v => v !== null);
            return values.length ? (median(values) / 1000).toFixed(2) : 'n/a';
        });
        console.log(['median', ...cells, String(median(loads.map(l => l.rebuilds.totalMs))), String(median(loads.map(l => l.mainThread.untilSettled.blockingMs)))].map(c => c.padStart(14)).join(''));
    }
};

const main = async () => {
    await ensureServer();
    const repeat = Math.max(1, Number(flags.get('repeat') ?? 1));
    const results = [];
    for (let i = 0; i < repeat; i++) {
        if (repeat > 1) console.log(`\n##### Run ${i + 1} of ${repeat} #####`);
        const result = await runOnce(); // eslint-disable-line no-await-in-loop
        results.push(result);
        if (result.load) printLoad(result.load);
        else printResults(result);
    }
    if (results[0].load) printLoadSummary(results.map(r => r.load));
    if (flags.get('out')) {
        const out = repeat > 1 || results[0].load ? results : results[0];
        fs.writeFileSync(String(flags.get('out')), JSON.stringify(out, null, 2));
        console.log(`\nWrote ${flags.get('out')}`);
    }
};

main().then(() => {
    stopServer();
}, (err) => {
    console.error(err);
    stopServer();
    process.exitCode = 1;
});
