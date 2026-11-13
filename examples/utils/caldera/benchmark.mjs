/**
 * Runs the Caldera meshlet benchmark (`?benchmark=1` in graphics/meshlet-caldera) in Chrome and
 * prints frame-time statistics per pass and per leg of the route.
 *
 * The camera flies a fixed low-altitude loop through the main regions twice: once looking along
 * the direction of travel, once looking at the mountain. See the benchmark section of
 * examples/src/examples/graphics/meshlet-caldera/example.mjs for the route.
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
 *                       cuts between regions and reports `slowUntil`, the ms after the cut of the
 *                       last frame over 20 ms.
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
 *   --param k=v         Extra example URL parameter, repeatable.
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
    benchmark: '1',
    pool: String(flags.get('pool') ?? 4096)
});
if (flags.get('speed')) query.set('benchSpeed', String(flags.get('speed')));
if (flags.get('passes')) query.set('benchPasses', String(flags.get('passes')));
if (flags.get('regions')) query.set('regions', String(flags.get('regions')));
if (flags.get('cache')) query.set('cache', '1');
if (flags.get('shadows')) query.set('shadows', '1');
for (const kv of extraParams) {
    const [k, ...v] = kv.split('=');
    query.set(k, v.join('='));
}

const baseUrl = flags.get('url') ?? `http://localhost:${port}/iframe/graphics_meshlet-caldera.html`;
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
    stats.lastSlowMs === undefined ? '' : `${String(stats.lastSlowMs).padStart(8)}`
].join(' ');

const header = `${'frames'.padStart(6)} ${'avg'.padStart(7)} ${'p95'.padStart(7)} ${'p99'.padStart(7)} ${'max'.padStart(7)} ${'>33ms'.padStart(6)} ${'>50ms'.padStart(6)} ${'>100ms'.padStart(7)}`;
const jumpHeader = `${header} ${'slowUntil'.padStart(8)}`;

/**
 * @param {object} results - The example's results.
 */
const printResults = (results) => {
    console.log(`\nCaldera benchmark: ${results.pathLengthM} m route at ${results.speed} m/s, pool ${results.pool} MiB, ${results.device}`);
    for (const [mode, pass] of Object.entries(results.passes)) {
        console.log(`\n== ${mode} pass (${pass.seconds}s${pass.settledBeforeStart ? '' : ', start did not fully settle'}) ==`);
        console.log(`${'overall'.padEnd(34)} ${header}`);
        console.log(`${''.padEnd(34)} ${row(pass)}`);
        console.log(`GPU avg ${pass.gpuAvgMs ?? 'n/a'} ms, pages fetched ${pass.pagesFetched}, evicted ${pass.pagesEvicted}`);
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

const main = async () => {
    await ensureServer();

    const browser = await launch({
        ...(process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : { channel: 'chrome' }),
        headless: !!flags.get('headless'),
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

        printResults(result);
        if (flags.get('out')) {
            fs.writeFileSync(String(flags.get('out')), JSON.stringify(result, null, 2));
            console.log(`\nWrote ${flags.get('out')}`);
        }
    } finally {
        await browser.close();
    }
};

main().then(() => {
    stopServer();
}, (err) => {
    console.error(err);
    stopServer();
    process.exitCode = 1;
});
