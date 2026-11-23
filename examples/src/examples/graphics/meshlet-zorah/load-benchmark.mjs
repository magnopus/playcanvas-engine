// The load timeline probe of `npm run bench:caldera -- --example zorah --load`
// (examples/utils/caldera/benchmark.mjs). Examples cannot import from each other, so this is a copy
// of the one in meshlet-caldera/benchmark.mjs: keep them in step.

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
