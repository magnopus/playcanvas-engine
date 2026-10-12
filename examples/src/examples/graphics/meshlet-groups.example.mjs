// @config
//
// Mission ISS baked with simplification groups of 8, 16 and 32. Orbit, pan and zoom any
// pane to move all cameras together. The LOD error and colour mode are shared. Each
// geometry pool fits its entire asset. See assets/meshlets/README.md for local bake setup.
//
// @flag WEBGL_DISABLED
// @flag WEBGPU_BARE_DISABLED
// @flag NO_MINISTATS
// @flag HIDDEN

import {
    AppBase,
    AppOptions,
    Asset,
    AssetListLoader,
    CameraComponentSystem,
    Color,
    ContainerHandler,
    Entity,
    FILLMODE_FILL_WINDOW,
    LightComponentSystem,
    MeshletComponentSystem,
    RESOLUTION_AUTO,
    TextureHandler,
    Vec3,
    basisInitialize,
    createGraphicsDevice
} from 'playcanvas';

const params = new URLSearchParams(location.search);
const approaches = params.get('approaches') === '1';
const variants = approaches
    ? [
          { key: '8', label: 'Original · group 8', file: 'iss-g8.glb' },
          { key: 'permissive', label: 'Relax internal seams', file: 'iss-permissive.glb' },
          { key: 'joined-safe', label: 'Join + seams · keep parts', file: 'iss-joined-safe.glb' }
      ]
    : [8, 16, 32].map((size) => ({ key: String(size), label: `Group ${size}`, file: `iss-g${size}.glb` }));
const directory = './assets/meshlets/iss-group-comparison/';
const channel = 'iss-meshlet-groups';
const variant = variants.find((entry) => entry.key === params.get('group'));
const canvas = /** @type {HTMLCanvasElement} */ (document.getElementById('application-canvas'));
const initialState = () => ({
    yaw: 35,
    pitch: 22,
    zoom: 1,
    pan: [0, 0, 0],
    threshold: 1,
    color: 0,
    coarse: false,
    framing: null
});

if (!variant) {
    canvas.hidden = true;
    const style = document.createElement('style');
    style.textContent = `
        body { background:#12161e; color:#e7edf5; font:14px system-ui,sans-serif; }
        main { height:100dvh; display:flex; flex-direction:column; }
        .toolbar { padding:16px; display:flex; gap:16px; flex-wrap:wrap; align-items:center; }
        strong { font-size:17px; margin-right:auto; } label { display:flex; align-items:center; gap:8px; }
        select,button { background:#242e3e; color:#e7edf5; border:1px solid #465368; border-radius:6px; padding:7px; font:inherit; }
        input { width:100px; accent-color:#78b8ff; }
        .panes { flex:1; min-height:0; display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:2px; }
        section { min-width:0; min-height:0; display:flex; flex-direction:column; background:#1c2330; }
        header { padding:13px; border-top:3px solid var(--accent); } h2 { margin:0 0 6px; font-size:18px; }
        small { color:#aab9cb; display:block; line-height:1.5; font-variant-numeric:tabular-nums; }
        iframe { border:0; width:100%; flex:1; min-height:0; background:#171e28; }
        .hint { margin:0; padding:11px 16px; color:#aab9cb; font-size:12px; }
    `;
    document.head.appendChild(style);
    const root = document.createElement('main');
    root.innerHTML = `<div class="toolbar"><strong>Mission ISS · LOD comparison</strong>
        <label>Compare <select id="comparison"><option value="0">Group sizes</option><option value="1">Bake approaches (trial)</option></select></label>
        <label>View <select id="colour"><option value="0">Material</option><option value="2">Meshlet colours</option><option value="1">LOD colours</option></select></label>
        <label>LOD error <input id="error" type="range" min="-2" max="6" step="0.25" value="0"><output>1 px</output></label>
        <button id="coarse">Coarsest LOD</button><button id="reset">Reset view</button></div>
        <div class="panes"></div><p class="hint">Drag to orbit · Shift-drag or right-drag to pan · Scroll to zoom · All views stay synchronized${approaches ? ' · Trials use group 8; joined trial welds exact positions only' : ''}</p>`;
    document.body.appendChild(root);
    const comparison = /** @type {HTMLSelectElement} */ (root.querySelector('#comparison'));
    comparison.value = approaches ? '1' : '0';
    comparison.addEventListener('change', () => {
        const url = new URL(location.href);
        url.searchParams.set('approaches', comparison.value);
        location.assign(url.href);
    });
    const frames = variants.map((entry, i) => {
        const pane = document.createElement('section');
        pane.style.setProperty('--accent', ['#78b8ff', '#72d9b2', '#e6b474'][i]);
        pane.innerHTML = `<header><h2>${entry.label}</h2><small class="bake">Loading bake…</small><small class="live">Loading model…</small></header>`;
        const frame = document.createElement('iframe');
        frame.title = `Mission ISS, ${entry.label}`;
        const url = new URL(location.href);
        url.searchParams.set('group', entry.key);
        url.searchParams.set('deviceType', 'webgpu');
        frame.src = url.href;
        pane.appendChild(frame);
        root.querySelector('.panes').appendChild(pane);
        return { frame, pane, key: entry.key };
    });
    const state = initialState();
    let radius = 1;
    const broadcast = () => {
        for (const { frame } of frames) {
            frame.contentWindow?.postMessage({ channel, type: 'state', state }, location.origin);
        }
    };
    window.addEventListener('message', (event) => {
        const i = frames.findIndex(({ frame }) => frame.contentWindow === event.source);
        if (event.origin !== location.origin || i < 0 || event.data?.channel !== channel) return;
        const m = event.data;
        if (m.type === 'ready') {
            if (i === 0) {
                radius = m.radius;
                state.framing = { radius, center: m.center };
            }
            broadcast();
        } else if (m.type === 'stats' || m.type === 'error') {
            frames[i].pane.querySelector('.live').textContent = m.text;
        } else if (m.type === 'input') {
            if (m.kind === 'zoom') {
                state.zoom = Math.max(0.015, Math.min(20, state.zoom * Math.exp(m.delta * 0.001)));
            } else if (m.kind === 'pan') {
                const yaw = (state.yaw * Math.PI) / 180,
                    pitch = (state.pitch * Math.PI) / 180;
                const scale = (radius * state.zoom * 4) / m.height;
                state.pan[0] += (-m.dx * Math.cos(yaw) - m.dy * Math.sin(yaw) * Math.sin(pitch)) * scale;
                state.pan[1] += m.dy * Math.cos(pitch) * scale;
                state.pan[2] += (m.dx * Math.sin(yaw) - m.dy * Math.cos(yaw) * Math.sin(pitch)) * scale;
            } else {
                state.yaw -= m.dx * 0.3;
                state.pitch = Math.max(-89, Math.min(89, state.pitch + m.dy * 0.3));
            }
            broadcast();
        }
    });
    const errorInput = /** @type {HTMLInputElement} */ (root.querySelector('#error'));
    const update = () => {
        root.querySelector('output').textContent = state.coarse
            ? 'Coarsest'
            : `${Number(state.threshold.toFixed(2))} px`;
        root.querySelector('#coarse').textContent = state.coarse ? 'Return to error LOD' : 'Coarsest LOD';
        broadcast();
    };
    errorInput.addEventListener('input', () => {
        state.threshold = 2 ** Number(errorInput.value);
        state.coarse = false;
        update();
    });
    root.querySelector('#colour').addEventListener('change', (event) => {
        state.color = Number(/** @type {HTMLSelectElement} */ (event.target).value);
        update();
    });
    root.querySelector('#coarse').addEventListener('click', () => {
        state.coarse = !state.coarse;
        update();
    });
    root.querySelector('#reset').addEventListener('click', () => {
        Object.assign(state, { yaw: 35, pitch: 22, zoom: 1, pan: [0, 0, 0] });
        update();
    });
    const response = await fetch(`${directory}comparison.json`);
    if (!response.ok) throw new Error('ISS bakes missing: see assets/meshlets/README.md.');
    const report = await response.json();
    frames.forEach(({ pane, key }) => {
        const entry = report.find((entry) => entry.key === key);
        pane.querySelector('.bake').textContent =
            `Coarsest: ${entry.rootTriangles.toLocaleString()} triangles · ${entry.rootMeshlets.toLocaleString()} meshlets`;
    });
} else {
    // Separate full viewports give identical projection without perspective offsets or a
    // shared memory-pressure controller. Only camera and display settings are synchronized.
    const send = (message) => window.parent.postMessage({ channel, ...message }, location.origin);
    try {
        basisInitialize({
            glueUrl: './assets/wasm/basis/basis.wasm.js',
            wasmUrl: './assets/wasm/basis/basis.wasm.wasm',
            fallbackUrl: './assets/wasm/basis/basis.js'
        });
        const device = await createGraphicsDevice(canvas, { deviceTypes: ['webgpu'], antialias: false });
        device.maxPixelRatio = Math.min(window.devicePixelRatio, 2);
        const options = new AppOptions();
        options.graphicsDevice = device;
        options.componentSystems = [CameraComponentSystem, LightComponentSystem, MeshletComponentSystem];
        options.resourceHandlers = [ContainerHandler, TextureHandler];
        const app = new AppBase(canvas);
        app.init(options);
        app.setCanvasFillMode(FILLMODE_FILL_WINDOW);
        app.setCanvasResolution(RESOLUTION_AUTO);
        const asset = new Asset(`ISS ${variant.label}`, 'container', { url: `${directory}${variant.file}` });
        await new Promise((resolve, reject) => {
            new AssetListLoader([asset], app.assets).load((error) => (error ? reject(error) : resolve()));
        });
        const director = app.systems.meshlet.director;
        director.world.poolBytes = 0;
        director.world.initialIndices = 0;
        director.world.initialRecords = 0;
        director.rebuild((world) => world.addStreamedResource(asset.resource.meshlets[0], null, directory));
        const bounds = director.world.worldBounds,
            radius = bounds.halfExtents.length();
        const center = bounds.center.clone(),
            target = new Vec3(),
            position = new Vec3();
        const camera = new Entity('Camera');
        camera.addComponent('camera', {
            clearColor: new Color(0.07, 0.09, 0.13),
            nearClip: radius / 10000,
            farClip: radius * 200,
            fov: 45
        });
        app.root.addChild(camera);
        director.cameraComponent = camera.camera;
        app.scene.ambientLight = new Color(0.4, 0.4, 0.4);
        const light = new Entity('Sun');
        light.addComponent('light', { type: 'directional', intensity: 1.5 });
        light.setEulerAngles(35, 45, 0);
        app.root.addChild(light);
        let state = initialState();
        const apply = () => {
            const yaw = (state.yaw * Math.PI) / 180,
                pitch = (state.pitch * Math.PI) / 180,
                halfFov = Math.PI / 8;
            const fitAngle = Math.min(
                halfFov,
                Math.atan((Math.tan(halfFov) * canvas.clientWidth) / canvas.clientHeight)
            );
            if (state.framing) center.set(...state.framing.center);
            const distance = ((state.framing?.radius ?? radius) / Math.sin(fitAngle)) * state.zoom;
            target.set(center.x + state.pan[0], center.y + state.pan[1], center.z + state.pan[2]);
            position
                .set(Math.sin(yaw) * Math.cos(pitch), Math.sin(pitch), Math.cos(yaw) * Math.cos(pitch))
                .mulScalar(distance)
                .add(target);
            camera.setPosition(position);
            camera.lookAt(target);
            director.dagPixelThreshold = state.coarse ? 1e30 : state.threshold;
            director.world.setColorMode(state.color);
        };
        const resize = () => {
            app.resizeCanvas(window.innerWidth, window.innerHeight);
            apply();
        };
        window.addEventListener('resize', resize);
        window.addEventListener('message', (event) => {
            if (
                event.source !== window.parent ||
                event.origin !== location.origin ||
                event.data?.channel !== channel ||
                event.data.type !== 'state'
            ) {
                return;
            }
            state = event.data.state;
            apply();
        });
        let dragging = false,
            previousX = 0,
            previousY = 0,
            pan = false;
        canvas.addEventListener('pointerdown', (event) => {
            dragging = true;
            previousX = event.clientX;
            previousY = event.clientY;
            pan = event.button !== 0 || event.shiftKey;
            canvas.setPointerCapture(event.pointerId);
        });
        canvas.addEventListener('pointermove', (event) => {
            if (!dragging) return;
            send({
                type: 'input',
                kind: pan ? 'pan' : 'orbit',
                dx: event.clientX - previousX,
                dy: event.clientY - previousY,
                height: canvas.clientHeight
            });
            previousX = event.clientX;
            previousY = event.clientY;
        });
        canvas.addEventListener('lostpointercapture', () => {
            dragging = false;
        });
        canvas.addEventListener('contextmenu', (event) => event.preventDefault());
        canvas.addEventListener(
            'wheel',
            (event) => {
                event.preventDefault();
                send({ type: 'input', kind: 'zoom', delta: event.deltaY });
            },
            { passive: false }
        );
        resize();
        let elapsed = 0;
        app.on('update', (dt) => {
            elapsed += dt;
            if (elapsed < 0.5) return;
            elapsed = 0;
            const view = director.views.get(camera.camera);
            if (view) {
                send({
                    type: 'stats',
                    text: `${view.renderedMeshlets.toLocaleString()} drawn meshlets · ${director.residency?.residentPages ?? 0}/${director.world.totalPages} pages resident`
                });
            }
        });
        window.addEventListener('pagehide', () => app.destroy(), { once: true });
        app.start();
        send({ type: 'ready', radius, center: center.toArray() });
    } catch (error) {
        send({ type: 'error', text: String(error) });
        throw error;
    }
}
