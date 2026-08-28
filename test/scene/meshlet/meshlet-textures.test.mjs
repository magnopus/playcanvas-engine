import { expect } from 'chai';

import { NullGraphicsDevice } from '../../../src/platform/graphics/null/null-graphics-device.js';
import { MESHLET_TEX_NO_MINLOD } from '../../../src/scene/meshlet/constants.js';
import { MeshletTextures } from '../../../src/scene/meshlet/textures/meshlet-textures.js';
import { jsdomSetup, jsdomTeardown } from '../../jsdom.mjs';

// A texture manifest array: `layers` entries of `mips` mip files, tail starting at tailMip.
// Byte ranges are fabricated - nothing is fetched (fetch is stubbed to fail) and the fine pool
// is fabricated directly on the family, so these tests exercise the CPU-side bookkeeping only:
// family mapping, residency words, demand -> slot acquisition, LRU protection and eviction.
const makeArray = (name, size, layers, tailMip, extra = {}) => {
    const mips = Math.round(Math.log2(size)) + 1;
    const mipRanges = [];
    for (let l = 0; l < layers; l++) {
        mipRanges.push(Array.from({ length: mips }, (_, m) => [1000 * (l * mips + m), 100]));
    }
    return {
        name,
        width: size,
        height: size,
        containerVersion: 2,
        tailMip,
        container: { uri: `${name}.bin` },
        tail: { byteOffset: 0, byteLength: 100 },
        layers: Array.from({ length: layers }, () => Array.from({ length: mips }, () => ({}))),
        mipRanges,
        ...extra
    };
};

describe('MeshletTextures', function () {

    /** @type {NullGraphicsDevice} */
    let device;
    let savedFetch;
    let savedWarn;

    beforeEach(function () {
        jsdomSetup();
        device = new NullGraphicsDevice(document.createElement('canvas'));
        // the null device has no storage buffers; the residency buffer only needs write()
        device.createBufferImpl = () => ({
            allocate() {},
            write() {},
            read() {
                return Promise.resolve();
            },
            clear() {},
            destroy() {},
            loseContext() {},
            buffer: null
        });
        savedFetch = globalThis.fetch;
        globalThis.fetch = () => Promise.reject(new Error('no network in tests'));
        savedWarn = console.warn;
        console.warn = () => {};
    });

    afterEach(function () {
        globalThis.fetch = savedFetch;
        console.warn = savedWarn;
        device.destroy();
        device = null;
        jsdomTeardown();
    });

    describe('addResource', function () {

        it('maps arrays to families by name prefix, then by type, and returns each resource base', function () {
            const textures = new MeshletTextures(device);
            const base0 = textures.addResource({ arrays: [
                makeArray('srgb_0', 256, 2, 0),
                makeArray('normal_0', 256, 1, 0),
                makeArray('linear_0', 256, 1, 0),
                makeArray('untyped', 256, 1, 0, { type: 'normal' })
            ] }, 'assets/a');
            const base1 = textures.addResource({ arrays: [makeArray('srgba_0', 256, 3, 0)] }, 'assets/b/');
            expect(base0).to.equal(0);
            expect(base1).to.equal(5);
            expect(textures.textureCount).to.equal(8);
            expect(textures.textures.map(t => t.family)).to.deep.equal([0, 0, 2, 3, 2, 1, 1, 1]);
            expect(textures.textures[0].source.baseUrl).to.equal('assets/a/');
            textures.destroy();
        });

        it('skips arrays whose container version it does not understand', function () {
            const textures = new MeshletTextures(device);
            textures.addResource({ arrays: [makeArray('srgb_0', 256, 2, 0, { containerVersion: 9 })] }, 'x');
            expect(textures.textureCount).to.equal(0);
            textures.destroy();
        });
    });

    describe('finalize', function () {

        it('assigns tail layers per family, size biases against the family slot size, and empty residency', function () {
            const textures = new MeshletTextures(device);
            textures.addResource({ arrays: [
                makeArray('srgb_big', 1024, 1, 2),    // tail top 256 -> two fine mips above it
                makeArray('srgb_small', 256, 2, 0),   // fits in the tail entirely
                makeArray('normal_0', 512, 1, 1)
            ] }, 'x');
            textures.finalize();

            const [big, small0, small1, normal] = textures.textures;
            expect([big.tailLayer, small0.tailLayer, small1.tailLayer, normal.tailLayer]).to.deep.equal([0, 1, 2, 0]);
            expect(big.tailStart, 'source mip where the tail begins').to.equal(2);
            expect(small0.tailStart).to.equal(0);
            expect(big.sizeBias).to.equal(0);
            expect(small0.sizeBias, 'log2(1024 / 256)').to.equal(2);

            const srgb = textures.families[0];
            expect(srgb.layerCount).to.equal(3);
            expect(srgb.size).to.equal(256);
            expect(srgb.slotSize).to.equal(1024);
            expect(srgb.slotLevels).to.equal(2);
            expect(srgb.fineDemandTex).to.equal(1);
            expect(srgb.fine, 'the fine pool waits for the first transcode').to.equal(null);

            const r = textures.residencyCpu;
            expect(r[0] & 0xFFFF, 'no fine slot').to.equal(0xFFFF);
            expect((r[0] >> 16) & 0xFF, 'family').to.equal(0);
            expect((r[1] >> 16) & 0xFFFF, 'tail layer').to.equal(0);
            expect(r[1] & 0xFFFF, 'nothing resident yet').to.equal(MESHLET_TEX_NO_MINLOD);
            expect(r[2 * 2] >>> 24, 'size bias of the small texture').to.equal(2);
            expect(r[2 * 3 + 1] & 0xFFFF).to.equal(MESHLET_TEX_NO_MINLOD);
            textures.destroy();
        });
    });

    describe('processMarks', function () {

        // three 1024px srgb textures (two fine mips above their 256px tails) sharing a two-slot
        // fine pool, one material row per texture (slot 0 of the row)
        const setup = () => {
            const textures = new MeshletTextures(device);
            textures.addResource({ arrays: [makeArray('srgb_0', 1024, 3, 2)] }, 'x');
            textures.finalize();
            const rowTex = new Int32Array([0, -1, -1, -1, 1, -1, -1, -1, 2, -1, -1, -1]);
            textures.setMaterialSlotMap(rowTex);
            // fabricate the fine pool the first transcode would have created
            const fam = textures.families[0];
            fam.fine = { destroy() {} };
            fam.slotCount = 2;
            fam.slotBytes = 1000;
            fam.slotTex = new Int32Array([-1, -1]);
            fam.slotLastUsed = new Uint32Array(2);
            fam.freeSlots = [1, 0];
            // never resolve fetches; the bookkeeping is what is under test
            textures._sources[0].fetchRegion = () => new Promise(() => {});
            // tails are "resident" so minLod bookkeeping is live
            for (let i = 0; i < 3; i++) textures.residencyCpu[i * 2 + 1] = 2 | (i << 16);
            textures._dirty = false;
            return { textures, fam };
        };

        // q = 16 * log2(screenPx / uvExtent): 160 asks for mip 0 of a 1024px texture, 32 for mip 8
        const FINE = 160;
        const COARSE = 32;

        it('acquires a free slot, points the residency at it and queues the missing fine mips', function () {
            const { textures, fam } = setup();
            textures.processMarks(new Uint32Array([FINE, 0, 0]));
            const tex = textures.textures[0];
            expect(tex.slot).to.equal(0);
            expect(fam.slotTex[0]).to.equal(0);
            expect(fam.slotLastUsed[0]).to.equal(1);
            expect(textures.residencyCpu[0] & 0xFFFF).to.equal(0);
            expect(textures._dirty).to.equal(true);
            expect(textures._inFlight.size, 'mips 1 and 0 requested, coarse to fine').to.equal(2);
            expect(textures.wantedAboveTail).to.equal(1);
            expect(textures.slotDenials).to.equal(0);
            textures.destroy();
        });

        it('leaves a texture on its tail when the demanded mip is within the tail', function () {
            const { textures } = setup();
            textures.processMarks(new Uint32Array([COARSE, 0, 0]));
            expect(textures.textures[0].slot).to.equal(-1);
            expect(textures.wantedAboveTail).to.equal(0);
            expect(textures._inFlight.size).to.equal(0);
            textures.destroy();
        });

        it('denies a request when every slot holder is wanted this frame', function () {
            const { textures, fam } = setup();
            textures.processMarks(new Uint32Array([FINE, FINE, FINE]));
            expect(textures.textures.map(t => t.slot)).to.deep.equal([0, 1, -1]);
            expect(textures.slotDenials).to.equal(1);
            expect(textures.fineEvictions, 'protected holders are never evicted').to.equal(0);
            expect(Array.from(fam.slotTex)).to.deep.equal([0, 1]);
            textures.destroy();
        });

        it('evicts the least recently wanted holder, and hands its slot out only the next frame', function () {
            const { textures, fam } = setup();
            textures.processMarks(new Uint32Array([FINE, FINE, 0]));   // frame 1: 0 and 1 hold the slots
            textures.processMarks(new Uint32Array([0, FINE, 0]));      // frame 2: only 1 is wanted
            textures.processMarks(new Uint32Array([0, FINE, FINE]));   // frame 3: 2 wants a slot; 0 is the LRU

            const [a, b, c] = textures.textures;
            expect(a.slot, 'the LRU holder lost its slot').to.equal(-1);
            expect(a.fineMask).to.equal(0);
            expect(textures.residencyCpu[0] & 0xFFFF, 'its residency points back at the tail').to.equal(0xFFFF);
            expect(textures.residencyCpu[1] & 0xFFFF, 'minLod falls back to the tail start').to.equal(2);
            expect(b.slot).to.equal(1);
            expect(textures.fineEvictions).to.equal(1);
            expect(fam.pendingFree, 'the freed slot waits a frame for the residency rewrite').to.deep.equal([0]);
            expect(c.slot, 'the requester is denied this frame').to.equal(-1);
            expect(textures.slotDenials).to.equal(1);

            textures.processMarks(new Uint32Array([0, FINE, FINE]));   // frame 4: the slot is free now
            expect(c.slot).to.equal(0);
            expect(Array.from(fam.slotTex), 'exactly one owner per slot').to.deep.equal([2, 1]);
            expect(fam.pendingFree).to.deep.equal([]);
            expect(fam.freeSlots).to.deep.equal([]);
            textures.destroy();
        });

        it('raises the mip bias on eviction churn and decays it when the pool is quiet', function () {
            const { textures } = setup();
            textures.fineEvictions = 10;
            textures._lastEvictions = 0;
            textures.processMarks(new Uint32Array([0, 0, 0]));
            expect(textures.mipBias).to.equal(0.5);
            textures.processMarks(new Uint32Array([0, 0, 0]));
            expect(textures.mipBias).to.be.closeTo(0.45, 1e-9);
            textures.mipBias = 7.8;
            textures.fineEvictions = 30;
            textures.processMarks(new Uint32Array([0, 0, 0]));
            expect(textures.mipBias, 'capped').to.equal(8);
            textures.mipBias = 0.02;
            textures.processMarks(new Uint32Array([0, 0, 0]));
            expect(textures.mipBias, 'floored').to.equal(0);
            textures.destroy();
        });
    });
});
