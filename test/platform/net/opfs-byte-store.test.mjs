import { expect } from 'chai';

import { OpfsPackWorker } from '../../../src/platform/net/opfs-byte-store.js';

// An in-memory stand-in for the parts of the Origin Private File System the worker uses.

class MemSyncHandle {
    constructor(file) {
        this.file = file;
        file.locked = true;
    }

    getSize() {
        return this.file.size;
    }

    read(view, { at }) {
        const n = Math.max(0, Math.min(view.byteLength, this.file.size - at));
        view.set(this.file.data.subarray(at, at + n));
        return n;
    }

    write(view, { at }) {
        const end = at + view.byteLength;
        if (end > this.file.data.byteLength) {
            const grown = new Uint8Array(Math.max(end, this.file.data.byteLength * 2));
            grown.set(this.file.data.subarray(0, this.file.size));
            this.file.data = grown;
        }
        this.file.data.set(view, at);
        this.file.size = Math.max(this.file.size, end);
        return view.byteLength;
    }

    truncate(size) {
        this.file.size = size;
    }

    close() {
        this.file.locked = false;
    }
}

class MemFile {
    kind = 'file';

    data = new Uint8Array(0);

    size = 0;

    locked = false;

    async createSyncAccessHandle() {
        if (this.locked) throw Object.assign(new Error('locked'), { name: 'NoModificationAllowedError' });
        return new MemSyncHandle(this);
    }
}

class MemDir {
    kind = 'directory';

    children = new Map();

    async getDirectoryHandle(name, { create = false } = {}) {
        let child = this.children.get(name);
        if (!child) {
            if (!create) throw Object.assign(new Error(name), { name: 'NotFoundError' });
            this.children.set(name, child = new MemDir());
        }
        return child;
    }

    async getFileHandle(name, { create = false } = {}) {
        let child = this.children.get(name);
        if (!child) {
            if (!create) throw Object.assign(new Error(name), { name: 'NotFoundError' });
            this.children.set(name, child = new MemFile());
        }
        return child;
    }

    async removeEntry(name) {
        this.children.delete(name);
    }

    async* entries() {
        for (const entry of [...this.children]) yield entry;
    }
}

const DIR = 'cache';
const bytes = (length, seed = 0) => Uint8Array.from({ length }, (_, i) => (seed + i * 7) & 0xff);
const settle = () => new Promise((resolve) => {
    setTimeout(resolve, 20);
});

describe('OpfsPackWorker', function () {
    let root;
    let savedNavigator;
    let savedFileHandle;

    beforeEach(function () {
        root = new MemDir();
        savedNavigator = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
        savedFileHandle = Object.getOwnPropertyDescriptor(globalThis, 'FileSystemFileHandle');
        Object.defineProperty(globalThis, 'navigator', {
            value: { storage: { getDirectory: async () => root } },
            configurable: true
        });
        // the worker checks for synchronous access handle support on this prototype
        Object.defineProperty(globalThis, 'FileSystemFileHandle', { value: MemFile, configurable: true });
    });

    afterEach(function () {
        if (savedNavigator) Object.defineProperty(globalThis, 'navigator', savedNavigator); else delete globalThis.navigator;
        if (savedFileHandle) Object.defineProperty(globalThis, 'FileSystemFileHandle', savedFileHandle); else delete globalThis.FileSystemFileHandle;
    });

    /**
     * Starts a worker on the in-memory file system and opens the directory.
     *
     * @param {number} [packBytes] - Pack size.
     * @param {number} [compactMinDead] - Dead bytes below which nothing is compacted.
     * @returns {Promise<{ call: (msg: object) => Promise<any> }>} Sends an operation.
     */
    const open = async (packBytes = 1 << 20, compactMinDead = 1 << 20) => {
        const calls = new Map();
        let nextId = 1;
        const scope = {
            postMessage(msg) {
                const call = calls.get(msg.id);
                calls.delete(msg.id);
                if (msg.error) call.reject(Object.assign(new Error(msg.error.message), { name: msg.error.name }));
                else call.resolve(msg.result);
            }
        };
        OpfsPackWorker(scope, packBytes, compactMinDead);
        const call = msg => new Promise((resolve, reject) => {
            const id = nextId++;
            calls.set(id, { resolve, reject });
            scope.onmessage({ data: { ...msg, id } });
        });
        await call({ op: 'open', directoryName: DIR });
        return { call };
    };

    // a page closing releases its handles
    const closePage = () => {
        for (const file of root.children.get(DIR).children.values()) file.locked = false;
    };

    const packFiles = () => [...root.children.get(DIR).children.keys()].sort();

    it('reads back what it wrote, whole and in part, and lists and removes it', async function () {
        const { call } = await open();
        await call({ op: 'write', name: 'a', data: bytes(100, 1) });
        await call({ op: 'write', name: 'b', data: bytes(50, 2) });
        expect(new Uint8Array(await call({ op: 'read', name: 'a' }))).to.deep.equal(bytes(100, 1));
        expect(new Uint8Array(await call({ op: 'readRange', name: 'a', offset: 10, length: 20 }))).to.deep.equal(bytes(100, 1).slice(10, 30));
        expect(await call({ op: 'readRange', name: 'a', offset: 90, length: 20 })).to.equal(null);
        expect((await call({ op: 'list' })).sort()).to.deep.equal(['a', 'b']);
        await call({ op: 'remove', name: 'a' });
        expect(await call({ op: 'read', name: 'a' })).to.equal(null);
        expect(await call({ op: 'list' })).to.deep.equal(['b']);
    });

    it('hands a lent buffer straight back, ahead of the writes queued before it, and stores it', async function () {
        const calls = new Map();
        const order = [];
        const scope = {
            postMessage(msg) {
                order.push(msg.id);
                calls.get(msg.id)?.(msg.result);
            }
        };
        OpfsPackWorker(scope, 1 << 20, 1 << 20);
        const send = (msg, id) => new Promise((resolve) => {
            calls.set(id, resolve);
            scope.onmessage({ data: { ...msg, id } });
        });
        await send({ op: 'open', directoryName: DIR }, 1);
        const queued = send({ op: 'write', name: 'q', data: bytes(10, 1) }, 2);
        const returned = new Promise((resolve) => {
            calls.set(4, resolve);
        });
        const stored = send({ op: 'write', name: 'lent', data: bytes(64, 5).buffer, returnId: 4 }, 3);
        expect(new Uint8Array(await returned)).to.deep.equal(bytes(64, 5));
        await Promise.all([queued, stored]);
        expect(order.slice(1)).to.deep.equal([4, 2, 3]);
        expect(new Uint8Array(await send({ op: 'read', name: 'lent' }, 5))).to.deep.equal(bytes(64, 5));
    });

    it('finds its values again in the next session, latest write winning and removals kept', async function () {
        let { call } = await open();
        await call({ op: 'write', name: 'a', data: bytes(10, 1) });
        await call({ op: 'write', name: 'a', data: bytes(20, 2) });
        await call({ op: 'write', name: 'b', data: bytes(30, 3) });
        await call({ op: 'remove', name: 'b' });
        await call({ op: 'write', name: 'c', data: new Uint8Array(0) });
        closePage();

        ({ call } = await open());
        expect(new Uint8Array(await call({ op: 'read', name: 'a' }))).to.deep.equal(bytes(20, 2));
        expect(await call({ op: 'read', name: 'b' })).to.equal(null);
        expect((await call({ op: 'read', name: 'c' })).byteLength).to.equal(0);
        expect((await call({ op: 'list' })).sort()).to.deep.equal(['a', 'c']);
    });

    it('drops a record cut short at the end of a pack, keeping the ones before it', async function () {
        let { call } = await open();
        await call({ op: 'write', name: 'a', data: bytes(64, 1) });
        await call({ op: 'write', name: 'b', data: bytes(64, 2) });
        closePage();
        // a crash part way through writing b's data
        root.children.get(DIR).children.get('pack-0.dat').size -= 10;

        ({ call } = await open());
        expect(new Uint8Array(await call({ op: 'read', name: 'a' }))).to.deep.equal(bytes(64, 1));
        expect(await call({ op: 'read', name: 'b' })).to.equal(null);
        // and appends after the truncation point
        await call({ op: 'write', name: 'b', data: bytes(64, 3) });
        closePage();
        ({ call } = await open());
        expect(new Uint8Array(await call({ op: 'read', name: 'b' }))).to.deep.equal(bytes(64, 3));
    });

    it('starts a new pack when one is full, and compacts the oldest once enough is dead', async function () {
        const { call } = await open(1024, 0);
        for (let i = 0; i < 6; i++) {
            await call({ op: 'write', name: `v${i}`, data: bytes(300, i) });
        }
        expect(packFiles().length).to.be.greaterThan(1);
        // replace most of them: the first packs become mostly dead
        for (let i = 0; i < 5; i++) {
            await call({ op: 'write', name: `v${i}`, data: bytes(300, 10 + i) });
        }
        await settle();
        await call({ op: 'list' });
        expect(packFiles()).to.not.include('pack-0.dat');
        for (let i = 0; i < 6; i++) {
            const expected = i < 5 ? bytes(300, 10 + i) : bytes(300, i);
            expect(new Uint8Array(await call({ op: 'read', name: `v${i}` }))).to.deep.equal(expected);
        }
    });

    it('does not resurrect a removed value when the pack holding its tombstone is compacted', async function () {
        let { call } = await open(512, 0);
        await call({ op: 'write', name: 'gone', data: bytes(200, 1) });
        await call({ op: 'remove', name: 'gone' });
        for (let i = 0; i < 8; i++) {
            await call({ op: 'write', name: 'churn', data: bytes(200, i) });
        }
        await settle();
        await call({ op: 'list' });
        closePage();
        ({ call } = await open(512, 0));
        expect(await call({ op: 'read', name: 'gone' })).to.equal(null);
        expect(new Uint8Array(await call({ op: 'read', name: 'churn' }))).to.deep.equal(bytes(200, 7));
    });

    it('fails to open while another page holds the packs', async function () {
        await open();
        let error = null;
        try {
            await open();
        } catch (err) {
            error = err;
        }
        expect(error?.name).to.equal('NoModificationAllowedError');
    });

    it('clears everything, and removes files that are not its packs when it opens', async function () {
        const dir = await root.getDirectoryHandle(DIR, { create: true });
        await dir.getFileHandle('stray-value', { create: true });
        const { call } = await open();
        await settle();
        expect(packFiles()).to.deep.equal(['pack-0.dat']);
        await call({ op: 'write', name: 'a', data: bytes(10) });
        await call({ op: 'clear' });
        expect(await call({ op: 'list' })).to.deep.equal([]);
        expect(await call({ op: 'read', name: 'a' })).to.equal(null);
    });
});
