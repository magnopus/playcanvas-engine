import { expect } from 'chai';
import nise from 'nise';
import { restore, spy } from 'sinon';

import { ByteCache } from '../../../src/platform/net/byte-cache.js';
import { getFetchCredentials, http, Http } from '../../../src/platform/net/http.js';
import { MemoryByteStore } from '../../../src/platform/net/memory-byte-store.js';
import { jsdomSetup, jsdomTeardown } from '../../jsdom.mjs';

describe('Http', function () {
    let retryDelay;

    beforeEach(function () {
        // Set up a JSDOM document so global.XMLHttpRequest exists and root-relative
        // request URLs resolve against the fixture server's origin, independent of
        // whether any other test file ran jsdomSetup() first.
        jsdomSetup();

        retryDelay = Http.retryDelay;
        Http.retryDelay = 1;
    });

    afterEach(function () {
        Http.retryDelay = retryDelay;
        restore();
        jsdomTeardown();
    });

    describe('#get()', function () {

        it('returns resource', (done) => {
            http.get('/test/assets/test.json', (err, data) => {
                expect(err).to.equal(null);
                expect(data).to.deep.equal({
                    a: 1,
                    b: true,
                    c: 'hello world'
                });
                done();
            });
        });

        it('does not retry if retry is false', (done) => {
            spy(http, 'request');
            http.get('/someurl.json', (err, data) => {
                expect(err).to.equal(404);
                expect(http.request.callCount).to.equal(1);
                done();
            });
        });

        it('retries resource and returns 404 in the end if not found', (done) => {
            spy(http, 'request');
            http.get('/someurl.json', {
                retry: true,
                maxRetries: 2
            }, (err) => {
                expect(err).to.equal(404);
                expect(http.request.callCount).to.equal(3);
                done();
            });
        });

        it('retries resource 5 times by default', (done) => {
            spy(http, 'request');
            http.get('/someurl.json', {
                retry: true
            }, (err) => {
                expect(http.request.callCount).to.equal(6);
                done();
            });
        });

        it('retries resource and returns result if eventually found', function (done) {
            spy(http, 'request');

            let requests = 0;
            const xhr = nise.fakeXhr.useFakeXMLHttpRequest();

            // Store original XMLHttpRequest
            const originalXHR = global.XMLHttpRequest;

            // Replace JSDOM's XMLHttpRequest with Sinon's fake
            global.XMLHttpRequest = xhr;

            xhr.onCreate = function (xhr) {
                setTimeout(function () {
                    try {
                        if (++requests === 3) {
                            xhr.respond(200, { ContentType: 'application/json' }, JSON.stringify({ test: 'value' }));
                        } else {
                            xhr.error();
                        }
                    } catch (err) {
                        done(new Error(`${err.message}\n${err.stack}`));
                    }
                });
            };

            http.get('/someurl.json', {
                retry: true,
                maxRetries: 2
            }, function (err, data) {
                expect(err).to.equal(null);
                expect(http.request.callCount).to.equal(3);
                expect(data).to.deep.equal({ test: 'value' });

                // Restore original XMLHttpRequest
                global.XMLHttpRequest = originalXHR;

                done();
            });
        });

        it('status 0 returns "Network error"', function (done) {
            const xhr = nise.fakeXhr.useFakeXMLHttpRequest();
            let isDone = false;

            // Store original XMLHttpRequest
            const originalXHR = global.XMLHttpRequest;

            // Replace JSDOM's XMLHttpRequest with Sinon's fake
            global.XMLHttpRequest = xhr;

            xhr.onCreate = function (xhr) {
                setTimeout(function () {
                    try {
                        xhr.error();
                    } catch (err) {
                        if (!isDone) {
                            isDone = true;
                            done(new Error(`${err.message}\n${err.stack}`));
                        }
                    }
                });
            };

            http.get('/someurl.json', function (err, data) {
                if (!isDone) {
                    isDone = true;
                    expect(err).to.equal('Network error');

                    // Restore original XMLHttpRequest
                    global.XMLHttpRequest = originalXHR;

                    done();
                }
            });
        });

        describe('status codes', function () {
            let originalXHR;

            // Respond to every request with the given status, headers and body
            const respondWith = (status, headers, body) => {
                const fakeXhr = nise.fakeXhr.useFakeXMLHttpRequest();
                global.XMLHttpRequest = fakeXhr;
                fakeXhr.onCreate = (xhr) => {
                    setTimeout(() => xhr.respond(status, headers, body));
                };
            };

            beforeEach(function () {
                originalXHR = global.XMLHttpRequest;
            });

            afterEach(function () {
                global.XMLHttpRequest = originalXHR;
            });

            it('treats 202 Accepted as success', function (done) {
                respondWith(202, { 'Content-Type': 'application/json' }, JSON.stringify({ queued: true }));
                http.get('/someurl.json', (err, data) => {
                    expect(err).to.equal(null);
                    expect(data).to.deep.equal({ queued: true });
                    done();
                });
            });

            for (const [status, name] of [[204, 'No Content'], [205, 'Reset Content']]) {
                it(`treats ${status} ${name} as success with a null response`, function (done) {
                    // The URL has no .json extension, so the JSON content type makes Http parse the
                    // body text itself (a .json URL is read through the XHR's own json response type)
                    respondWith(status, { 'Content-Type': 'application/json' }, '');
                    http.get('/someurl', (err, data) => {
                        expect(err).to.equal(null);
                        expect(data).to.equal(null);
                        done();
                    });
                });
            }

            it('does not retry a 204 No Content response', function (done) {
                spy(http, 'request');
                respondWith(204, {}, '');
                http.get('/someurl', { retry: true }, (err, data) => {
                    expect(err).to.equal(null);
                    expect(data).to.equal(null);
                    expect(http.request.callCount).to.equal(1);
                    done();
                });
            });

            it('treats 404 Not Found as an error', function (done) {
                respondWith(404, {}, '');
                http.get('/someurl.json', (err, data) => {
                    expect(err).to.equal(404);
                    expect(data).to.equal(null);
                    done();
                });
            });

        });

        describe('status 0', function () {
            let originalXHR;

            // A minimal XMLHttpRequest that completes with status 0 and the given response URL. nise
            // can't be used here as it fires an error event for every status 0 response, whereas a
            // browser completes a successful load from a non-http scheme with a load event
            const respondWithStatus0 = (responseURL, body) => {
                global.XMLHttpRequest = class {
                    readyState = 0;

                    status = 0;

                    responseType = '';

                    open() {
                        this.readyState = 1;
                    }

                    setRequestHeader() {
                    }

                    getResponseHeader() {
                        return null;
                    }

                    send() {
                        setTimeout(() => {
                            this.readyState = 4;
                            this.responseURL = responseURL;
                            this.responseText = body;
                            this.response = this.responseType === 'json' && body ? JSON.parse(body) : body;
                            this.onreadystatechange();
                        });
                    }
                };
            };

            beforeEach(function () {
                originalXHR = global.XMLHttpRequest;
            });

            afterEach(function () {
                global.XMLHttpRequest = originalXHR;
            });

            for (const responseURL of [
                'file:///app/test.json',
                'file://localhost/app/test.json',
                'ionic://localhost/test.json',
                'capacitor://localhost/test.json',
                'app://bundle/test.json'
            ]) {
                it(`treats a response from ${responseURL} as success`, function (done) {
                    respondWithStatus0(responseURL, JSON.stringify({ a: 1 }));
                    http.get('/test.json', (err, data) => {
                        expect(err).to.equal(null);
                        expect(data).to.deep.equal({ a: 1 });
                        done();
                    });
                });
            }

            for (const responseURL of ['http://example.com/test.json', 'https://example.com/test.json', '']) {
                it(`treats a response from ${responseURL || 'an empty URL'} as a network error`, function (done) {
                    respondWithStatus0(responseURL, '');
                    http.get('/test.json', (err, data) => {
                        expect(err).to.equal('Network error');
                        expect(data).to.equal(null);
                        done();
                    });
                });
            }

            it('retries a status 0 response from an https URL', function (done) {
                spy(http, 'request');
                respondWithStatus0('https://example.com/test.json', '');
                http.get('/test.json', { retry: true, maxRetries: 2 }, (err, data) => {
                    expect(err).to.equal('Network error');
                    expect(http.request.callCount).to.equal(3);
                    done();
                });
            });

        });

    });

    describe('#maxConcurrentRequests', function () {
        let originalXHR;
        let created;

        beforeEach(function () {
            created = [];
            originalXHR = global.XMLHttpRequest;
            const fakeXhr = nise.fakeXhr.useFakeXMLHttpRequest();
            global.XMLHttpRequest = fakeXhr;
            // collect every created XHR but don't auto-respond, so requests stay in flight
            fakeXhr.onCreate = (xhr) => {
                created.push(xhr);
            };

            // start each test from a known, clean throttle state on the shared singleton
            http.maxConcurrentRequests = 128;
            http._activeRequests = 0;
            http._sendQueue.length = 0;
        });

        afterEach(function () {
            global.XMLHttpRequest = originalXHR;
            // restore defaults so state doesn't leak into other test files
            http.maxConcurrentRequests = 128;
            http._activeRequests = 0;
            http._sendQueue.length = 0;
        });

        const respond = (xhr) => {
            xhr.respond(200, { 'Content-Type': 'application/json' }, '{}');
        };

        it('limits in-flight requests and queues the rest', function () {
            http.maxConcurrentRequests = 2;

            let completed = 0;
            const onDone = () => {
                completed++;
            };
            for (let i = 0; i < 4; i++) {
                http.get(`/url${i}.json`, onDone);
            }

            // all 4 XHRs are created (cheap), but only 2 are sent; the other 2 are queued
            expect(created.length).to.equal(4);
            expect(http._activeRequests).to.equal(2);
            expect(http._sendQueue.length).to.equal(2);

            // completing the 2 in-flight requests dispatches the 2 queued ones
            respond(created[0]);
            respond(created[1]);
            expect(http._activeRequests).to.equal(2);
            expect(http._sendQueue.length).to.equal(0);
            expect(completed).to.equal(2);

            // completing those drains everything
            respond(created[2]);
            respond(created[3]);
            expect(http._activeRequests).to.equal(0);
            expect(completed).to.equal(4);
        });

        it('does not throttle when set to 0', function () {
            http.maxConcurrentRequests = 0;

            const noop = () => {};
            for (let i = 0; i < 5; i++) {
                http.get(`/url${i}.json`, noop);
            }

            // everything is sent immediately and nothing is queued or slot-accounted
            expect(created.length).to.equal(5);
            expect(http._sendQueue.length).to.equal(0);
            expect(http._activeRequests).to.equal(0);
        });

        it('dispatches queued requests when the limit is raised', function () {
            http.maxConcurrentRequests = 1;

            const noop = () => {};
            for (let i = 0; i < 3; i++) {
                http.get(`/url${i}.json`, noop);
            }
            expect(http._activeRequests).to.equal(1);
            expect(http._sendQueue.length).to.equal(2);

            // raising the limit immediately pumps the queued requests
            http.maxConcurrentRequests = 3;
            expect(http._activeRequests).to.equal(3);
            expect(http._sendQueue.length).to.equal(0);
        });

        it('tracks slots per request, not per shared options object', function () {
            http.maxConcurrentRequests = 2;

            // a single options object reused across concurrent requests must not corrupt slot
            // accounting (state is keyed on the xhr, not on the caller's options object)
            const shared = {};
            const onDone = () => {};
            for (let i = 0; i < 4; i++) {
                http.get(`/url${i}.json`, shared, onDone);
            }

            expect(http._activeRequests).to.equal(2);
            expect(http._sendQueue.length).to.equal(2);

            // completing all four returns the active count cleanly to 0 (no leak)
            created.forEach(respond);
            expect(http._activeRequests).to.equal(0);
            expect(http._sendQueue.length).to.equal(0);
        });

    });

    describe('#byteCache', function () {
        let cache;

        beforeEach(async function () {
            cache = new ByteCache(new MemoryByteStore(), { flushDelay: 1e6 });
            await cache.ready;
            http.byteCache = cache;
            spy(http, '_acquire');
        });

        afterEach(function () {
            http.byteCache = null;
        });

        const get = (url, options = {}) => new Promise((resolve, reject) => {
            http.get(url, options, (err, data) => (err ? reject(err) : resolve(data)));
        });

        // the response is stored asynchronously after the callback runs
        const settle = () => new Promise((resolve) => {
            setTimeout(resolve, 10);
        });

        it('serves a repeated binary request from the cache', async function () {
            const first = await get('/test/assets/test.bin', { responseType: Http.ResponseType.ARRAY_BUFFER });
            await settle();
            const second = await get('/test/assets/test.bin', { responseType: Http.ResponseType.ARRAY_BUFFER });
            expect(http._acquire.callCount).to.equal(1);
            expect(cache.hits).to.equal(1);
            expect(new Uint8Array(second)).to.deep.equal(new Uint8Array(first));
        });

        it('serves a repeated JSON request from the cache, parsed', async function () {
            await get('/test/assets/test.json');
            await settle();
            const data = await get('/test/assets/test.json');
            expect(http._acquire.callCount).to.equal(1);
            expect(data).to.deep.equal({ a: 1, b: true, c: 'hello world' });
        });

        it('bypasses the cache when the request opts out', async function () {
            await get('/test/assets/test.json');
            await settle();
            await get('/test/assets/test.json', { byteCache: false });
            expect(http._acquire.callCount).to.equal(2);
        });

        it('waits for a reopened cache\'s index instead of refetching what it holds', async function () {
            await get('/test/assets/test.bin', { responseType: Http.ResponseType.ARRAY_BUFFER });
            await settle();
            await cache.flush();

            // a new session over the same store: its index is still loading when the request is made
            const reopened = new ByteCache(cache.store, { flushDelay: 1e6 });
            http.byteCache = reopened;
            const data = await get('/test/assets/test.bin', { responseType: Http.ResponseType.ARRAY_BUFFER });
            expect(http._acquire.callCount, 'only the first session fetched').to.equal(1);
            expect(reopened.hits).to.equal(1);
            expect(data.byteLength).to.be.above(0);
        });

        it('does not store a failed response', async function () {
            await get('/someurl.json').catch(() => {});
            await settle();
            expect(cache.entryCount).to.equal(0);
        });
    });
    describe('#withCredentials', function () {

        afterEach(function () {
            http.withCredentials = false;
        });

        it('maps to the fetch credentials mode of the loaders that bypass XHR', function () {
            // 'same-origin' is fetch's own default, so the flag being off changes nothing
            expect(getFetchCredentials()).to.equal('same-origin');

            http.withCredentials = true;
            expect(getFetchCredentials()).to.equal('include');
        });

    });

});
