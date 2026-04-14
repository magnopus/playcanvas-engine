# URL Resolver

This fork adds an application-level `urlResolver` hook that allows assets to keep a logical path while loading from a different final URL.

This is especially useful for asset systems that do not behave like a filesystem, such as:

- database-backed asset stores
- signed or random S3 URLs
- virtual asset namespaces
- gsplat LOD content where relative references need remapping

## What It Does

The resolver works with a URL pair:

- `original`: the logical asset path
- `load`: the final URL to fetch

For example:

- logical path: `/gsplats/lod-meta.json`
- final fetch URL: `https://signed-bucket-url.example.com/abc123.json`

Relative child references are resolved from `original`, not from `load`.

That means a gsplat file can be loaded from a signed URL while still resolving child paths against a stable logical namespace.

## Application Usage

Pass `urlResolver` into `pc.Application`:

```js
const app = new pc.Application(canvas, {
    graphicsDevice,
    urlResolver: (url, { baseUrl, asset, handler }) => {
        return url;
    }
});
```

The resolver receives:

- `url.load`: the current fetch URL
- `url.original`: the current logical URL
- `baseUrl`: the logical base URL for resolving relative references
- `asset`: the asset being loaded, when available
- `handler`: the resource handler making the request, when available

It can return:

- the original `url` object unchanged
- a new string to replace `load`
- an object with `load` and/or `original`

## Simple Remap Example

This example keeps the top-level gsplat at `/gsplats/lod-meta.json`, but redirects any nested files under `/gsplats/` to load from `/gplats2/`.

```js
const app = new pc.Application(canvas, {
    graphicsDevice,
    urlResolver: (url) => {
        if (url.original === '/gsplats/lod-meta.json') {
            return url;
        }

        if (url.original.startsWith('/gsplats/')) {
            return {
                original: url.original,
                load: url.original.replace('/gsplats/', '/gplats2/')
            };
        }

        return url;
    }
});

app.assets.loadFromUrl('/gsplats/lod-meta.json', 'gsplat', (err, asset) => {
    if (err) {
        console.error(err);
        return;
    }

    console.log('loaded gsplat', asset.resource);
});
```

With that resolver:

- `/gsplats/lod-meta.json` loads from `/gsplats/lod-meta.json`
- `./0_0/meta.json` resolves logically to `/gsplats/0_0/meta.json`
- the resolver rewrites the fetch URL to `/gplats2/0_0/meta.json`

## Signed URL Example

This example keeps a logical namespace in the engine, but resolves final fetches through a lookup table.

```js
const assetMap = new Map([
    ['/gsplats/lod-meta.json', {
        assetCollectionId: 1,
        assetDetailId: 10,
        url: 'https://cdn.example.com/root-lod-meta.json?sig=abc'
    }],
    ['/gsplats/0_0/meta.json', {
        assetCollectionId: 1,
        assetDetailId: 11,
        url: 'https://cdn.example.com/0_0-meta.json?sig=def'
    }],
    ['/gsplats/0_0/chunk.ply', {
        assetCollectionId: 1,
        assetDetailId: 12,
        url: 'https://cdn.example.com/0_0-chunk.ply?sig=ghi'
    }]
]);

const app = new pc.Application(canvas, {
    graphicsDevice,
    urlResolver: (url) => {
        const record = assetMap.get(url.original);
        if (!record) {
            return url;
        }

        return {
            original: url.original,
            load: record.url
        };
    }
});

app.assets.loadFromUrl('/gsplats/lod-meta.json', 'gsplat', (err, asset) => {
    if (err) {
        console.error(err);
        return;
    }

    console.log('loaded from signed URLs', asset.resource);
});
```

## Manual Asset Example

If you create an asset manually, you can provide both the final load URL and the logical URL:

```js
const asset = new pc.Asset('lod', 'gsplat', {
    url: 'https://cdn.example.com/root-lod-meta.json?sig=abc',
    originalUrl: '/gsplats/lod-meta.json'
});

app.assets.add(asset);
app.assets.load(asset);
```

This is useful when:

- the root asset already has a signed URL
- relative gsplat references should still resolve from a logical path
- you want the resolver to operate on stable keys instead of transient URLs

## Notes

- `originalUrl` is optional for normal assets, but recommended when the fetch URL is not stable.
- gsplat nested references now preserve both logical and load URLs.
- top-level loads also pass through the resolver.
- the resolver can be used for non-gsplat assets too, but gsplat benefits the most because of nested relative references.
