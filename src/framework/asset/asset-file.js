/**
 * Wraps a source of asset data.
 *
 * @ignore
 */
class AssetFile {
    constructor(url = '', filename = '', hash = null, size = null, opt = null, contents = null, originalUrl = null) {
        this.url = url;
        this.filename = filename;
        this.hash = hash;
        this.size = size;
        this.opt = opt;
        this.contents = contents;
        // magnopus patched
        this.originalUrl = originalUrl;
    }

    // Compare this AssetFile with another. Returns true if they have the same data
    // and false otherwise.
    equals(other) {
        return this.url === other.url &&
            this.filename === other.filename &&
            this.hash === other.hash &&
            this.size === other.size &&
            this.opt === other.opt &&
            this.contents === other.contents &&
            // magnopus patched
            this.originalUrl === other.originalUrl;
    }
}

export { AssetFile };
