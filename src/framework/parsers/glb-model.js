import { path } from '../../core/path.js';
import { Http } from '../../platform/net/http.js';
import { GlbContainerResource } from './glb-container-resource.js';
import { GlbParser } from './glb-parser.js';

class GlbModelParser {
    constructor(modelHandler) {
        this._device = modelHandler.device;
        this._defaultMaterial = modelHandler.defaultMaterial;
        this._assets = modelHandler.assets;
    }

    canParse(context) {
        return context.ext === 'glb';
    }

    load(url, callback, asset) {
        this.handler.fetch(url, Http.ResponseType.ARRAY_BUFFER, (err, data) => {
            if (err) {
                callback(err);
            } else {
                this.parse(data, callback, asset);
            }
        }, asset);
    }

    parse(data, callback, asset) {
        // magnopus patched
        const loadUrlBase = asset ? path.extractPath(asset.getFileUrl()) : '';
        const originalUrlBase = asset ? path.extractPath(asset.getFileOriginalUrl()) : '';

        GlbParser.parse('filename.glb', loadUrlBase, originalUrlBase, data, this._device, this._assets, asset?.options ?? {}, (err, result) => {
            if (err) {
                callback(err);
            } else {
                const model = GlbContainerResource.createModel(result, this._defaultMaterial);
                result.destroy();
                callback(null, model);
            }
        });
    }
}

export { GlbModelParser };
