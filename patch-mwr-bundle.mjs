// Post-deploy patch for the bundle copied into magnopus-web-renderer.
//
// The build strips the JSDoc `@type Class` annotation on `scriptType` inside
// `createScript` (src/framework/script/script-create.js). Without it, MWR's dnt
// build (`deno task deploy`) fails declaration emit with:
//   TS9005: Declaration emit for this file requires using private name 'scriptType'.
// Re-insert the annotation into the deployed bundle.

import fs from 'node:fs';

const target = '../magnopus-web-renderer/libs/overrides/playcanvas-engine/build/playcanvas.mjs';
const anchor = '\tconst scriptType = function(args) {';
const annotation = '  /** @type Class */\n';

let source = fs.readFileSync(target, 'utf8');

if (source.includes('/** @type Class */')) {
    console.log('patch-mwr-bundle: @type Class annotation already present');
} else if (source.includes(anchor)) {
    source = source.replace(anchor, annotation + anchor);
    fs.writeFileSync(target, source);
    console.log('patch-mwr-bundle: re-inserted @type Class annotation on scriptType');
} else {
    console.error('patch-mwr-bundle: scriptType anchor not found — bundle shape changed, update this script');
    process.exit(1);
}
