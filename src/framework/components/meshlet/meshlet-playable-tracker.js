import { MeshletComponent } from './component.js';

/**
 * @import { MeshletComponentSystem } from './system.js'
 * @import { MeshletStreamStatus } from './component.js'
 */

// shared component statuses for the states with nothing to measure
const STATUS_NO_GROUP = Object.freeze({ playable: true, failed: false, progress: 1, totalBytes: 0, loadedBytes: 0 });
const STATUS_NOT_PLACED = Object.freeze({ playable: false, failed: false, progress: 0, totalBytes: 0, loadedBytes: 0 });

/**
 * Tracks when the meshlet content, and each component waited on, becomes playable.
 *
 * @ignore
 */
class MeshletPlayableTracker {
    /**
     * @type {MeshletComponentSystem}
     * @private
     */
    _system;

    /**
     * {@link MeshletComponentSystem#playable} as of the last frame, to fire on transitions.
     *
     * @type {boolean}
     * @private
     */
    _wasPlayable = true;

    /**
     * Components with {@link MeshletComponent#whenPlayable} waiters, polled each frame.
     *
     * @type {Set<MeshletComponent>}
     * @private
     */
    _watch = new Set();

    /**
     * @param {MeshletComponentSystem} system - The system.
     */
    constructor(system) {
        this._system = system;
    }

    /**
     * @param {boolean} scanAssets - Check every component for a loading asset.
     * @returns {boolean} Whether the content is playable.
     */
    isPlayable(scanAssets) {
        const system = this._system;
        const group = system._group;
        if (!group) return true;
        if (system._dirty) return false;
        // a world with nothing in it is never finalized, and has nothing to stream
        if (!group.playable) return false;
        return !scanAssets || !system._hasPendingAssets();
    }

    /**
     * @param {MeshletComponent} component - The component.
     * @returns {MeshletStreamStatus} Its load state.
     */
    componentStatus(component) {
        const group = this._system._group;
        if (!group) return STATUS_NO_GROUP;
        const director = group.directors[component._partition];
        const resource = component._effectiveResource;
        // not placed yet: the asset is loading, or the rebuild that adds it has not run
        if (!resource || !director || component._placementIndex < 0 || component._placedResource !== resource) {
            return STATUS_NOT_PLACED;
        }
        return director.resourceStatus(resource) ?? STATUS_NOT_PLACED;
    }

    /**
     * @param {MeshletComponent} component - A component with playable waiters.
     */
    watch(component) {
        this._watch.add(component);
    }

    /**
     * @param {MeshletComponent} component - The component.
     */
    unwatch(component) {
        this._watch.delete(component);
    }

    /**
     * Called once per frame.
     *
     * @returns {boolean} True when the content has just become playable.
     */
    updatePlayable() {
        // the director's state is from last frame's update: a transition fires one frame late.
        // Once playable, a new asset marks the system dirty when assigned, so the per-component
        // asset scan only runs while waiting.
        const playable = this.isPlayable(!this._wasPlayable);
        const becamePlayable = playable && !this._wasPlayable;
        this._wasPlayable = playable;
        return becamePlayable;
    }

    /**
     * Reports progress to, and settles, the components being waited on. Called once per frame.
     */
    updateWatch() {
        for (const component of this._watch) {
            const status = this.componentStatus(component);
            if (status.playable) {
                this._watch.delete(component);
                component.fire(MeshletComponent.EVENT_PROGRESS, 1, status.totalBytes);
                component.fire(MeshletComponent.EVENT_PLAYABLE, status);
                component._settlePlayable(status);
            } else if (status.progress !== component._reportedProgress) {
                component._reportedProgress = status.progress;
                component.fire(MeshletComponent.EVENT_PROGRESS, status.progress, status.totalBytes);
            }
        }
    }

    destroy() {
        for (const component of this._watch) {
            component._settlePlayable({ playable: false, failed: true, progress: 0, totalBytes: 0, loadedBytes: 0 });
        }
        this._watch.clear();
    }
}

export { MeshletPlayableTracker };
