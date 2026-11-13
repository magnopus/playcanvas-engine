class GpuTimer {
    _enabled = false;

    constructor(device) {
        this.device = device;
        this.enabled = true;
        this.unitsName = 'ms';
        this.decimalPlaces = 1;

        this._timings = [];
    }

    /**
     * Whether the timer runs. The GPU profiler is on only while the timer is: profiling changes
     * how some work is encoded (per-pass timestamps), so it should not outlive the overlay.
     *
     * @type {boolean}
     */
    set enabled(value) {
        this._enabled = value;
        if (this.device.gpuProfiler) {
            this.device.gpuProfiler.enabled = value;
        }
    }

    get enabled() {
        return this._enabled;
    }

    get timings() {
        this._timings[0] = this.device.gpuProfiler?._frameTime ?? 0;
        return this._timings;
    }
}

export { GpuTimer };
