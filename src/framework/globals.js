import { GraphicsDeviceAccess } from "../platform/graphics/graphics-device-access.js";

let currentApplication;

function getApplication() {
    console.log("***** getApplication() *****");
    return currentApplication;
}

function setApplication(app) {
    currentApplication = app;
    GraphicsDeviceAccess.set(app?.graphicsDevice);
}

export {
    getApplication,
    setApplication
};
