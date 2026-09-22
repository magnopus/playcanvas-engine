export default /* glsl */`
// magnopus patched ensure outline pass is written
#ifdef PCOUTLINE_PASS
gl_FragColor.rgb = gammaCorrectOutput(pcOutlineColor);
#endif
`;
