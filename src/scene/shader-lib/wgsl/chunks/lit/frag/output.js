export default /* wgsl */`
// magnopus patched ensure outline pass is written
#ifdef PCOUTLINE_PASS
output.color = vec4f(gammaCorrectOutput(uniform.pcOutlineColor), output.color.a);
#endif
`;
