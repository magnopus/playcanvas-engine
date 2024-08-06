export default /* glsl */`
#ifdef MULTIVIEW
    #extension GL_OVR_multiview : require
    layout(num_views=2) in;
#endif
attribute vec3 vertex_position;
attribute vec3 vertex_normal;
attribute vec4 vertex_tangent;
attribute vec2 vertex_texCoord0;
attribute vec2 vertex_texCoord1;
attribute vec2 vertex_texCoord2;
attribute vec2 vertex_texCoord3;
attribute vec2 vertex_texCoord4;
attribute vec4 vertex_color;

uniform mat4 matrix_viewProjection;
#ifdef MULTIVIEW
    uniform mat4 matrix_viewProjection_2;
#endif
uniform mat4 matrix_model;
uniform mat3 matrix_normal;

vec3 dPositionW;
mat4 dModelMatrix;
mat3 dNormalMatrix;
`;
