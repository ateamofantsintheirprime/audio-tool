
#version 330

//uniform float time;
uniform sampler2D t0;
in vec2 uv;
out vec4 fragColor;

vec4 hue2rgba(float h) {
	return vec4(
		abs(h * 6.0 - 3.0) - 1.0, 
		2.0 - abs(h * 6.0 - 2.0), 
		2.0 - abs(h * 6.0 - 4.0), 
		1.0
	);
}

vec4 colormap(float t) {
	
	t = min(t,1.0);
	vec3 c = vec3(0.0);
	c.r = smoothstep(0.4, 0.6, t) + smoothstep(0.6, 0.8, t) * 0.5;
	c.g = smoothstep(0.2, 0.5, t) - smoothstep(0.6, 0.9, t);
	c.b = smoothstep(0.0, 0.1, t) - smoothstep(0.3, 0.5, t);
	return vec4(clamp(c, 0.0, 1.0), 1.0);
}

void main()
{
	float val = texture(t0, uv).r; // Stored in the red channel
	fragColor = colormap(val);

	// DISPLAY THE AUDIO DELTA

		// Get texture dimensions to calculate pixel offset
	vec2 texSize = vec2(textureSize(t0, 0));
	vec2 pixelOffset = 1.0 / texSize;
	
	// Sample neighboring pixels to the left
	float val_left1 = texture(t0, uv - vec2(pixelOffset.x, 0.0)).r;
	float val_left2 = texture(t0, uv - vec2(pixelOffset.x * 2.0, 0.0)).r;
	
	// Calculate differences
	float diff1 = abs(val - val_left1);
	float diff2 = abs(val - val_left2);
	
	// Threshold for "sufficiently different"
	float threshold = 0.1; // Adjust this value to change sensitivity
	
	// fragColor = vec4(0.0, 0.0, 0.0, 0.0); 
	// if ((diff1 > threshold) || (diff2 > threshold)) {
	// 	fragColor = vec4(1.0, 1.0, 1.0, 1.0);
	// }
	//if ((val > 0.04) && (diff1 < 0.01) && (diff2 < 0.01)) {
	if ((val > 0.06)){
		fragColor = vec4(1.0, 0.0, 0.0, 1.0);
	}
	fragColor = vec4(val, diff1, diff2, 1.0);
	fragColor = colormap(val);
}
