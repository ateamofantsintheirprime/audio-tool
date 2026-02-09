vec4 hsv2rgb(float h) {
	vec3 c = vec3(h,1.0,1.0);
	// Define constants for the color conversion math
	vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
	
	// Use the fractional part (fract) of the hue and offset to map to the RGB range
	// abs() and clamp() are used to create a "wave" pattern for the R, G, B channels
	vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
	
	// mix() interpolates between black (K.xxx or vec3(1.0)) and the calculated color, 
	// controlled by the saturation (c.y), and finally scaled by the value (c.z)
	return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {

	// Normalized pixel coordinates (from 0 to 1)
	vec2 uv = fragCoord/iResolution.xy;
	vec 
	// How far is the current pixel from the origin (0, 0)
	float distance = length(uv);

	// Are we are 20% of the screen away from the origin?
	if (distance > 0.2) {
		// Black
		fragColor = vec4(0.0, 0.0, 0.0, 1.0);
	} else {
		// White
		fragColor = vec4(1.0, 1.0, 1.0, 1.0);
	}
}
