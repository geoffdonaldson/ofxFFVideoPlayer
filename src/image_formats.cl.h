/*
  Copyright (c) 2011 Michael Zucchi

  This file is part of socles, an OpenCL image processing library.

  socles is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  socles is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with socles.  If not, see <http://www.gnu.org/licenses/>.
*/

/*
  Basic image handling routines, format conversion and so on
*/

/* Copy all of an image to another image.
   Does implicit format conversions to compatible formats which the enqueue_copy_image doesn't. */

#ifndef IMAGE_FORMATSp_H
#define IMAGE_FORMATSp_H

#define stringify(x...) #x

string image_conversions = stringify(
                                                                        
kernel void convert_i_rgba_i_rgba(image2d_t src, write_only image2d_t dst) {
	int2 pos = { get_global_id(0), get_global_id(1) };

	if ((pos.x < get_image_width(dst)) & (pos.y < get_image_height(dst))) {
		const sampler_t smp = CLK_FILTER_NEAREST;

		write_imagef(dst, pos, read_imagef(src, smp, pos));
	}
}

static inline float
rgb2grey(float4 pixel) {
	/* uses weighting equation from OpenCV */
	return pixel.s0 * 0.299f + pixel.s1 * 0.587f + pixel.s2 * 0.114f;
}

/* convert image rgb to image grey ignoring alpha */
kernel void
convert_i_rgb_i_y(read_only image2d_t src, write_only image2d_t dst) {
	int gx = get_global_id(0);
	int gy = get_global_id(1);

	if ( (gx < get_image_width(src)) & (gy < get_image_height(src)) ) {
		const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
		int2 pos = { gx, gy };
		float4 pixel = read_imagef(src, smp, pos);
		float g = rgb2grey(pixel);

		write_imagef(dst, pos, (float4)g);
	}
}

/* convert 1 channel of greyscale to rgba image */
kernel void
convert_i_y_i_rgba(read_only image2d_t src, int chid, write_only image2d_t dst) {
	int gx = get_global_id(0);
	int gy = get_global_id(1);

	if ( (gx < get_image_width(src)) & (gy < get_image_height(src)) ) {
		const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
		int2 pos = { gx, gy };
		float4 pixel = read_imagef(src, smp, pos);
		float res = pixel.s0;

		res = chid == 1 ? pixel.s1 : res;
		res = chid == 2 ? pixel.s2 : res;
		res = chid == 3 ? pixel.s3 : res;

		write_imagef(dst, pos, (float4)res);
	}
}

// planar yuv 4:2:0
kernel void
convert_b_yuv420p_i_rgb(global uchar *src, uint w, uint h, write_only image2d_t dst) {
	uint gx = get_global_id(0);
	uint gy = get_global_id(1);

	if ((gx < w) & (gy<h)) {
		float4 p;

		float Y = 1.1643 * (src[gx + gy*w] / 255.0f - 0.0625);
		float Cr = src[gx/2+(gy/2)*(w/2)+w*h] / 255.0f - 0.5f;
		float Cb = src[gx/2+(gy/2)*(w/2)+w*h+(w/2)*(h/2)] / 255.0f - 0.5f;

		p.s0 = Y + 1.5958 * Cb;
		p.s1 = Y - 0.39173*Cr-0.81290*Cb;
		p.s2 = Y + 2.017*Cr;
		p.s3 = 1.0f;

		write_imagef(dst, (int2){ gx, gy }, p);
	}
}

// planar yuv 4:2:0 stored as 3 separate planes
//  for converting raw data as from ffmpeg libraries
kernel void
convert_bbb_yuv420p_i_rgb(global uchar *srcY, uint strideY,
			  global uchar *srcU, uint strideU,
			  global uchar *srcV, uint strideV,
			  uint w, uint h, write_only image2d_t dst) {
	uint gx = get_global_id(0);
	uint gy = get_global_id(1);

	if ((gx < w) & (gy<h)) {
		float4 p;

		float Y = 1.1643 * (srcY[gx + gy*strideY] / 255.0f - 0.0625);
		float Cr = srcU[gx/2+(gy/2)*(strideU)] / 255.0f - 0.5f;
		float Cb = srcV[gx/2+(gy/2)*(strideV)] / 255.0f - 0.5f;

		p.s0 = Y + 1.5958 * Cb;
		p.s1 = Y - 0.39173*Cr-0.81290*Cb;
		p.s2 = Y + 2.017*Cr;
		p.s3 = 1.0f;

		write_imagef(dst, (int2){ gx, gy }, p);
	}
}

// planar yuv 4:2:2
kernel void
convert_b_yuv422p_i_rgb(global uchar *src, uint w, uint h, write_only image2d_t dst) {
	uint gx = get_global_id(0);
	uint gy = get_global_id(1);

	if ((gx < w) & (gy<h)) {
		float4 p;

		float Y = 1.1643 * (src[gx + gy*w] / 255.0f - 0.0625);
		float Cr = src[gx/2+(gy/2)*(w)+w*h] / 255.0f - 0.5f;
		float Cb = src[gx/2+(gy/2)*(w)+w*h+(w)*(h/2)] / 255.0f - 0.5f;

		p.s0 = Y + 1.5958 * Cb;
		p.s1 = Y - 0.39173*Cr-0.81290*Cb;
		p.s2 = Y + 2.017*Cr;
		p.s3 = 1.0f;

		write_imagef(dst, (int2){ gx, gy }, p);
	}
}

// packed YUYV format
// each thread does 2 pixels
kernel void
convert_b_yuyv_i_rgb(global uchar *src, uint w, uint h, write_only image2d_t dst) {
	uint gx = get_global_id(0);
	uint x = gx*2;
	uint gy = get_global_id(1);
    
	if ((x+1 < w) & (gy<h)) {
		uint off = gy * w * 2 + x * 2;
		float4 p;
		float Y0 = 1.1643 * (src[off+1] / 255.0f - 0.0625);
		float Cb = src[off+2] / 255.0f - 0.5f;
		float Y1 = 1.1643 * (src[off+3] / 255.0f - 0.0625);
		float Cr = src[off] / 255.0f - 0.5f;

		p.s0 = Y0 + 1.5958 * Cb;
		p.s1 = Y0 - 0.39173*Cr-0.81290*Cb;
		p.s2 = Y0 + 2.017*Cr;
		p.s3 = 1.0f;

		write_imagef(dst, (int2){ x, gy }, p);

		p.s0 = Y1 + 1.5958 * Cb;
		p.s1 = Y1 - 0.39173*Cr-0.81290*Cb;
		p.s2 = Y1 + 2.017*Cr;
		p.s3 = 1.0f;

		write_imagef(dst, (int2){ x+1, gy }, p);
	}
}

// packed BGR to RGBx
// 12 input bytes are converted to 4 output pixels at a time
// TODO: this isn't finished
kernel void
convert_b_bgr_i_rgb(global uchar4 *src, int w, int h, write_only image2d_t dst) {
	int gx = get_global_id(0);
	int gy = get_global_id(1);

	if ((gx*4+3 < w) & (gy < h)) {
		int off = gx * 3 + gy * w;
		uchar4 s0 = src[off+0];
		uchar4 s1 = src[off+1];
		uchar4 s2 = src[off+2];

		float4 p0 = { s0.s0, s0.s1, s0.s2, 1 };
		float4 p1 = { s0.s3, s1.s0, s1.s1, 1 };
		float4 p2 = { s1.s2, s1.s3, s2.s0, 1 };
		float4 p3 = { s2.s1, s2.s2, s2.s3, 1 };

		write_imagef(dst, (int2){ gx*4+0, gy }, p0);
		write_imagef(dst, (int2){ gx*4+1, gy }, p1);
		write_imagef(dst, (int2){ gx*4+2, gy }, p2);
		write_imagef(dst, (int2){ gx*4+3, gy }, p3);
	}
}
);

#endif