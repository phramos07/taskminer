# include "../../timing.h"
# include "ldp/common.hpp"


static int mandel(float c_re, float c_im, int count) {

	float z_re = c_re, z_im = c_im;

	int i;
	for (i = 0; i < count; ++i) {

        if (z_re * z_re + z_im * z_im > 4.f) break;

        float new_re = z_re*z_re - z_im*z_im;
        float new_im = 2.f * z_re * z_im;

		z_re = c_re + new_re;
        z_im = c_im + new_im;

    }

    return i;
}

void mandelbrot(float x0, float y0, float x1, float y1,
		int width, int height, int maxIterations, int output[]) {

	float dx = (x1 - x0) / width;
    float dy = (y1 - y0) / height;

    for (int j = 0; j < height; j++) {

		for (int i = 0; i < width; ++i) {

			float x = x0 + i * dx;
            float y = y0 + j * dy;

            int index = (j * width + i);
            output[index] = mandel(x, y, maxIterations);

		}

	}

}


int main(int argc, char *argv[]) {

	float x0 = -2, x1 = 1;
	float y0 = -1, y1 = 1;
	int width = 768, height = 512;

	int max_it = 256;
	int *buf = new int[width * height];
	for (int i = 0; i < width * height; ++i) buf[i] = 0;

	reset_and_start_timer();
	mandelbrot(x0, y0, x1, y1, width, height, max_it, buf);
	double elapsed = get_elapsed_mcycles();

	print_runtime(elapsed);

}
