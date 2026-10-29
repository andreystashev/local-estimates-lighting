# C++ image denoising example

`main.cpp` loads an image with FreeImage, converts its RGB channels to floating
point, applies Intel Open Image Denoise's `RT` filter, and writes a JPEG image.

This directory contains source only, without a supplied CMake target. To build,
install a compatible C++ compiler, Intel Open Image Denoise development headers
and libraries, and FreeImage development headers and libraries. Add their include
and library paths to your platform's build command and link OpenImageDenoise and
FreeImage.

Before running, replace `/your_way/sampleInput.jpg` and
`/your_way/sampleOutput.jpg` in `main()` with your input and output paths. The
output is JPEG regardless of the output filename extension. Existing output is
replaced.

The source is a historical API example: filter parameters and image color-space
settings must be checked against the installed OIDN version and input image
before relying on the result.

<img width="1162" alt="Example noisy and denoised images" src="https://github.com/andreystashev/local-estimates-lighting/assets/70208784/b6a80135-2b72-4775-bd0f-2b83111863ba">
