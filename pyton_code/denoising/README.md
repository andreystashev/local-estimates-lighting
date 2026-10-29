# Cornell box rendering and denoising

A Taichi Monte Carlo rendering experiment with Intel Open Image Denoise (OIDN).
The renderer opens a Taichi window and, every 100 rendering iterations, saves the
current image, denoises it, and displays the result through Pygame.

## Setup and run

Use a separate virtual environment. From the repository root:

```sh
python -m pip install -r pyton_code/denoising/requirements.txt
python pyton_code/denoising/main.py
```

This example needs a graphical desktop, a GPU backend supported by Taichi, and
the native OIDN runtime expected by the `oidn` Python wrapper. The requirements
file records historical package versions; it does not provision GPU drivers or
native libraries. Consult the Taichi and OIDN installation instructions for your
platform if initialization fails.

The program writes `CornellBoxNoised.png` and `CornellBoxDenoised.png` to the
current working directory, replacing the previous pair at each denoising step.
`pix_resolution` controls image dimensions; the interval is set in `main()`.
Close the rendering window to end the loop.

The local-estimate function in the renderer is a research prototype with unused
sampling parameters. Rendering and image quality require further validation;
this example is not a production renderer.

<img width="1404" alt="Cornell box rendering and denoised output" src="https://github.com/andreystashev/local-estimates-lighting/assets/70208784/ddf152e3-ffea-4136-be0b-b6f08792b449">
