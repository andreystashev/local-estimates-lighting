# Double local-estimate visualization

A randomized 3D illustration of a double local estimate. The script draws an open
box, random scene points and viewing directions, and a computed value at each
point. This is an exploratory visualization, not a complete light transport
solver or a validated physical prediction.

From the repository root, install the example's dependencies in a virtual
environment and run:

```sh
python -m pip install -r pyton_code/double_local_estimate/requirements.txt
python pyton_code/double_local_estimate/main.py
```

A graphical desktop is needed for the interactive Matplotlib window. Edit
`box_size`, `num_directions`, `epsilon`, and `num_points_per_direction` in the
script to change the experiment. Points use a Gaussian distribution around their
center, so `epsilon` is a standard deviation rather than a hard distance limit.
No random seed is set; results differ between runs.

The current kernel and projection are experimental. Review their normalization
and behavior near coincident points before using the values for research
comparisons.

<img width="625" alt="Double local-estimate visualization" src="https://github.com/andreystashev/local-estimates-lighting/assets/70208784/21f3c901-8634-473b-90a4-3f0cf81a9e5e">
