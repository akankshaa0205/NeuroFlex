# NeuroFlex
A real-time, touchless physical rehabilitation assessment engine utilizing CPU-GPU parallel processing for continuous kinematic scoring and gesture-based UI control.

## Run

Install the desktop dependencies from the repository root:

```powershell
python -m pip install -e .
```

Install real 33-joint pose tracking:

```powershell
python -m pip install -e ".[pose]"
```

Without MediaPipe, the runtime uses its clearly separated synthetic fallback.

Start a webcam session with the OpenCV window enabled:

```powershell
python -m neuroflex --live --camera-index 0 --frames 0
```

The window stays open until you press `q`. Use a positive `--frames` value for
a short test session. NeuroFlex requires
`opencv-python` for the GUI window; do not install `opencv-python-headless` in
the same environment.
