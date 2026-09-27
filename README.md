# Smart Vision Event Filter

A small computer vision project that turns a video into motion events. It uses OpenCV to find changing areas between frames. Optionally, a pretrained YOLO model checks frames with motion and keeps objects whose boxes overlap those areas. The result is a JSON report, with the option to export short video clips.

I built this project to practice an end-to-end AI workflow, including packaging, tests, Docker, and CI. It is a prototype, not a production surveillance system.

## Run locally

You need Python 3.14 and your own video file. Videos, model weights, and generated outputs are not stored in this repository.

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
mkdir -p outputs
python -m smart_vision_event_filter path/to/video.mp4 --no-preview --output outputs/events.json
```

To also detect objects, install the optional AI dependencies and enable AI:

```bash
python -m pip install -e ".[dev,ai]"
SVF_DEVICE=cpu python -m smart_vision_event_filter path/to/video.mp4 --enable-ai --target-class person --no-preview --output outputs/person_events.json
```

`SVF_DEVICE=cpu` makes this example work without Apple MPS. On a supported Mac, you can omit it to use the project's MPS default. Leave out `--target-class person` to keep all motion events. Add `--clips-dir outputs/clips` if you also want video clips.

## Run with Docker

The Docker image includes the AI dependencies and uses the CPU. Create the folders, put a video at `videos/input.mp4`, and then build and run the image:

```bash
mkdir -p videos outputs
```

After adding the video:

```bash
docker build -t smart-vision-filter:dev .
docker run -v "$PWD/videos:/app/videos:ro" -v "$PWD/outputs:/app/outputs" smart-vision-filter:dev videos/input.mp4 --enable-ai --target-class person --no-preview --output outputs/person_events.json
```

The mounted `videos` directory provides the input; the mounted `outputs` directory keeps the JSON on your computer. The first AI run may also download the model weights.

## What the output means

Motion in nearby frames is grouped into an event, allowing short gaps without motion. Each JSON entry contains the start and end frames, times in seconds, duration, and any overlapping object classes found by YOLO. For example:

```json
{"start_frame": 1, "end_frame": 240, "start_seconds": 0.03, "end_seconds": 8.03, "duration_seconds": 8.0, "detected_objects": ["person"]}
```

## Checks and limitations

Run `python -m ruff check src tests` and `python -m pytest` locally. GitHub Actions runs the same checks on pushes and pull requests. These automated tests do not run YOLO on a real video; I checked that path separately with a Docker run.

Frame differencing can react to lighting changes or camera movement. Object classes come from a pretrained model, and the motion/YOLO overlap is a simple heuristic. The small ground-truth example is useful for learning, but not a large benchmark or a claim of production accuracy.
