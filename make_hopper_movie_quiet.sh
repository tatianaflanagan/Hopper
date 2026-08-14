#!/usr/bin/env bash
set -euo pipefail
start_time=$(date +%s)
# Usage:
#   ./make_hopper_movie_quiet.sh INPUT_DIR OUTPUT_MOVIE [NOTEBOOK] [FPS] [STRIDE]
#
# STRIDE: use every Nth timestep (default 1 = every timestep).
#
# Example:
#
#./make_hopper_movie_quiet.sh \
#    ./dumps \
#    hopper_movie.mp4 \
#    ./2DHopperForces.ipynb \
#    10 \
#    5

#Defaults
#./make_hopper_movie_quiet.sh
#default parameters:
#Input directory: .
#Output movie: hopper_movie.mp4
#Notebook: 2DHopperForces.ipynb
#FPS: 10
#Stride: 1

INPUT_DIR="${1:-.}" #use arg number 1 if it exists, otherwise, use .
OUTPUT_MOVIE="${2:-hopper_movie.mp4}" #same idea
NOTEBOOK="${3:-2DHopperForces.ipynb}" #same idea
FPS="${4:-10}" #same idea
STRIDE="${5:-1}" #same idea

#eliminating "file not found" errors:
INPUT_DIR="$(cd "$INPUT_DIR" && pwd)"
NOTEBOOK="$(cd "$(dirname "$NOTEBOOK")" && pwd)/$(basename "$NOTEBOOK")"
OUTPUT_MOVIE="$(cd "$(dirname "$OUTPUT_MOVIE")" && pwd)/$(basename "$OUTPUT_MOVIE")"

#check if commands exist
command -v python3 >/dev/null 2>&1 || {
    echo "Error: python3 is not installed or not on PATH." >&2
    exit 1
}

command -v jupyter >/dev/null 2>&1 || {
    echo "Error: jupyter is not installed or not on PATH." >&2
    exit 1
}

command -v ffmpeg >/dev/null 2>&1 || {
    echo "Error: ffmpeg is not installed or not on PATH." >&2
    echo "On macOS with Homebrew: brew install ffmpeg" >&2
    exit 1
}

[[ -f "$NOTEBOOK" ]] || {
    echo "Error: notebook not found: $NOTEBOOK" >&2
    exit 1
}

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/hopper_movie.XXXXXX")"
FRAME_DIR="$WORK_DIR/frames"
RUN_DIR="$WORK_DIR/notebook_run"

mkdir -p "$FRAME_DIR" "$RUN_DIR"
trap 'rm -rf "$WORK_DIR"' EXIT

RUN_NOTEBOOK="$RUN_DIR/$(basename "$NOTEBOOK")"
cp "$NOTEBOOK" "$RUN_NOTEBOOK"

frame_number=0

# Count total frames = sum of timesteps across all .dump/.forces pairs
total_frames=0
found_any_dump=0
for particle_file in "$INPUT_DIR"/*.dump; do
    [[ -e "$particle_file" ]] || continue
    found_any_dump=1
    base="${particle_file%.dump}"
    contact_file="${base}.forces"
    [[ -f "$contact_file" ]] || continue
    n=$(python3 -c "
import sys
count = sum(1 for line in open(sys.argv[1]) if line.startswith('ITEM: TIMESTEP'))
print(count)
" "$particle_file")
    total_frames=$((total_frames + (n + STRIDE - 1) / STRIDE))
done

if (( found_any_dump == 0 )); then
    echo "Error: no .dump files found in $INPUT_DIR" >&2
    exit 1
fi

if (( total_frames == 0 )); then
    echo "Error: no complete .dump/.forces pairs found in $INPUT_DIR" >&2
    exit 1
fi

for particle_file in "$INPUT_DIR"/*.dump; do

    [[ -e "$particle_file" ]] || continue

    base="${particle_file%.dump}"
    contact_file="${base}.forces"

    if [[ ! -f "$contact_file" ]]; then
        echo "Warning: skipping $(basename "$particle_file"); matching .forces file not found." >&2
        continue
    fi

    n_timesteps=$(python3 -c "
import sys
count = sum(1 for line in open(sys.argv[1]) if line.startswith('ITEM: TIMESTEP'))
print(count)
" "$particle_file")

    for ((ts_idx=0; ts_idx<n_timesteps; ts_idx+=STRIDE)); do

        frame_png=$(printf '%s/frame_%06d.png' "$FRAME_DIR" "$frame_number")

        python3 - "$RUN_DIR/config.py" "$particle_file" "$contact_file" "$frame_png" "$ts_idx" <<'PYCONFIG'
import sys
from pathlib import Path

config_path, particle_file, contact_file, output_png, timestep_index = sys.argv[1:]

text = (
    "from pathlib import Path\n\n"
    f"PARTICLE_FILE = Path({str(Path(particle_file).resolve())!r})\n"
    f"CONTACT_FILE = Path({str(Path(contact_file).resolve())!r})\n"
    f"OUTPUT_PNG = Path({str(Path(output_png).resolve())!r})\n"
    f"TIMESTEP_INDEX = {int(timestep_index)}\n"
)

Path(config_path).write_text(text, encoding="utf-8")
PYCONFIG

        printf "[%d/%d] Rendering %s (timestep index %d)\n" \
            "$((frame_number + 1))" \
            "$total_frames" \
            "$(basename "$particle_file")" \
            "$ts_idx"

        executed_name="executed_${frame_number}.ipynb"

        (
            cd "$RUN_DIR"

            if ! jupyter nbconvert \
                --to notebook \
                --execute "$RUN_NOTEBOOK" \
                --output "$executed_name" \
                --output-dir "$WORK_DIR" \
                --ExecutePreprocessor.timeout=-1 \
                --log-level=ERROR \
                >/dev/null
            then
                echo "Error: notebook execution failed for:" >&2
                echo "       $(basename "$particle_file") timestep index $ts_idx" >&2
                exit 1
            fi
        )

        if [[ ! -s "$frame_png" ]]; then
            echo "Error: notebook did not create $frame_png" >&2
            echo "Check that the notebook imports OUTPUT_PNG from config.py" >&2
            echo "and passes save_path=OUTPUT_PNG to the plotting function." >&2
            exit 1
        fi

        frame_number=$((frame_number + 1))

        #Add ETA after each frame
        current_time=$(date +%s)
        elapsed_so_far=$((current_time - start_time))

        average_seconds=$(awk "BEGIN {
            if ($frame_number > 0)
                printf \"%.2f\", $elapsed_so_far / $frame_number
            else
                printf \"0\"
        }")

        remaining_frames=$((total_frames - frame_number))

        eta_seconds=$(awk "BEGIN {
            printf \"%.0f\", $average_seconds * $remaining_frames
        }")

        eta_hours=$((eta_seconds / 3600))
        eta_minutes=$(((eta_seconds % 3600) / 60))
        eta_secs=$((eta_seconds % 60))

        printf "        Average: %s s/frame | ETA: %02d:%02d:%02d\n" \
            "$average_seconds" \
            "$eta_hours" \
            "$eta_minutes" \
            "$eta_secs"

    done
done

if (( frame_number == 0 )); then
    echo "Error: no complete .dump/.forces pairs were processed." >&2
    exit 1
fi

echo "Creating movie from $frame_number frames..."

ffmpeg \
    -hide_banner \
    -loglevel error \
    -nostats \
    -y \
    -framerate "$FPS" \
    -i "$FRAME_DIR/frame_%06d.png" \
    -c:v libx264 \
    -pix_fmt yuv420p \
    -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" \
    "$OUTPUT_MOVIE"


end_time=$(date +%s)
elapsed=$((end_time - start_time))
hours=$((elapsed / 3600))
minutes=$(((elapsed % 3600) / 60))
seconds=$((elapsed % 60))
echo
echo "======================================="
echo "Movie created"
echo "Frames rendered : $frame_number"
echo "Output movie    : $OUTPUT_MOVIE"
printf "Elapsed time    : %02d:%02d:%02d\n" \
    "$hours" "$minutes" "$seconds"
average=$(awk "BEGIN {printf \"%.2f\", $elapsed / $frame_number}")
echo "Average/frame   : ${average} seconds"
echo "======================================="

#open movie
if command -v open >/dev/null 2>&1; then
    open "$OUTPUT_MOVIE"
fi