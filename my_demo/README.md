# Look Once to Hear Demo Scripts

This folder provides runnable examples that sit on top of the core Look Once to Hear implementation. Use them to validate your setup and to experiment with your own recordings.

## Files
- `interactive_demo.py` - Command-line walkthrough that scans `demo_recordings/` for `.wav` files, guides you through speaker enrollment, and saves an enhanced mixture.
- `personal_demo.py` - Importable helper that exposes a `PersonalDemo` class and an optional quick-demo routine for integrating the pipeline into notebooks or scripts.

## Prerequisites
- Install the project dependencies from the repository root.
- Keep `../configs/tsh_cipic_only.json` and the checkpoint `../runs/tsh/best.ckpt` in place; `PersonalDemo` loads them using paths relative to this folder.
- Place your enrollment and mixture `.wav` files under `my_demo/demo_recordings/`. Aim for 3-5 seconds of clean speech for enrollment and a longer, noisy mixture that contains the same speaker.

## Run the Interactive Demo
1. Open a terminal, activate your Python environment, and `cd my_demo`.
2. Execute `python interactive_demo.py`.
3. Follow the prompts to choose the enrollment clip and the mixture. The script writes the enhanced result to `enhanced_<mixture-name>.wav` and prints quick diagnostics.

Choose this path when you want a guided experience without writing any code.

## Use `PersonalDemo` in Python
Import `PersonalDemo` when you prefer to control the workflow programmatically:

```python
from personal_demo import PersonalDemo

demo = PersonalDemo()  # loads the TSH model and speaker encoder
demo.enroll_speaker("demo_recordings/my_target_speaker.wav")
results = demo.separate_target_speaker(
    "demo_recordings/noisy_mixture.wav",
    "enhanced_output.wav",
)
analysis = demo.analyze_results(results)
```

Key pieces:
- `enroll_speaker(path)` - Extracts and stores a speaker embedding from a clean enrollment clip.
- `separate_target_speaker(mixture_path, output_path=None)` - Runs Look Once to Hear on a mixture using the cached embedding; optionally saves the enhanced audio.
- `analyze_results(results)` - Computes simple metrics and, when run in IPython, renders interactive audio comparisons.

Running `python personal_demo.py` from this directory prints the same instructions and can attempt a quick demo if sample data exists at `../data/MixLibriSpeech/...`.

## Tips
- Keep enrollment recordings quiet and close-mic'd; the cleaner the enrollment, the better the separation.
- Stereo mixtures provide stronger cues. Mono files are duplicated to stereo automatically, but real binaural recordings yield better results.
- If you see import errors, make sure you launched Python from inside `my_demo` so that relative paths resolve correctly.
