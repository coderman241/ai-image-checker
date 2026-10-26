# AI Image Checker

A simple program that checks an image and gives a best-guess
verdict about whether it looks AI-generated or like a real photo.

> **Important limitation, first:** this is a heuristic checker, not a
> definitive AI detector. It does not prove whether an image was
> AI-generated or real. It does not identify a person, a camera, or a
> specific AI tool, and it does not download a model, call an online
> service, or need an API key.

## What it checks

Both versions below use the same simple, local checks:

- File data for camera makes, generator keywords (for example
  Midjourney, Stable Diffusion, Adobe Firefly), and software tags.
- A few lightweight image clues: very smooth / low-noise areas and
  unusually high saturation. These are weak evidence on their own —
  real photos can be smooth and vivid too.

Each check gives a verdict, a score, and the reasons behind it.

## What the verdicts and score mean

The score runs from 0 to 100, starting at 50 (no useful signal either way):

- 0 = most consistent with a real / camera photo
- 50 = uncertain
- 100 = most consistent with an AI-generated image

Thresholds:

- Score **75 or higher** → `likely AI-generated`
- Score **35 or lower** → `likely real / camera photo`
- Anything in between → `uncertain`

Weak clues alone will not reach `likely AI-generated` — they land in
`uncertain` instead. The score is a heuristic lean, not a measured
percentage or a certainty.

## Web version (HTML, CSS, JS)

Files: `index.html`, `style.css`, `script.js`.

This version is mobile-friendly and PC-friendly: a single-column
layout on phones, two columns on wider screens (800px and up), large
tap targets, and no external fonts or frameworks.

It runs entirely in your browser. Your image is never uploaded to a
server — it is read on your device, checked, and shown back to you.

To use it:

1. Open the site (see GitHub Pages below) or open `index.html` locally.
2. Tap to choose an image, or drag one onto the drop zone.
   Supported formats: JPEG, PNG, GIF, WebP, BMP, TIFF.
3. Tap **Check image** to see the verdict, score, and reasons.

## Python command-line version

File: `ai_image_check.py`.

Install (Python 3, no third-party packages required):

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with:

```sh
.venv\Scripts\activate
```

Run, from the project folder:

```sh
python ai_image_check.py <image_path>
python ai_image_check.py <image_path> --json
```

Replace `<image_path>` with the path to the image you want to check.
`--json` gives the same verdict, score, and reasons in JSON format.
Pillow is optional: if installed, simple image statistics are included;
if not, the program still runs on metadata and format clues alone.

Tests: `python -m unittest test_ai_image_check.py -v`

## GitHub Pages

`index.html` sits at the repo root, so GitHub Pages can serve it
directly. After uploading the files to GitHub:

1. Go to the repo's **Settings > Pages**.
2. Under Build and deployment, choose **Deploy from a branch**.
3. Choose branch **main** and folder **/ (root)**, then save.
4. The site will be at:

   `https://<owner>.github.io/ai-image-checker/`

   Replace `<owner>` with the GitHub username or organization that
   owns the repo.

## Limitations

- The result is a guess based on simple checks, not proof of origin.
- It can be wrong in both directions: a real photo that has been
  heavily edited, compressed, or screenshotted can look unusual, and
  an AI-generated image can look very photographic.
- Metadata can be stripped, edited, or faked, so its absence or
  presence is weak evidence on its own. Many real photos have no
  camera data after being shared or re-saved, and many AI images can
  carry fake camera data.
- Editing, cropping, resizing, and compression can change the result.
- Do not use this result on its own as evidence that an image is, or
  is not, AI-generated. If the answer matters, check other evidence too.
