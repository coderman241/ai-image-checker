"""Tests for the simple AI-image checker.

Run from the project folder:

    python -m unittest test_ai_image_check.py -v

Or, equivalently:

    python test_ai_image_check.py

These tests use only Python's standard library. They generate every
test image inside a temporary directory, so no binary test images are
committed to the repository.

Expected program API (from README.md)
------------------------------------
The checker is a command-line program:

    python ai_image_check.py <image_path> [--json]

It prints a verdict (``likely AI``, ``likely real`` or ``uncertain``),
a score, and the reasons behind that verdict. ``--json`` gives the
same result in JSON form.

The tests therefore prefer the documented CLI with ``--json``. If the
module also exposes a callable such as ``check_image`` /
``analyze_image`` / ``detect`` that returns a dict, that path is used
as a fallback. Either way, results are normalised to a dict with
``verdict`` (str), ``score`` (float or None) and ``reasons`` (list),
so the tests do not depend on the exact internal key names.

A clear error is *not* a crash: for a missing file or a non-image
file, the program may either exit non-zero with a human-readable
error message, or return a JSON result carrying an error. What it
must not do is die with an unhandled traceback and no explanation.
"""

import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib
from pathlib import Path


# ---------------------------------------------------------------------------
# Locating the program under test
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
SCRIPT = PROJECT_DIR / "ai_image_check.py"

# Candidate module-level callables, in preference order. The documented
# interface is the CLI, so these are only a fallback.
_FUNCTION_NAMES = (
    "check_image",
    "analyze_image",
    "analyse_image",
    "detect",
    "detect_ai_image",
    "check",
    "assess_image",
    "is_ai_generated",
)


def _write_png(path, rgb=(200, 50, 50), size=16, text_chunks=None):
    """Write a minimal, valid solid-colour PNG using only stdlib.

    ``text_chunks`` is an optional list of (keyword, text) pairs which
    are written as PNG tEXt chunks, e.g. ("Software", "Midjourney").
    """
    path = Path(path)
    width = height = size
    row = b"\x00" + bytes(rgb) * width
    raw = row * height
    idat = zlib.compress(raw)

    def chunk(chunk_type, data):
        return (
            struct.pack(">I", len(data))
            + chunk_type
            + data
            + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
        )

    out = bytearray(b"\x89PNG\r\n\x1a\n")
    out += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    for keyword, text in text_chunks or []:
        out += chunk(b"tEXt", keyword.encode("latin-1") + b"\x00" + text.encode("latin-1"))
    out += chunk(b"IDAT", idat)
    out += chunk(b"IEND", b"")
    path.write_bytes(bytes(out))
    return path


def _write_checkerboard_png(path, size=16):
    """Write a valid, high-texture checkerboard PNG using only stdlib.

    Alternating black/white pixels give a very high mean neighbour
    difference, the opposite of a smooth solid-colour image, so it
    exercises the textured-image branch of the checker.
    """
    path = Path(path)
    rows = []
    for y in range(size):
        row = bytearray(b"\x00")
        for x in range(size):
            row += bytes((0, 0, 0)) if (x + y) % 2 == 0 else bytes((255, 255, 255))
        rows.append(bytes(row))
    idat = zlib.compress(b"".join(rows))

    def chunk(chunk_type, data):
        return (
            struct.pack(">I", len(data))
            + chunk_type
            + data
            + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
        )

    out = bytearray(b"\x89PNG\r\n\x1a\n")
    out += chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
    out += chunk(b"IDAT", idat)
    out += chunk(b"IEND", b"")
    path.write_bytes(bytes(out))
    return path


def _normalise_result(raw):
    """Normalise a dict / object / tuple result to verdict/score/reasons."""
    verdict = None
    score = None
    reasons = []
    error = None

    if isinstance(raw, dict):
        # Case-insensitive key lookup.
        lowered = {str(k).lower(): v for k, v in raw.items()}
        for key in ("verdict", "label", "result", "classification", "prediction"):
            if lowered.get(key) is not None:
                verdict = str(lowered[key])
                break
        for key in ("score", "ai_score", "ai_probability", "probability", "confidence"):
            if lowered.get(key) is not None:
                try:
                    score = float(lowered[key])
                except (TypeError, ValueError):
                    score = None
                break
        for key in ("reasons", "reason", "evidence", "details", "explanations"):
            value = lowered.get(key)
            if value is None:
                continue
            if isinstance(value, (list, tuple)):
                reasons = [str(v) for v in value]
            else:
                reasons = [str(value)]
            break
        error = lowered.get("error")
    elif isinstance(raw, (list, tuple)) and raw:
        # e.g. (verdict, score, reasons)
        verdict = str(raw[0]) if raw[0] is not None else None
        if len(raw) > 1:
            try:
                score = float(raw[1])
            except (TypeError, ValueError):
                score = None
        if len(raw) > 2 and isinstance(raw[2], (list, tuple)):
            reasons = [str(v) for v in raw[2]]
    else:
        # Object with attributes.
        for attr in ("verdict", "label", "result"):
            if hasattr(raw, attr):
                verdict = str(getattr(raw, attr))
                break
        for attr in ("score", "ai_score", "confidence"):
            if hasattr(raw, attr):
                try:
                    score = float(getattr(raw, attr))
                except (TypeError, ValueError):
                    score = None
                break
        if hasattr(raw, "reasons"):
            value = getattr(raw, "reasons")
            reasons = [str(v) for v in value] if isinstance(value, (list, tuple)) else [str(value)]

    return {"verdict": verdict, "score": score, "reasons": reasons, "error": error}


def _call_function(path):
    """Try the module-function API. Returns None if it is not available."""
    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))
    try:
        import ai_image_check  # noqa: PLC0415 - lazy, so a missing module is a clear test failure, not an import crash
    except Exception:
        return None
    for name in _FUNCTION_NAMES:
        func = getattr(ai_image_check, name, None)
        if callable(func):
            return _normalise_result(func(str(path)))
    return None


def _run_cli_json(path):
    """Run the documented CLI with --json. Returns (result, completed_process)."""
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), str(path), "--json"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    stdout = (proc.stdout or "").strip()
    if stdout:
        try:
            return _normalise_result(json.loads(stdout)), proc
        except json.JSONDecodeError:
            # Not JSON: fall through to the plain-text handling below.
            pass
    # No usable JSON. Return a synthetic result carrying the raw output
    # so the error tests can inspect it, and so a plain-text verdict
    # (if any) is still visible.
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    verdict = None
    for candidate in ("likely ai", "likely real", "uncertain"):
        if candidate in combined.lower():
            verdict = candidate
            break
    return {"verdict": verdict, "score": None, "reasons": [], "error": combined.strip() or None}, proc


def _check(path):
    """Run the checker on ``path``.

    Returns (result_dict_or_None, exception_or_None, process_or_None).
    A missing/non-image input is expected to surface as a clear error
    through one of these channels, not as an unhandled traceback.
    """
    if SCRIPT.exists():
        result, proc = _run_cli_json(path)
        # A non-zero exit with an error message is a clear error, not a
        # successful verdict. Surface it as an exception-like outcome
        # for the error tests, while still returning the raw text.
        if proc.returncode != 0:
            message = ((proc.stderr or "") + "\n" + (proc.stdout or "")).strip()
            return None, ValueError(message or "checker failed"), proc
        return result, None, proc

    # No CLI script yet: fall back to a module function, if present.
    try:
        result = _call_function(path)
    except (FileNotFoundError, ValueError, OSError) as exc:
        return None, exc, None
    if result is None:
        raise AssertionError(
            "ai_image_check.py is not present and no module function "
            f"({', '.join(_FUNCTION_NAMES)}) was found in ai_image_check. "
            "The checker program has not been written yet."
        )
    return result, None, None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAiImageCheck(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    # -- helpers ---------------------------------------------------------

    def _solid_png(self, name="plain.png", rgb=(200, 50, 50), text_chunks=None):
        path = self.dir / name
        _write_png(path, rgb=rgb, text_chunks=text_chunks)
        return path

    # -- required coverage ----------------------------------------------

    def test_missing_file_gives_clear_error_not_crash(self):
        """A missing file gives a clear error, not a crash."""
        missing = self.dir / "does-not-exist.png"
        result, exc, proc = _check(missing)

        if proc is not None:
            combined = ((proc.stderr or "") + "\n" + (proc.stdout or "")).lower()
            self.assertNotIn("traceback", combined,
                             "Missing file produced an unhandled traceback")
            self.assertNotEqual(proc.returncode, 0,
                                "Missing file should not be reported as a success")
            self.assertTrue(
                any(word in combined for word in ("not found", "does not exist", "no such", "missing", "cannot", "error")),
                f"Missing-file error message was not clear: {combined!r}",
            )
            return

        # Function-API path: a clear exception, or an error result.
        if exc is not None:
            self.assertIsInstance(exc, (FileNotFoundError, ValueError, OSError))
            self.assertTrue(str(exc).strip(), "Missing-file error had no message")
            return
        self.assertIsNotNone(result, "Checker returned nothing for a missing file")
        self.assertTrue(
            result.get("error") or "uncertain" in str(result.get("verdict", "")).lower(),
            f"Missing file did not produce a clear error: {result!r}",
        )

    def test_non_image_file_gives_clear_error(self):
        """A non-image file gives a clear error, not a crash."""
        bad = self.dir / "not-an-image.txt"
        bad.write_text("this is definitely not an image\n", encoding="utf-8")

        result, exc, proc = _check(bad)

        if proc is not None:
            combined = ((proc.stderr or "") + "\n" + (proc.stdout or "")).lower()
            self.assertNotIn("traceback", combined,
                             "Non-image file produced an unhandled traceback")
            self.assertNotEqual(proc.returncode, 0,
                                "Non-image file should not be reported as a success")
            self.assertTrue(
                any(word in combined for word in ("not an image", "invalid", "unsupported", "cannot", "error", "unrecognised", "unrecognized")),
                f"Non-image error message was not clear: {combined!r}",
            )
            return

        if exc is not None:
            self.assertIsInstance(exc, (ValueError, OSError))
            self.assertTrue(str(exc).strip(), "Non-image error had no message")
            return
        self.assertIsNotNone(result, "Checker returned nothing for a non-image file")
        self.assertTrue(
            result.get("error") or "uncertain" in str(result.get("verdict", "")).lower(),
            f"Non-image file did not produce a clear error: {result!r}",
        )

    def test_generator_metadata_does_not_score_lower(self):
        """Generator metadata must not lower the AI score.

        Tolerant by design, as the task requires: if the detector
        uses metadata, the metadata image should score at least as
        high for AI as the identical image without it. If it does
        not use metadata, the two scores should at least be equal.
        In both cases a verdict string must be returned.
        """
        base = self._solid_png("base.png")
        with_metadata = self._solid_png(
            "with-metadata.png",
            text_chunks=[
                ("Software", "Midjourney"),
                ("Generator", "Stable Diffusion"),
            ],
        )

        base_result, base_exc, _ = _check(base)
        meta_result, meta_exc, _ = _check(with_metadata)

        self.assertIsNone(base_exc, f"Plain image raised an error: {base_exc!r}")
        self.assertIsNone(meta_exc, f"Metadata image raised an error: {meta_exc!r}")
        self.assertIsNotNone(base_result)
        self.assertIsNotNone(meta_result)

        for label, result in (("base", base_result), ("metadata", meta_result)):
            verdict = result.get("verdict")
            self.assertIsInstance(verdict, str, f"{label} verdict was not a string: {verdict!r}")
            self.assertTrue(verdict.strip(), f"{label} verdict was empty")

        base_score = base_result.get("score")
        meta_score = meta_result.get("score")
        if base_score is not None and meta_score is not None:
            self.assertGreaterEqual(
                meta_score, base_score,
                f"Metadata image scored lower for AI ({meta_score}) than the "
                f"same image without metadata ({base_score})",
            )

    def test_plain_generated_image_returns_valid_structure(self):
        """A plain solid-colour PNG returns a valid verdict/score/reasons structure."""
        image = self._solid_png("solid.png", rgb=(120, 180, 60))
        result, exc, _ = _check(image)

        self.assertIsNone(exc, f"Plain image raised an error: {exc!r}")
        self.assertIsNotNone(result, "Checker returned nothing for a plain image")

        verdict = result.get("verdict")
        self.assertIsInstance(verdict, str, f"Verdict was not a string: {verdict!r}")
        self.assertTrue(verdict.strip(), "Verdict was empty")
        self.assertIn(
            verdict.strip().lower(),
            ("likely ai", "likely real", "uncertain", "ai", "real"),
            f"Unexpected verdict: {verdict!r}",
        )

        score = result.get("score")
        self.assertIsNotNone(score, "No score was returned for a plain image")
        self.assertIsInstance(score, (int, float), f"Score was not numeric: {score!r}")
        # Accept either a 0-1 or a 0-100 scale.
        self.assertTrue(0 <= score <= 100, f"Score out of range: {score!r}")

        reasons = result.get("reasons")
        self.assertIsInstance(reasons, list, f"Reasons was not a list: {reasons!r}")
        for reason in reasons:
            self.assertIsInstance(reason, str)

    # -- added coverage: direction, thresholds, errors, JSON ---------

    def test_generator_metadata_scores_strictly_higher_and_reaches_ai(self):
        """A generator text chunk raises the score, not just keeps it level.

        Same solid image, with and without a Midjourney tEXt chunk.
        The metadata version must score strictly higher, and with
        this checker's documented thresholds it must reach
        'likely AI-generated' (>= 75) while the plain version does not.
        """
        base_result, base_exc, _ = _check(self._solid_png("base2.png"))
        meta_result, meta_exc, _ = _check(
            self._solid_png(
                "meta2.png",
                text_chunks=[("Software", "Midjourney")],
            )
        )
        self.assertIsNone(base_exc)
        self.assertIsNone(meta_exc)
        self.assertGreater(meta_result["score"], base_result["score"],
                           "Generator metadata did not raise the AI score")
        self.assertGreaterEqual(meta_result["score"], 75)
        self.assertIn("likely ai", meta_result["verdict"].lower())
        self.assertNotIn("likely ai", base_result["verdict"].lower())

    def test_camera_metadata_scores_lower_and_reaches_real(self):
        """A camera-make text chunk lowers the score.

        Same solid image, with and without a Canon tEXt chunk. The
        camera version must score strictly lower, and with this
        checker's documented thresholds it must reach
        'likely real / camera photo' (<= 35).

        Note: the test colour is a low-saturation grey, so the
        saturation bonus does not apply and the arithmetic is
        base 65 (50 + 5 no-EXIF + 10 smooth) vs camera 35
        (50 - 25 camera + 10 smooth).
        """
        base_result, base_exc, _ = _check(
            self._solid_png("base3.png", rgb=(120, 120, 110)))
        cam_result, cam_exc, _ = _check(
            self._solid_png(
                "camera.png",
                rgb=(120, 120, 110),
                text_chunks=[("Make", "Canon")],
            )
        )
        self.assertIsNone(base_exc)
        self.assertIsNone(cam_exc)
        self.assertLess(cam_result["score"], base_result["score"],
                        "Camera metadata did not lower the AI score")
        self.assertLessEqual(cam_result["score"], 35)
        self.assertIn("likely real", cam_result["verdict"].lower())

    def test_software_only_scores_between_plain_and_camera(self):
        """Software-only provenance scores above plain, below generator.

        A Photoshop tEXt chunk with no camera make must not be
        treated like a camera photo, and must not by itself reach
        'likely AI-generated'.

        Note: low-saturation grey is used (see the camera test), so
        the arithmetic is plain 65 vs software-only 70
        (50 + 10 software + 10 smooth), both below the 75 threshold.
        """
        plain_result, _, _ = _check(
            self._solid_png("plain2.png", rgb=(120, 120, 110)))
        soft_result, soft_exc, _ = _check(
            self._solid_png(
                "software.png",
                rgb=(120, 120, 110),
                text_chunks=[("Software", "Adobe Photoshop")],
            )
        )
        self.assertIsNone(soft_exc)
        self.assertGreaterEqual(soft_result["score"], plain_result["score"])
        self.assertLess(soft_result["score"], 75,
                        "Software-only provenance alone reached 'likely AI'")

    def test_textured_image_scores_lower_than_smooth_image(self):
        """A high-texture checkerboard scores lower than a smooth solid image."""
        smooth_result, _, _ = _check(self._solid_png("smooth.png"))
        textured_path = _write_checkerboard_png(self.dir / "checker.png")
        textured_result, textured_exc, _ = _check(textured_path)
        self.assertIsNone(textured_exc)
        self.assertLess(textured_result["score"], smooth_result["score"],
                        "Textured image did not score lower than a smooth image")

    def test_empty_file_and_fake_image_extension_give_clear_errors(self):
        """Empty files and text renamed as .png give clear errors, not crashes."""
        empty = self.dir / "empty.png"
        empty.write_bytes(b"")
        result, exc, proc = _check(empty)
        if proc is not None:
            combined = ((proc.stderr or "") + "\n" + (proc.stdout or "")).lower()
            self.assertNotIn("traceback", combined)
            self.assertNotEqual(proc.returncode, 0)
        else:
            self.assertIsNotNone(exc or (result and result.get("error")))

        fake = self.dir / "fake.png"
        fake.write_text("this is text, not a PNG\n", encoding="utf-8")
        result, exc, proc = _check(fake)
        if proc is not None:
            combined = ((proc.stderr or "") + "\n" + (proc.stdout or "")).lower()
            self.assertNotIn("traceback", combined)
            self.assertNotEqual(proc.returncode, 0)
            self.assertTrue(
                any(w in combined for w in ("not a supported", "not an image", "error", "invalid", "unsupported")),
                f"Fake-image error was not clear: {combined!r}",
            )
        else:
            self.assertIsNotNone(exc or (result and result.get("error")))

    def test_truncated_png_does_not_crash(self):
        """A PNG header with no image data must not produce a traceback."""
        bad = self.dir / "truncated.png"
        bad.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)
        result, exc, proc = _check(bad)
        if proc is not None:
            combined = ((proc.stderr or "") + "\n" + (proc.stdout or "")).lower()
            self.assertNotIn("traceback", combined,
                             "Truncated PNG produced an unhandled traceback")
            # Either a clear error (exit 2) or a handled uncertain
            # verdict is acceptable; a crash is not.
            if proc.returncode == 0:
                self.assertIsNotNone(result)
        else:
            # Function path: an exception with a message, or a result.
            self.assertTrue(exc is not None or result is not None)

    def test_json_output_carries_metadata_and_score(self):
        """--json output carries format, metadata keywords, score and reasons."""
        image = self._solid_png(
            "json.png",
            text_chunks=[("Software", "Midjourney")],
        )
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), str(image), "--json"],
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(proc.returncode, 0, f"CLI failed: {proc.stderr!r}")
        self.assertNotIn("traceback", (proc.stdout + proc.stderr).lower())
        data = json.loads(proc.stdout)
        self.assertEqual(data["format"], "PNG")
        self.assertIn("score", data)
        self.assertTrue(0 <= data["score"] <= 100)
        self.assertIn("likely ai", str(data["verdict"]).lower())
        self.assertIsInstance(data["reasons"], list)
        self.assertTrue(data["reasons"])
        self.assertTrue(all(isinstance(r, str) for r in data["reasons"]))
        # Metadata block, when present, must name what was found.
        metadata = data.get("metadata") or {}
        generator = metadata.get("generator_keywords") or []
        self.assertTrue(any("midjourney" in str(g).lower() for g in generator),
                        f"JSON metadata did not report Midjourney: {metadata!r}")

    def test_same_image_is_deterministic(self):
        """Checking the same file twice gives the same score and verdict."""
        image = self._solid_png("twice.png")
        first, exc1, _ = _check(image)
        second, exc2, _ = _check(image)
        self.assertIsNone(exc1)
        self.assertIsNone(exc2)
        self.assertEqual(first["score"], second["score"])
        self.assertEqual(first["verdict"], second["verdict"])


if __name__ == "__main__":
    unittest.main()
