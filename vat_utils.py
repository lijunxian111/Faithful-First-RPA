from __future__ import annotations

import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Callable, Iterator, List
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen

from datasets import load_dataset
from PIL import Image, ImageFilter


# --- Image transforms ---------------------------------------------------------

def to_edge_sketch(img: Image.Image) -> Image.Image:
    """Convert an RGB image to a binary edge sketch."""
    gray = img.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    bw = edges.point(lambda x: 255 if x > 50 else 0)
    return bw.convert("RGB")


def to_binarized(img: Image.Image) -> Image.Image:
    """Convert an RGB image to a simple black/white abstraction."""
    gray = img.convert("L")
    bw = gray.point(lambda x: 255 if x > 128 else 0)
    return bw.convert("RGB")


def generate_vat_variants(image: Image.Image) -> List[Image.Image]:
    """
    Produce the VAT image set (original + abstracted variants).
    Returns a list ordered as [original, edge sketch, binarised].
    """
    if image.mode != "RGB":
        base = image.convert("RGB")
    else:
        base = image.copy()
    return [base, to_edge_sketch(base), to_binarized(base)]


VAT_PROMPT_TEMPLATE = (
    "Question: {question}\n"
    "You are a visual reasoning agent. You see both the full image and its abstract version(s).\n"
    "The abstract images preserve key structural, relational, and contour information, and help you focus on essential visual elements.\n"
    "Please use both the original and abstract inputs to decide the correct answer.\n"
    "Think step by step. Steps should be separated by \\n\\n."
)


def build_vat_prompt(question: str) -> str:
    return VAT_PROMPT_TEMPLATE.format(question=question.strip())


def sanitize_realworldqa_question(question: str) -> str:
    stripped = question.replace(
        "Please answer directly with only the letter of the correct option and nothing else.", ""
    )
    return stripped.strip()


def image_to_bytes(image: Image.Image, format: str = "PNG") -> BytesIO:
    buffer = BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    return buffer


# --- Dataset helpers ----------------------------------------------------------

@dataclass
class VATSample:
    sample_id: int | str
    question: str
    answer: str
    image: Image.Image


class VATIterable:
    def __init__(self, total: int, iterator_factory: Callable[[], Iterator[VATSample]]):
        self.total = total
        self._iterator_factory = iterator_factory

    def __iter__(self) -> Iterator[VATSample]:
        return self._iterator_factory()


def _resolve_pope_path(data_path: str | Path | None = None) -> Path:
    """
    Find a POPE eval JSONL file. Tries the provided path first, then a few common fallbacks.
    """
    candidates = []
    if data_path:
        candidates.append(Path(data_path))
    candidates.extend(
        [
            Path("/home/ubuntu/codebases/cot-faithful/POPE/eval_data.jsonl"),
            Path("train_pope/eval_data_pope.jsonl"),
            Path("train_pope/eval_data.jsonl"),
            Path("POPE/eval_data.jsonl"),
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"POPE eval data file not found. Searched: {searched}")


def _resolve_pope_idxes_path(index_path: str | Path | None = None) -> Path:
    """
    Locate the POPE idx list. Defaults to the repository-level POPE/idxes.txt.
    """
    candidates = []
    if index_path:
        candidates.append(Path(index_path))
    candidates.extend(
        [
            Path("/home/ubuntu/codebases/cot-faithful/POPE/idxes.txt"),
            Path("POPE/idxes.txt"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"POPE idx list not found. Searched: {searched}")


def _load_pope_indices(index_path: str | Path | None = None) -> set[int]:
    idx_path = _resolve_pope_idxes_path(index_path)
    allowed: set[int] = set()
    with idx_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                allowed.add(int(line))
            except ValueError:
                continue
    if not allowed:
        raise ValueError(f"POPE idx list at '{idx_path}' is empty or invalid.")
    return allowed


def _coerce_int(value, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _load_pope_image(raw_image) -> Image.Image:
    if isinstance(raw_image, Image.Image):
        return raw_image.convert("RGB")
    if isinstance(raw_image, (str, Path)):
        return Image.open(raw_image).convert("RGB")
    if isinstance(raw_image, dict) and "bytes" in raw_image:
        return Image.open(BytesIO(raw_image["bytes"])).convert("RGB")
    raise ValueError(f"Unsupported image payload in POPE record: {type(raw_image)!r}")


def iterate_pope(data_path: str | Path | None = None, index_path: str | Path | None = None):
    """
    Iterate over locally dumped POPE eval data (JSONL with image/question/answer fields).
    """
    path = _resolve_pope_path(data_path)
    allowed_indices = _load_pope_indices(index_path)
    records: List[tuple[int, dict]] = []

    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample_id = record.get("idx", record.get("id", idx))
            numeric_id = _coerce_int(sample_id, idx)
            if numeric_id not in allowed_indices:
                continue
            records.append((numeric_id, record))

    total = len(records)

    def _iterator() -> Iterator[VATSample]:
        for numeric_id, record in records:
            sample_id = record.get("idx", record.get("id", numeric_id))
            raw_image = record.get("image") or record.get("image_path")
            if raw_image is None:
                continue
            try:
                image = _load_pope_image(raw_image)
            except Exception:
                continue

            yield VATSample(
                sample_id=sample_id,
                question=record.get("question", ""),
                answer=record.get("answer", ""),
                image=image.convert("RGB"),
            )

    return VATIterable(total, _iterator)


def iterate_llava_bench():
    dataset = load_dataset("lmms-lab/LLaVA-Bench-in-the-Wild", split="train")
    total = len(dataset)

    def _iterator() -> Iterator[VATSample]:
        for item in dataset:
            image = item["image"]
            if not isinstance(image, Image.Image):
                image = Image.open(BytesIO(image["bytes"])).convert("RGB")
            yield VATSample(
                sample_id=item.get("question_id", item.get("idx", "")),
                question=item.get("question", ""),
                answer=item.get("caption", ""),
                image=image.convert("RGB"),
            )

    return VATIterable(total, _iterator)


def iterate_realworldqa():
    dataset = load_dataset("xai-org/RealworldQA", split="test")
    total = len(dataset)

    def _iterator() -> Iterator[VATSample]:
        for idx, item in enumerate(dataset):
            image = item["image"]
            if not isinstance(image, Image.Image):
                image = Image.open(BytesIO(image["bytes"])).convert("RGB")
            question = sanitize_realworldqa_question(item.get("question", ""))
            answer = item.get("answer", "")
            yield VATSample(sample_id=idx, question=question, answer=answer, image=image.convert("RGB"))

    return VATIterable(total, _iterator)


def iterate_mmhal_bench(data_path: str | Path = Path("mmhal-bench/mmhal-bench.jsonl")):
    """
    Iterate over a local MMHal-Bench dump stored as JSON or JSONL.
    This loader expects entries matching the public MMHal schema, including remote `image_src` links.
    When a referenced image is not already cached locally, the loader will attempt to download it on demand.
    """
    path = Path(data_path)
    if not path.exists():
        alternatives = [
            Path("mmhal-bench.jsonl"),
            Path("mmhal-bench/mmhal-bench.json"),
            Path("mmhal-bench.json"),
        ]
        for candidate in alternatives:
            if candidate.exists():
                path = candidate
                break
        else:
            raise FileNotFoundError(
                f"MMHal-Bench data file not found. Checked {[str(p) for p in [data_path, *alternatives]]}"
            )

    records: List[dict] = []
    raw_text = path.read_text(encoding="utf-8")

    def _load_as_json(text: str) -> List[dict]:
        payload = json.loads(text)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("data", "items", "samples", "annotations"):
                value = payload.get(key)
                if isinstance(value, list):
                    return value
            return [payload]
        raise TypeError("Unsupported MMHal-Bench file format; expected JSON object or list.")

    try:
        records = _load_as_json(raw_text)
    except json.JSONDecodeError:
        # Fallback to line-wise JSONL parsing
        for line in raw_text.splitlines():
            line = line.strip()
            if not line or line in {"[", "]", ","}:
                continue
            if line.endswith(","):
                line = line[:-1]
            records.append(json.loads(line))

    total = len(records)

    def _normalise_question(item: dict) -> str:
        return item.get("question") or item.get("prompt") or ""

    def _normalise_answer(item: dict) -> str:
        for key in ("gt_answer", "answer", "label"):
            value = item.get(key, "")
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    def _load_image(item: dict) -> Image.Image | None:
        if "image" in item and isinstance(item["image"], Image.Image):
            return item["image"]

        raw = item.get("image")
        if isinstance(raw, dict) and "bytes" in raw:
            return Image.open(BytesIO(raw["bytes"]))

        if isinstance(raw, bytes):
            return Image.open(BytesIO(raw))

        image_id = item.get("image_id")
        candidates: List[Path | str] = []
        for key in ("image_path", "image_local", "image_src", "image_url"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                candidates.append(value.strip())

        if image_id:
            base_dir = Path("mmhal-bench").resolve()
            for ext in ("jpg", "jpeg", "png", "webp"):
                candidates.append(base_dir / f"{image_id}.{ext}")
                candidates.append(base_dir / "images" / f"{image_id}.{ext}")

        for candidate in candidates:
            if isinstance(candidate, Path):
                if candidate.exists():
                    return Image.open(candidate)
                continue

            parsed = urlparse(candidate)
            if parsed.scheme in {"http", "https"}:
                try:
                    with urlopen(candidate) as response:
                        content = response.read()
                    return Image.open(BytesIO(content))
                except (URLError, OSError):
                    continue
            else:
                local_path = Path(candidate)
                if local_path.exists():
                    return Image.open(local_path)

        return None

    def _iterator() -> Iterator[VATSample]:
        for idx, item in enumerate(records):
            image = _load_image(item)
            if image is None:
                # Skip samples whose image asset cannot be resolved.
                continue
            sample_id = item.get("question_id") or item.get("id") or idx
            yield VATSample(
                sample_id=sample_id,
                question=_normalise_question(item),
                answer=_normalise_answer(item),
                image=image.convert("RGB"),
            )

    return VATIterable(total, _iterator)
