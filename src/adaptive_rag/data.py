"""Dataset adapters. Labels and reference answers never enter the search index."""
import hashlib
import json
from pathlib import Path
import random
import re
import urllib.request

URLS = {
    "squad_train": "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json",
    "squad_dev": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json",
    "hotpot_train": "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/distractor/train-00000-of-00002.parquet",
    "hotpot_dev": "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/distractor/validation-00000-of-00001.parquet",
}


def stable_id(title, text):
    return hashlib.sha256((title.strip() + "\n" + text.strip()).encode()).hexdigest()[:24]


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def download(name, directory):
    if name not in URLS:
        raise ValueError(f"Unknown dataset {name}")
    extension = ".parquet" if URLS[name].endswith(".parquet") else ".json"
    target = Path(directory) / (name + extension)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        temp = target.with_suffix(".download")
        with urllib.request.urlopen(URLS[name], timeout=120) as response, temp.open("wb") as output:
            while block := response.read(1024 * 1024):
                output.write(block)
        # Validate before replacing the destination.
        if extension == ".json":
            json.loads(temp.read_text())
        else:
            import pyarrow.parquet as pq
            pq.read_metadata(temp)
        temp.replace(target)
    return target


def sentence_spans(text):
    # Stable sentence units ensure chunk size stays inside the small encoder budget.
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def units(title, sentences, dataset):
    result = []
    for sentence_index, sentence in enumerate(sentences):
        words = sentence.split()
        ids = []
        # Conservatively small word windows; references cover all pieces of a sentence.
        for offset in range(0, len(words), 70):
            text = " ".join(words[offset:offset + 70])
            identifier = stable_id(title, text)
            ids.append(identifier)
            result.append({"id": identifier, "title": title, "text": text,
                           "dataset": dataset, "sentence_index": sentence_index})
        yield ids, result
        result = []


def squad_records(raw):
    for article in raw["data"]:
        for paragraph in article["paragraphs"]:
            # Keep offsets for precise answer-to-evidence mapping.
            context = paragraph["context"]
            sentences = sentence_spans(context)
            chunks, locations = [], []
            cursor = 0
            for sentence, (ids, pieces) in zip(sentences, units(article["title"], sentences, "squad")):
                start = context.find(sentence, cursor)
                end = start + len(sentence)
                locations.append((start, end, ids))
                chunks.extend(pieces)
                cursor = end
            for qa in paragraph["qas"]:
                gold = set()
                for answer in qa["answers"]:
                    a, b = answer["answer_start"], answer["answer_start"] + len(answer["text"])
                    for start, end, ids in locations:
                        if start < b and a < end:
                            gold.update(ids)
                if not gold:
                    continue
                yield chunks, {"id": "squad:" + qa["id"], "question": qa["question"],
                               "answers": list(dict.fromkeys(a["text"] for a in qa["answers"])),
                               "gold_chunk_ids": sorted(gold), "dataset": "squad",
                               "category": "single_passage", "expected_route": None}


def hotpot_records(raw):
    for qa in raw:
        if isinstance(qa["context"], dict):
            qa = dict(qa)
            qa["context"] = list(zip(qa["context"]["title"], qa["context"]["sentences"]))
            qa["supporting_facts"] = list(zip(qa["supporting_facts"]["title"], qa["supporting_facts"]["sent_id"]))
            qa["_id"] = qa["id"]
        chunks, lookup = [], {}
        for title, sentences in qa["context"]:
            for index, (ids, pieces) in enumerate(units(title, sentences, "hotpot")):
                lookup[(title, index)] = ids
                chunks.extend(pieces)
        gold = set()
        for title, index in qa["supporting_facts"]:
            gold.update(lookup.get((title, index), []))
        if not gold or any((t, i) not in lookup for t, i in qa["supporting_facts"]):
            continue
        yield chunks, {"id": "hotpot:" + qa["_id"], "question": qa["question"],
                       "answers": [qa["answer"]], "gold_chunk_ids": sorted(gold),
                       "dataset": "hotpot", "category": qa["type"], "expected_route": None}


def prepare(inputs: list[tuple[str, str, Path]], output: Path, per_dataset=150, seed=42):
    """Split by source's published train/dev; sample reproducibly, deduplicate evidence."""
    if per_dataset < 1:
        raise ValueError("per_dataset must be positive")
    corpus, questions = {}, {"development": [], "evaluation": []}
    selected_ids = set()
    for dataset, split, path in inputs:
        if Path(path).suffix == ".parquet":
            import pyarrow.parquet as pq
            raw = pq.read_table(path).to_pylist()
        else:
            raw = json.loads(Path(path).read_text())
        records = list(squad_records(raw) if dataset == "squad" else hotpot_records(raw))
        random.Random(f"{seed}:{dataset}:{split}").shuffle(records)
        chosen = 0
        for chunks, question in records:
            key = " ".join(question["question"].lower().split())
            if key in selected_ids:
                continue
            selected_ids.add(key)
            for chunk in chunks:
                corpus.setdefault(chunk["id"], chunk)
            question["split"] = split
            questions[split].append(question)
            chosen += 1
            if chosen == per_dataset:
                break
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "chunks.jsonl", sorted(corpus.values(), key=lambda c: c["id"]))
    for split, rows in questions.items():
        write_jsonl(output / f"{split}.jsonl", rows)
    manifest = {"seed": seed, "per_dataset_requested": per_dataset, "chunks": len(corpus),
                "questions": {k: len(v) for k, v in questions.items()},
                "inputs": [{"dataset": d, "split": s, "file": p.name,
                            "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for d, s, p in inputs],
                "note": "Published train -> development; published dev -> local evaluation. Initial Hotpot training download uses shard 0 of 2. No route labels inferred from dataset."}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def make_pilot(source: Path, output: Path, chunk_limit=200, seed=42):
    if chunk_limit < 1:
        raise ValueError("chunk_limit must be positive")
    chunks = read_jsonl(source / "chunks.jsonl")
    development = read_jsonl(source / "development.jsonl")
    evaluation = read_jsonl(source / "evaluation.jsonl")
    rng = random.Random(seed)
    selected = {}
    selected_questions = set()
    candidates = development + evaluation
    rng.shuffle(candidates)
    for question in candidates:
        needed = [c for c in chunks if c["id"] in question["gold_chunk_ids"]]
        if len(selected) + len(needed) > chunk_limit:
            continue
        for chunk in needed:
            selected[chunk["id"]] = chunk
        selected_questions.add(question["id"])
        if len(selected) == chunk_limit:
            break
    for chunk in chunks:
        if len(selected) >= chunk_limit:
            break
        selected.setdefault(chunk["id"], chunk)
    allowed = set(selected)
    def retain(rows):
        return [q for q in rows if q["id"] in selected_questions and set(q["gold_chunk_ids"]) <= allowed]
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "chunks.jsonl", sorted(selected.values(), key=lambda c: c["id"]))
    write_jsonl(output / "development.jsonl", retain(development))
    write_jsonl(output / "evaluation.jsonl", retain(evaluation))
    manifest = {"seed": seed, "pilot": True, "chunks": len(selected),
                "questions": {"development": len(retain(development)), "evaluation": len(retain(evaluation))},
                "note": "Controlled subset preserving complete gold evidence for retained questions."}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
