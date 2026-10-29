import json
from pathlib import Path
from .data import write_jsonl, stable_id


def build(output: Path):
    departments = ["Data Science", "Cybersecurity", "Software Engineering", "Robotics"]
    technologies = ["Vector Search", "Graph Databases", "Cloud Computing", "Computer Vision"]
    projects = ["Atlas", "Beacon", "Cobalt", "Delta"]
    chunks, triples = [], []
    for i, department in enumerate(departments):
        text = (f"The {department} department teaches practical methods and supervises projects. "
                f"Its projects use {technologies[i]} and focus on reliable domain solutions.")
        cid = stable_id("department-" + department, text)
        chunks.append({"id": cid, "title": department, "text": text, "dataset": "synthetic", "sentence_index": 0})
        triples.append({"chunk_id": cid, "triples": [
            {"subject": department, "relation": "USES", "object": technologies[i], "evidence": f"Its projects use {technologies[i]}"},
            {"subject": department, "relation": "SUPERVISES", "object": projects[i], "evidence": "supervises projects"}
        ], "model": "deterministic-synthetic"})
    for i, project in enumerate(projects):
        text = (f"Project {project} is a university research project. "
                f"It applies {technologies[i]} to improve {departments[i]} workflows and reporting.")
        cid = stable_id("project-" + project, text)
        chunks.append({"id": cid, "title": "Project " + project, "text": text, "dataset": "synthetic", "sentence_index": 0})
        triples.append({"chunk_id": cid, "triples": [
            {"subject": project, "relation": "USES", "object": technologies[i], "evidence": f"It applies {technologies[i]}"},
            {"subject": project, "relation": "SUPPORTS", "object": departments[i], "evidence": f"improve {departments[i]} workflows"}
        ], "model": "deterministic-synthetic"})
    for j in range(92):
        department, technology = f"Reference Lab {j}", technologies[j % 4]
        text = f"Reference note {j}: {department} evaluates {technology} for reproducible project work."
        cid = stable_id(f"reference-{j}", text)
        chunks.append({"id": cid, "title": f"Reference note {j}", "text": text, "dataset": "synthetic", "sentence_index": 0})
        triples.append({"chunk_id": cid, "triples": [{"subject": department, "relation": "EVALUATES", "object": technology, "evidence": f"{department} evaluates {technology}"}], "model": "deterministic-synthetic"})
    questions = []
    for i, department in enumerate(departments):
        cid = chunks[i]["id"]
        questions.append({"id": f"synthetic-vector-{i}", "question": f"What does the {department} department focus on?", "answers": ["practical methods"], "gold_chunk_ids": [cid], "dataset": "synthetic", "category": "vector", "expected_route": "vector"})
        questions.append({"id": f"synthetic-graph-{i}", "question": f"Which technology does the {department} department use?", "answers": [technologies[i]], "gold_chunk_ids": [cid], "dataset": "synthetic", "category": "graph", "expected_route": "graph"})
        questions.append({"id": f"synthetic-hybrid-{i}", "question": f"How does the {department} department use {technologies[i]} in its project?", "answers": [technologies[i]], "gold_chunk_ids": [cid, chunks[4+i if 4+i < len(chunks) else i]["id"]], "dataset": "synthetic", "category": "hybrid", "expected_route": "hybrid"})
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "chunks.jsonl", chunks)
    write_jsonl(output / "extractions.jsonl", triples)
    write_jsonl(output / "development.jsonl", questions[:8])
    write_jsonl(output / "evaluation.jsonl", questions[8:])
    manifest = {"synthetic": True, "chunks": len(chunks), "questions": {"development": 8, "evaluation": len(questions)-8}, "note": "Deterministic relationships; no LLM extraction required."}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
