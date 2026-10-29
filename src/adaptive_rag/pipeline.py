from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from time import perf_counter
from .core import RuleRouter, rrf

MODES = ("vector", "graph", "static_hybrid", "adaptive")


class Pipeline:
    def __init__(self, vector, graph, generator, top_k=5, router=None):
        self.vector, self.graph, self.generator = vector, graph, generator
        self.top_k = top_k
        self.router = router or RuleRouter()

    def run(self, question, mode="adaptive", generate=True, fallback=False):
        if mode not in MODES:
            raise ValueError(f"Mode must be one of {MODES}")
        if not question.strip():
            raise ValueError("Enter a non-empty question")
        started = perf_counter()
        routed = perf_counter()
        decision = self.router.route(question) if mode == "adaptive" else None
        route = decision.route if decision else ("hybrid" if mode == "static_hybrid" else mode)
        routing_ms = (perf_counter() - routed) * 1000
        calls = {"vector": 0, "graph": 0}
        branch_ms, errors, fallback_used = {}, {}, False

        def retrieve(name):
            calls[name] += 1
            before = perf_counter()
            try:
                store = self.vector if name == "vector" else self.graph
                if store is None:
                    raise RuntimeError(f"{name} database is not initialized")
                return store.search(question, self.top_k)
            except Exception as exc:
                errors[name] = str(exc)
                return []
            finally:
                branch_ms[name] = (perf_counter() - before) * 1000

        retrieval_start = perf_counter()
        if route == "hybrid":
            # Static and adaptive hybrid use identical parallel retrieval behavior.
            with ThreadPoolExecutor(max_workers=2) as pool:
                v = pool.submit(retrieve, "vector")
                g = pool.submit(retrieve, "graph")
                hits = rrf([v.result(), g.result()], self.top_k)
        else:
            hits = retrieve(route)
        if fallback and mode == "adaptive" and route != "hybrid" and not hits:
            fallback_used = True
            hits = retrieve("graph" if route == "vector" else "vector")
        retrieval_ms = (perf_counter() - retrieval_start) * 1000
        answer, usage, generation_error = None, {}, None
        before_generation = perf_counter()
        if generate:
            try:
                answer, usage = self.generator.answer(question, hits)
            except Exception as exc:
                generation_error = str(exc)
        generation_ms = (perf_counter() - before_generation) * 1000 if generate else 0
        return {"question": question, "mode": mode, "selected_route": route,
                "router_version": self.router.version, "decision": asdict(decision) if decision else None,
                "hits": [h.to_dict() for h in hits], "answer": answer, "usage": usage,
                "database_calls": calls, "branch_ms": branch_ms,
                "routing_ms": routing_ms, "retrieval_ms": retrieval_ms,
                "generation_ms": generation_ms, "total_ms": (perf_counter() - started) * 1000,
                "fallback_used": fallback_used, "retrieval_errors": errors,
                "generation_error": generation_error,
                "status": "error" if errors or generation_error else ("no_evidence" if not hits else "ok")}
