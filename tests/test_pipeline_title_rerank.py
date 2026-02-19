from sim_agent.pipeline import _expand_topic_keywords, _select_top_by_title_with_llm
from sim_agent.types import PaperMetadata, RankedPaper


class _StubLLM:
    def __init__(self, response: dict):
        self.enabled = True
        self._response = response

    def chat_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.0, max_tokens: int = 1200):
        return self._response


def test_expand_topic_keywords_uses_llm_output():
    llm = _StubLLM({"keywords": ["semicrystalline polymer", "molecular dynamics", "crystallization kinetics"]})
    keywords = _expand_topic_keywords("molecular simulation on polymer considering crystallinity", llm)
    assert "molecular simulation on polymer considering crystallinity" in keywords
    assert "semicrystalline polymer" in keywords
    assert "molecular dynamics" in keywords


def test_select_top_by_title_with_llm_and_fill():
    ranked = [
        RankedPaper(metadata=PaperMetadata(paper_id=f"p{i}", title=f"Title {i}", abstract=""), score=1.0 / i)
        for i in range(1, 6)
    ]
    llm = _StubLLM({"selected_refs": ["R002", "R004"]})
    selected = _select_top_by_title_with_llm("polymer crystallinity", ranked, top_n=3, llm_client=llm)
    assert [paper.metadata.paper_id for paper in selected] == ["p2", "p4", "p1"]
