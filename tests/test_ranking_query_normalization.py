from sim_agent.config import RankingWeights
from sim_agent.ranking import rank_papers
from sim_agent.types import PaperMetadata


def test_hyphenated_query_matches_space_separated_title():
    papers = [
        PaperMetadata(
            paper_id="p1",
            title="In Silico Design and Analysis of Plastic Binding Peptides",
            abstract="Molecular simulation of peptide binding on plastics.",
            year=2023,
            citation_count=5,
        ),
        PaperMetadata(
            paper_id="p2",
            title="Hydrogen diffusion on metal surfaces",
            abstract="Unrelated adsorption system.",
            year=2024,
            citation_count=10,
        ),
    ]
    ranked = rank_papers(
        papers=papers,
        query="plastic-binding compounds simulation",
        weights=RankingWeights(),
        years_window=20,
    )
    assert ranked[0].metadata.paper_id == "p1"
