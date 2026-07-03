from voiceanalysis.config import AnalysisConfig
from voiceanalysis.embeddings import cosine_similarity


def test_config_defaults():
    cfg = AnalysisConfig()
    assert cfg.sample_rate == 16000
    assert cfg.n_mfcc == 20


def test_cosine_similarity_identity():
    assert cosine_similarity([1, 0, 0], [1, 0, 0]) > 0.999
