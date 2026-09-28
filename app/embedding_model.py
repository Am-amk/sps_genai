import spacy


class EmbeddingModel:
    """Word embeddings using spaCy (Module 2, Practical 3)."""

    def __init__(self, model_name: str = "en_core_web_lg"):
        # Load the model once, when the app starts
        self.nlp = spacy.load(model_name)

    def embed(self, word: str) -> list[float]:
        # .tolist() turns the numpy array into a normal list so FastAPI can send it as JSON
        return self.nlp(word).vector.tolist()

    def has_vector(self, word: str) -> bool:
        # False for made-up words that spaCy doesn't know
        return self.nlp(word).has_vector

    def similarity(self, word1: str, word2: str) -> float:
        return float(self.nlp(word1).similarity(self.nlp(word2)))