from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app.bigram_model import BigramModel
from app.embedding_model import EmbeddingModel
from app.image_classifier import ImageClassifier
from app.digit_generator import DigitGenerator

app = FastAPI(title="SPS GenAI API")

# Sample corpus for the bigram model
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
    "It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective",
]

bigram_model = BigramModel(corpus)
embedding_model = EmbeddingModel()
image_classifier = ImageClassifier()
digit_generator = DigitGenerator()


class TextGenerationRequest(BaseModel):
    start_word: str
    length: int = Field(10, ge=1, le=200)


class EmbeddingRequest(BaseModel):
    word: str = Field(..., min_length=1, examples=["apple"])


class SimilarityRequest(BaseModel):
    word1: str = Field(..., min_length=1, examples=["apple"])
    word2: str = Field(..., min_length=1, examples=["orange"])


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}


@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    if not embedding_model.has_vector(request.word):
        raise HTTPException(status_code=404, detail=f"No embedding found for '{request.word}'")
    vector = embedding_model.embed(request.word)
    return {"word": request.word, "dimension": len(vector), "embedding": vector}


@app.post("/similarity")
def get_similarity(request: SimilarityRequest):
    score = embedding_model.similarity(request.word1, request.word2)
    return {"word1": request.word1, "word2": request.word2, "similarity": score}


@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    """Upload an image; returns its predicted CIFAR-10 class."""
    image_bytes = await file.read()
    try:
        return image_classifier.predict(image_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the uploaded file as an image")


@app.get("/generate_digits", responses={200: {"content": {"image/png": {}}}})
def generate_digits(
    num_images: int = Query(16, ge=1, le=64, description="Number of digits to generate"),
    seed: int | None = Query(None, description="Optional random seed for reproducible output"),
):
    """Generate hand-written digits with the GAN trained on MNIST; returns a PNG grid."""
    png = digit_generator.generate_png(num_images=num_images, seed=seed)
    return Response(content=png, media_type="image/png")
