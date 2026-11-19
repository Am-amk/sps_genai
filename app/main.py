from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

import io
import torch
from PIL import Image
from torchvision.utils import make_grid

from app.bigram_model import BigramModel
from app.embedding_model import EmbeddingModel
from app.image_classifier import ImageClassifier
from app.digit_generator import DigitGenerator

from app.energy_model import EnergyModel, generate_samples
from app.diffusion_model import (
    UNet,
    DiffusionModel,
    offset_cosine_diffusion_schedule,
)

app = FastAPI(title="SPS GenAI API")


# -----------------------------
# Sample corpus for bigram model
# -----------------------------
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
    "It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective",
]


# -----------------------------
# Existing models
# -----------------------------
bigram_model = BigramModel(corpus)
embedding_model = EmbeddingModel()
image_classifier = ImageClassifier()
digit_generator = DigitGenerator()


# -----------------------------
# Energy Model
# -----------------------------
device = torch.device("cpu")

energy_model = EnergyModel().to(device)

energy_model.load_state_dict(
    torch.load(
        "app/energy_cifar10.pth",
        map_location=device
    )
)

energy_model.eval()
# -----------------------------
# Diffusion Model
# -----------------------------
unet = UNet(
    image_size=32,
    num_channels=3,
    embedding_dim=32
)

diffusion_model = DiffusionModel(
    unet,
    offset_cosine_diffusion_schedule
)

checkpoint = torch.load(
    "app/diffusion_cifar10.pth",
    map_location=device
)

diffusion_model.network.load_state_dict(
    checkpoint["network"]
)

diffusion_model.ema_network.load_state_dict(
    checkpoint["ema_network"]
)

diffusion_model.set_normalizer(
    checkpoint["normalizer_mean"],
    checkpoint["normalizer_std"]
)

diffusion_model.to(device)
diffusion_model.eval()

# -----------------------------
# Helper: convert tensors to PNG
# -----------------------------
def tensor_grid_to_png(images):
    grid = make_grid(
        images.cpu(),
        nrow=min(4, images.size(0)),
        normalize=True
    )

    grid = (
        grid.permute(1, 2, 0)
        .clamp(0, 1)
        .mul(255)
        .byte()
        .numpy()
    )

    image = Image.fromarray(grid)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    return buffer.getvalue()


# -----------------------------
# Request models
# -----------------------------
class TextGenerationRequest(BaseModel):
    start_word: str
    length: int = Field(10, ge=1, le=200)


class EmbeddingRequest(BaseModel):
    word: str = Field(..., min_length=1, examples=["apple"])


class SimilarityRequest(BaseModel):
    word1: str = Field(..., min_length=1, examples=["apple"])
    word2: str = Field(..., min_length=1, examples=["orange"])


# -----------------------------
# Existing API endpoints
# -----------------------------
@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(
        request.start_word,
        request.length
    )

    return {
        "generated_text": generated_text
    }


@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    if not embedding_model.has_vector(request.word):
        raise HTTPException(
            status_code=404,
            detail=f"No embedding found for '{request.word}'"
        )

    vector = embedding_model.embed(
        request.word
    )

    return {
        "word": request.word,
        "dimension": len(vector),
        "embedding": vector
    }


@app.post("/similarity")
def get_similarity(request: SimilarityRequest):
    score = embedding_model.similarity(
        request.word1,
        request.word2
    )

    return {
        "word1": request.word1,
        "word2": request.word2,
        "similarity": score
    }


@app.post("/classify")
async def classify_image(
    file: UploadFile = File(...)
):
    image_bytes = await file.read()

    try:
        return image_classifier.predict(
            image_bytes
        )

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not read the uploaded file as an image"
        )


@app.get(
    "/generate_digits",
    responses={
        200: {
            "content": {
                "image/png": {}
            }
        }
    }
)
def generate_digits(
    num_images: int = Query(
        16,
        ge=1,
        le=64,
        description="Number of digits to generate"
    ),
    seed: int | None = Query(
        None,
        description="Optional random seed for reproducible output"
    ),
):
    png = digit_generator.generate_png(
        num_images=num_images,
        seed=seed
    )

    return Response(
        content=png,
        media_type="image/png"
    )

# -----------------------------
# Energy Model endpoint
# -----------------------------
@app.get(
    "/generate_energy",
    responses={
        200: {
            "content": {
                "image/png": {}
            }
        }
    }
)
def generate_energy_images(
    num_images: int = Query(
        4,
        ge=1,
        le=16,
        description="Number of CIFAR-10-like images to generate"
    )
):
    initial_images = (
        torch.rand(
            num_images,
            3,
            32,
            32,
            device=device
        ) * 2 - 1
    )

    samples = generate_samples(
        energy_model,
        initial_images,
        steps=60,
        step_size=10.0,
        noise_std=0.005
    )

    png = tensor_grid_to_png(samples)

    return Response(
        content=png,
        media_type="image/png"
    )


# -----------------------------
# Diffusion Model endpoint
# -----------------------------
@app.get(
    "/generate_diffusion",
    responses={
        200: {
            "content": {
                "image/png": {}
            }
        }
    }
)
def generate_diffusion_images(
    num_images: int = Query(
        4,
        ge=1,
        le=16,
        description="Number of CIFAR-10-like images to generate"
    ),
    diffusion_steps: int = Query(
        50,
        ge=10,
        le=200,
        description="Number of reverse diffusion steps"
    )
):
    samples = diffusion_model.generate(
        num_images=num_images,
        diffusion_steps=diffusion_steps,
        image_size=32
    )

    png = tensor_grid_to_png(samples)

    return Response(
        content=png,
        media_type="image/png"
    )
# -----------------------------
# Energy Model endpoint
# -----------------------------
@app.get(
    "/generate_diffusion",
    responses={
        200: {
            "content": {
                "image/png": {}
            }
        }
    }
)
def generate_diffusion_images(
    num_images: int = Query(
        4,
        ge=1,
        le=16,
        description="Number of CIFAR-10-like images to generate"
    ),
    diffusion_steps: int = Query(
        50,
        ge=10,
        le=200,
        description="Number of reverse diffusion steps"
    )
):
    samples = diffusion_model.generate(
        num_images=num_images,
        diffusion_steps=diffusion_steps,
        image_size=32
    )

    png = tensor_grid_to_png(samples)

    return Response(
        content=png,
        media_type="image/png"
    )