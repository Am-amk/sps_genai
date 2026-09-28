# SPS GenAI API

FastAPI app with a bigram text generator and spaCy word embeddings.

## Run with Docker

```
docker build -t sps-genai .
docker run -p 8000:80 sps-genai
```

Then open http://127.0.0.1:8000/docs

## Run without Docker

```
uv sync
uv run fastapi dev app/main.py
```

## Endpoints

- `POST /generate`: `{"start_word": "the", "length": 10}` returns generated text
- `POST /embedding`: `{"word": "apple"}` returns a 300-dimension word vector
- `POST /similarity`: `{"word1": "apple", "word2": "orange"}` returns cosine similarity