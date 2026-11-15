# LlamaIndex Embeddings Integration: Voyage AI by MongoDB

The `llama-index-embeddings-voyageai` package contains LlamaIndex integrations for building applications with Voyage AI by MongoDB's state-of-the-art embedding models. This integration provides support for text embeddings, multimodal embeddings, and contextual embeddings via the Voyage AI API.

## Installation

```shell
pip install llama-index-embeddings-voyageai
```

## Setup

### 1. Get Your API Key

Sign up for a Voyage AI by MongoDB account and obtain your API key from the [Voyage AI Dashboard](https://dash.voyageai.com/).

### 2. Set Environment Variable

Export your API key as an environment variable:

```bash
export VOYAGE_API_KEY="your-api-key-here"
```

## Usage

### Basic Usage

```python
from llama_index.embeddings.voyageai import VoyageEmbedding

# Initialize the Voyage AI Embedding model
embedding_model = VoyageEmbedding(
    model_name="voyage-3.5",
    voyage_api_key="your-api-key",  # Optional if VOYAGE_API_KEY is set
)

# Get a single embedding
embedding = embedding_model.get_text_embedding("Your text here")
print(f"Embedding dimension: {len(embedding)}")

# Get embeddings for multiple texts
texts = ["Text 1", "Text 2", "Text 3"]
embeddings = embedding_model.get_text_embedding_batch(texts)
print(f"Number of embeddings: {len(embeddings)}")
```

### Query vs Document Embeddings

Voyage AI embeddings distinguish between queries and documents for optimal retrieval performance:

```python
from llama_index.embeddings.voyageai import VoyageEmbedding

embedding_model = VoyageEmbedding(model_name="voyage-3.5")

# Get query embedding (automatically uses input_type="query")
query_embedding = embedding_model.get_query_embedding(
    "What is machine learning?"
)

# Get document embedding (automatically uses input_type="document")
doc_embedding = embedding_model.get_text_embedding("Machine learning is...")
```

### Advanced Parameters

```python
from llama_index.embeddings.voyageai import VoyageEmbedding

embedding_model = VoyageEmbedding(
    model_name="voyage-3.5",
    voyage_api_key="your-api-key",
    truncation=True,  # Enable text truncation
    output_dtype="float",  # Options: "float", "int8", "uint8", "binary", "ubinary"
    output_dimension=512,  # Reduce dimensionality (256, 512, 1024, 2048)
    embed_batch_size=128,  # Batch size for processing
)

# Use general text embedding with custom input type
embedding = embedding_model.get_general_text_embedding(
    "Your text here", input_type="query"
)
```

### Multimodal Embeddings

Voyage AI supports multimodal embeddings for text and images with `voyage-multimodal-3`, and text, images, and **video** with `voyage-multimodal-3.5`. **Important:** You must set `truncation=True` when using multimodal models.

```python
from llama_index.embeddings.voyageai import VoyageEmbedding
from io import BytesIO

# Initialize with multimodal model (truncation=True is REQUIRED)
embedding_model = VoyageEmbedding(
    model_name="voyage-multimodal-3",  # or "voyage-multimodal-3.5" for video support
    truncation=True,  # Required for multimodal models
)

# Embed an image from file path (PNG, JPEG, JPG, WEBP, GIF supported)
image_embedding = embedding_model.get_image_embedding("path/to/image.jpg")
print(f"Image embedding dimension: {len(image_embedding)}")  # 1024

# Embed an image from BytesIO
with open("path/to/image.png", "rb") as f:
    image_data = BytesIO(f.read())
    image_embedding = embedding_model.get_image_embedding(image_data)

# The multimodal model also works with text
text_embedding = embedding_model.get_text_embedding("Description of the image")
query_embedding = embedding_model.get_query_embedding(
    "Find images with red color"
)

# Batch text embeddings
batch_embeddings = embedding_model.get_text_embedding_batch(
    ["Image description 1", "Image description 2", "Image description 3"]
)
```

#### Video Embeddings (voyage-multimodal-3.5 only)

```python
from llama_index.embeddings.voyageai import VoyageEmbedding

# Initialize with voyage-multimodal-3.5 for video support
embedding_model = VoyageEmbedding(
    model_name="voyage-multimodal-3.5",
    truncation=True,
)

# Embed a single video (max 20MB, supports MP4, MPEG, MOV, AVI, FLV, MPG, WEBM, WMV, 3GP)
video_embedding = embedding_model.get_video_embedding("path/to/video.mp4")
print(f"Video embedding dimension: {len(video_embedding)}")  # 1024

# Embed multiple videos
video_embeddings = embedding_model.get_video_embeddings(
    ["video1.mp4", "video2.mp4", "video3.mp4"]
)

# Async video embedding
video_embedding = await embedding_model.aget_video_embedding(
    "path/to/video.mp4"
)
```

### Contextual Embeddings

For contextualized chunk embeddings use the `voyage-context-4` (or `voyage-context-3`) model:

```python
from llama_index.embeddings.voyageai import VoyageEmbedding

# Initialize with contextual model
embedding_model = VoyageEmbedding(
    model_name="voyage-context-4", output_dtype="float", output_dimension=1024
)

# Each text is one document. The API chunks documents server-side
# (`enable_auto_chunking=True`, `chunk_size=32000`) and each text yields one
# embedding. A text that does not fit in a single 32k-token chunk raises a
# `ValueError`.
embeddings = embedding_model.get_text_embedding_batch(
    ["First document", "Second document", "Third document"]
)

# Queries are embedded without auto-chunking
query_embedding = embedding_model.get_query_embedding("What is in the first?")
```

Contextualized models only accept `input_type="document"` (texts) or `input_type="query"` (queries); any other value raises a `ValueError`.

### Async Usage

The integration supports async operations for better performance:

```python
import asyncio
from llama_index.embeddings.voyageai import VoyageEmbedding


async def get_embeddings_async():
    # Regular text embeddings
    embedding_model = VoyageEmbedding(model_name="voyage-3.5")

    # Get async query embedding
    query_embedding = await embedding_model.aget_query_embedding("Your query")

    # Get async text embeddings
    embeddings = await embedding_model.aget_text_embedding_batch(
        ["Text 1", "Text 2", "Text 3"]
    )

    # For multimodal image embeddings
    multimodal_model = VoyageEmbedding(
        model_name="voyage-multimodal-3",
        truncation=True,  # Required for multimodal
    )
    image_embedding = await multimodal_model.aget_image_embedding(
        "path/to/image.jpg"
    )

    return query_embedding, embeddings, image_embedding


# Run async function
results = asyncio.run(get_embeddings_async())
```

### Integration with LlamaIndex

```python
from llama_index.core import VectorStoreIndex, Settings, Document
from llama_index.embeddings.voyageai import VoyageEmbedding
from llama_index.llms.openai import OpenAI

# Configure LlamaIndex settings
Settings.llm = OpenAI()
Settings.embed_model = VoyageEmbedding(
    model_name="voyage-3.5", voyage_api_key="your-api-key"
)

# Create documents
documents = [
    Document(text="LlamaIndex is a data framework for LLM applications."),
    Document(
        text="Voyage AI by MongoDB provides state-of-the-art embedding models."
    ),
    Document(text="Embeddings convert text into numerical vectors."),
]

# Create vector index
index = VectorStoreIndex.from_documents(documents)

# Query the index
query_engine = index.as_query_engine(similarity_top_k=2)
response = query_engine.query("What is LlamaIndex?")
print(response)
```

## Available Models

Voyage AI by MongoDB offers several specialized embedding models:

### Text Embeddings

- **voyage-4-large**: Best retrieval quality in the voyage-4 series, 1024 dimensions (supports 256, 512, 1024, 2048)
- **voyage-4**: General-purpose and multilingual retrieval with 1024 dimensions (supports 256, 512, 1024, 2048)
- **voyage-4-lite**: Cost and latency optimized with highest throughput, 1024 dimensions (supports 256, 512, 1024, 2048)
- **voyage-code-4**: Specialized for code retrieval and coding agents, 1024 dimensions (supports 256, 512, 1024, 2048)
- **voyage-3-large**: General-purpose and multilingual retrieval, 1024 dimensions (supports 256, 512, 1024, 2048)
- **voyage-3.5**: General-purpose model with 1024 dimensions (supports 256, 512, 1024, 2048)
- **voyage-3.5-lite**: Cost and latency optimized variant with 1024 dimensions (supports 256, 512, 1024, 2048)
- **voyage-code-3**: Specialized for code retrieval, 1024 dimensions (supports 256, 512, 1024, 2048)
- **voyage-3**: General-purpose model (1024 dimensions)
- **voyage-3-lite**: Lightweight variant (512 dimensions)

### Domain-Specific Models

- **voyage-finance-2**: Optimized for financial documents (1024 dimensions, 32K context)
- **voyage-law-2**: Specialized for legal documents (1024 dimensions, 16K context)
- **voyage-multilingual-2**: Enhanced multilingual support (1024 dimensions)

### Specialized Models

- **voyage-multimodal-3**: Supports text and image embeddings (1024 dimensions)
- **voyage-multimodal-3.5**: Supports text, image, and video embeddings (1024 dimensions, supports 256, 512, 1024, 2048)
- **voyage-context-4**: Contextualized chunk embeddings, 120K tokens per request with auto-chunking (1024 dimensions, supports 256, 512, 1024, 2048)
- **voyage-context-3**: Previous generation contextualized chunk embeddings, 120K tokens per request with auto-chunking (1024 dimensions, supports 256, 512, 1024, 2048)

### Legacy Models

- **voyage-2**: Earlier generation model (1024 dimensions)
- **voyage-large-2**: Large variant (1536 dimensions)
- **voyage-large-2-instruct**: Large instruct variant (1024 dimensions)
- **voyage-code-2**: Code embedding model (1536 dimensions)

For the latest model information, visit the [Voyage AI documentation](https://docs.voyageai.com/docs/embeddings).

## Configuration Options

| Parameter          | Type            | Default  | Description                                                   |
| ------------------ | --------------- | -------- | ------------------------------------------------------------- |
| `model_name`       | str             | Required | The embedding model to use                                    |
| `voyage_api_key`   | str             | `None`   | Voyage AI API key (falls back to VOYAGE_API_KEY env var)      |
| `embed_batch_size` | int             | `1000`   | Batch size for embedding calls (max 1000)                     |
| `truncation`       | bool            | `None`   | Enable text truncation for long inputs                        |
| `output_dtype`     | str             | `None`   | Output format: "float", "int8", "uint8", "binary", "ubinary"  |
| `output_dimension` | int             | `None`   | Reduce dimensionality (256, 512, 1024, 2048, model-dependent) |
| `callback_manager` | CallbackManager | `None`   | LlamaIndex callback manager for observability                 |

## Features

- **Dynamic Batching**: Automatically batches requests based on token limits for each model
- **Token Management**: Respects per-model token limits (ranging from 32K to 1M tokens)
- **Multimodal Support**: Process text, images, and videos with multimodal models
- **Video Embeddings**: Embed video content with `voyage-multimodal-3.5`
- **Contextual Embeddings**: Contextualized chunk embeddings with server-side auto-chunking (`voyage-context-4`, `voyage-context-3`)
- **Async Support**: Full async/await support for better performance
- **Flexible Output**: Support for various output data types and dimensions
- **Auto-truncation**: Optional text truncation for inputs exceeding model limits

## API Batch Token Limits

These limits represent the maximum total tokens that can be sent in a single API request (across all texts in the batch):

| Model                   | Batch Token Limit |
| ----------------------- | ----------------- |
| voyage-4-lite           | 1,000,000         |
| voyage-3.5-lite         | 1,000,000         |
| voyage-4                | 320,000           |
| voyage-3.5              | 320,000           |
| voyage-multimodal-3     | 320,000           |
| voyage-multimodal-3.5   | 320,000           |
| voyage-2                | 320,000           |
| voyage-4-large          | 120,000           |
| voyage-3-large          | 120,000           |
| voyage-code-3           | 120,000           |
| voyage-large-2-instruct | 120,000           |
| voyage-finance-2        | 120,000           |
| voyage-multilingual-2   | 120,000           |
| voyage-law-2            | 120,000           |
| voyage-large-2          | 120,000           |
| voyage-3                | 120,000           |
| voyage-3-lite           | 120,000           |
| voyage-code-2           | 120,000           |
| voyage-context-4        | 120,000           |
| voyage-context-3        | 120,000           |

Contextualized models take 120,000 tokens per request in document mode (auto-chunking) and 32,000 tokens in query mode. `voyage-code-4` has no documented per-request token limit, so it is batched with the conservative 120,000 default.

**Note:** The maximum batch size is 1,000 items per API request. The integration automatically handles batching based on both token limits and batch size.

## Environment Variables

| Variable         | Description                  |
| ---------------- | ---------------------------- |
| `VOYAGE_API_KEY` | Voyage AI API key (required) |

## Error Handling

The integration includes proper error handling for:

- Missing or invalid API keys
- Unsupported image formats (for multimodal models)
- Invalid model selection
- Network errors and API failures
- Token limit violations

## Additional Information

For more information about Voyage AI by MongoDB and its embedding models:

- [Voyage AI Documentation](https://docs.voyageai.com/)
- [Voyage AI Embeddings Guide](https://docs.voyageai.com/docs/embeddings)
- [Voyage AI Dashboard](https://dash.voyageai.com/)
- [API Reference](https://docs.voyageai.com/reference/embeddings-api)

## License

This project is licensed under the MIT License.
