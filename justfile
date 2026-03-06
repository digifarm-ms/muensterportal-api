# Justfile for Münster4You RAG system

# Show available commands
default:
    @just --list

# Build embeddings from wiki database
build:
    @echo "Building embeddings from wiki.sqlite..."
    uv run python -m muenster4you.rag.pipeline build

# Force rebuild embeddings even if they exist
rebuild:
    @echo "Force rebuilding embeddings..."
    uv run python -m muenster4you.rag.pipeline build --force

# Show embedding statistics
stats:
    @echo "Loading embedding statistics..."
    uv run python -m muenster4you.rag.pipeline stats

# Query the RAG system from command line
query QUESTION:
    @echo "Querying RAG system..."
    uv run python -m muenster4you.rag.pipeline query "{{QUESTION}}"

# Run Streamlit app
app:
    @echo "Starting Streamlit app..."
    uv run streamlit run src/muenster4you/app.py

# Clean generated data
clean:
    @echo "Cleaning generated data..."
    rm -rf data/wiki_embeddings.parquet
    @echo "Cleaned!"

# Test extraction
test-extract:
    @echo "Testing extraction..."
    uv run python -m muenster4you.rag.extraction

# Test embeddings
test-embed:
    @echo "Testing embeddings..."
    uv run python -m muenster4you.rag.embeddings

# Test retrieval
test-retrieval:
    @echo "Testing retrieval..."
    uv run python -m muenster4you.rag.retrieval

# Test generation
test-generation:
    @echo "Testing generation..."
    uv run python -m muenster4you.rag.generation

# Run all tests
test-all: test-extract test-embed

# Install dependencies
install:
    @echo "Installing dependencies..."
    uv sync

# Show project info
info:
    @echo "Münster4You RAG System"
    @echo "====================="
    @echo ""
    @echo "Configuration:"
    @echo "  Wiki DB: wiki.sqlite"
    @echo "  Embeddings: data/wiki_embeddings.parquet"
    @echo "  Embedding Model: mixedbread-ai/deepset-mxbai-embed-de-large-v1"
    @echo "  Generation Model: qwen3:30b"
    @echo ""
    @echo "Available commands:"
    @just --list
