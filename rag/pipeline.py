"""Orchestration for RAG pipeline: extraction, embedding, and storage."""

import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from rag.config import config
from rag.embeddings import GermanEmbedder
from rag.extraction import extract_all_pages
from rag.storage import get_embedding_stats, load_embeddings, save_embeddings

console = Console()


def build_embeddings(force: bool = False) -> None:
    """
    Build embeddings from wiki database.

    Args:
        force: Force rebuild even if embeddings already exist
    """
    embeddings_path = config.embeddings_path_resolved

    # Check if embeddings already exist
    if embeddings_path.exists() and not force:
        console.print(
            f"[yellow]Embeddings already exist at {embeddings_path}[/yellow]"
        )
        console.print("[yellow]Use force=True to rebuild[/yellow]")
        return

    console.print("[bold blue]Step 1: Extracting pages from wiki database[/bold blue]")
    pages = extract_all_pages()
    console.print(f"[green]✓ Extracted {len(pages)} pages[/green]")

    console.print("\n[bold blue]Step 2: Loading embedding model[/bold blue]")
    embedder = GermanEmbedder()
    console.print("[green]✓ Model loaded[/green]")

    console.print("\n[bold blue]Step 3: Generating embeddings[/bold blue]")
    texts = [page.content_text for page in pages]
    embeddings = embedder.embed_documents(texts, show_progress=True)
    console.print(f"[green]✓ Generated {len(embeddings)} embeddings[/green]")

    console.print("\n[bold blue]Step 4: Saving to parquet file[/bold blue]")
    save_embeddings(pages, embeddings, str(embeddings_path))
    console.print(f"[green]✓ Saved to {embeddings_path}[/green]")

    console.print("\n[bold green]✓ Embedding build complete![/bold green]")


def show_stats() -> None:
    """Show statistics about the embeddings dataset."""
    embeddings_path = config.embeddings_path_resolved

    if not embeddings_path.exists():
        console.print(
            f"[red]Embeddings not found at {embeddings_path}[/red]",
            style="bold"
        )
        console.print("[yellow]Run 'build' command first[/yellow]")
        return

    console.print(f"[bold blue]Loading embeddings from {embeddings_path}[/bold blue]")
    df, embeddings = load_embeddings(str(embeddings_path))

    stats = get_embedding_stats(df)

    # Create a nice table
    table = Table(title="Embeddings Statistics", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Number of pages", str(stats["num_pages"]))
    table.add_row("Embedding dimension", str(embeddings.shape[1]))
    table.add_row("Avg content length", f"{stats['avg_content_length']:.0f} chars")
    table.add_row("Min content length", f"{stats['min_content_length']} chars")
    table.add_row("Max content length", f"{stats['max_content_length']} chars")
    table.add_row("Total characters", f"{stats['total_characters']:,}")

    file_size_mb = embeddings_path.stat().st_size / (1024 * 1024)
    table.add_row("File size", f"{file_size_mb:.2f} MB")

    console.print(table)

    # Show sample pages
    console.print("\n[bold blue]Sample pages:[/bold blue]")
    for i, row in enumerate(df.head(5).iter_rows(named=True)):
        console.print(f"{i+1}. {row['page_title']} ({row['page_len']} chars)")


def query_system(question: str, top_k: int = None) -> None:
    """
    Query the RAG system (for testing).

    Args:
        question: Question to ask
        top_k: Number of documents to retrieve
    """
    from rag.retrieval import WikiRetriever
    from rag.generation import RAGGenerator

    if top_k is None:
        top_k = config.default_top_k

    console.print(f"[bold blue]Question:[/bold blue] {question}\n")

    # Retrieve documents
    console.print("[bold blue]Retrieving relevant documents...[/bold blue]")
    retriever = WikiRetriever()
    results = retriever.retrieve(question, top_k=top_k)

    console.print(f"[green]✓ Found {len(results)} relevant documents[/green]\n")

    # Display retrieved documents
    console.print("[bold blue]Retrieved Documents:[/bold blue]")
    for i, result in enumerate(results, 1):
        console.print(
            f"{i}. [cyan]{result.page_title}[/cyan] "
            f"(score: {result.similarity_score:.3f})"
        )
        console.print(f"   {result.content_text[:150]}...\n")

    # Generate answer
    console.print("[bold blue]Generating answer...[/bold blue]")
    generator = RAGGenerator()
    answer = generator.generate(question, results)

    console.print(f"\n[bold green]Answer:[/bold green]\n{answer}")


def main():
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        console.print("[bold red]Usage:[/bold red]")
        console.print("  python -m rag.pipeline build         - Build embeddings")
        console.print("  python -m rag.pipeline stats         - Show statistics")
        console.print('  python -m rag.pipeline query "text"  - Query the system')
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "build":
        force = "--force" in sys.argv
        build_embeddings(force=force)

    elif command == "stats":
        show_stats()

    elif command == "query":
        if len(sys.argv) < 3:
            console.print("[red]Error: Please provide a question[/red]")
            sys.exit(1)
        question = sys.argv[2]
        query_system(question)

    else:
        console.print(f"[red]Unknown command: {command}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
