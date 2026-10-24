import argparse
import logging

from src.app import GraphRAG


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple Graph RAG")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Index a PDF file")
    ingest.add_argument("path", help="Path to a PDF file")

    ask = sub.add_parser("ask", help="Ask a question")
    ask.add_argument("question")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    rag = GraphRAG()
    try:
        if args.command == "ingest":
            print(rag.ingest_pdf(args.path))
        elif args.command == "ask":
            print(rag.ask(args.question))
    finally:
        rag.close()


if __name__ == "__main__":
    main()