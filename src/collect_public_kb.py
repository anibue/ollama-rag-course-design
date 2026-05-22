import argparse
import csv
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.parse import quote
from xml.etree import ElementTree

import requests

from config import BASE_DIR


OUTPUT_DIR = BASE_DIR / "data" / "public_kb"
CSE_REPO = "hatakekksheeshh/CSE_course_RAG"
ARXIV_HF_REPO = "CCRss/arXiv_dataset"
ARXIV_API_URL = "https://export.arxiv.org/api/query"
HF_DATASET_SERVER = "https://datasets-server.huggingface.co"
HF_DATASET_URL = "https://huggingface.co/datasets"
DEFAULT_CS_CATEGORIES = ["cs.AI", "cs.CL", "cs.DB", "cs.NI", "cs.OS", "cs.LG"]
ARXIV_SEED_RECORDS = [
    {
        "id": "2005.11401",
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "authors": "Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, and others",
        "categories": "cs.CL",
        "abstract": "This public arXiv record describes retrieval-augmented generation, where a parametric generator is combined with retrieved non-parametric memory to improve knowledge-intensive NLP tasks.",
    },
    {
        "id": "2004.04906",
        "title": "Dense Passage Retrieval for Open-Domain Question Answering",
        "authors": "Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, and others",
        "categories": "cs.CL",
        "abstract": "This public arXiv record presents dense passage retrieval for open-domain question answering and is relevant to vector retrieval design in RAG systems.",
    },
    {
        "id": "1810.04805",
        "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "authors": "Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova",
        "categories": "cs.CL",
        "abstract": "This public arXiv record presents BERT, a Transformer encoder model that became a foundation for many language understanding, retrieval, and question-answering systems.",
    },
    {
        "id": "1908.10084",
        "title": "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
        "authors": "Nils Reimers, Iryna Gurevych",
        "categories": "cs.CL",
        "abstract": "This public arXiv record describes sentence embedding methods for semantic similarity, a core idea behind embedding-based document retrieval.",
    },
]


def slugify(value: str, max_length: int = 64) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-").lower()
    return (value or "document")[:max_length]


def request_json(url: str, timeout: int, params: Optional[Dict[str, object]] = None) -> object:
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def request_text(url: str, timeout: int) -> str:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.text


def write_markdown(path: Path, title: str, metadata: Dict[str, object], body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    for key, value in metadata.items():
        if value not in (None, "", []):
            lines.append(f"- {key}: {value}")
    lines.extend(["", body.strip(), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def add_manifest_record(
    manifest: List[Dict[str, object]],
    *,
    title: str,
    source: str,
    license_name: str,
    url: str,
    output_path: Path,
    source_type: str,
    extra: Optional[Dict[str, object]] = None,
) -> None:
    record = {
        "title": title,
        "source": source,
        "source_type": source_type,
        "license": license_name,
        "url": url,
        "output_path": str(output_path.relative_to(BASE_DIR)),
        "collected_at": datetime.now().isoformat(timespec="seconds"),
    }
    if extra:
        record.update(extra)
    manifest.append(record)


def extract_text_items(value: object) -> Iterable[Dict[str, str]]:
    text_keys = ("raw_text", "markdown", "text", "content", "page_content", "body")
    title_keys = ("title", "course", "course_id", "source_file", "doc_id", "name")

    if isinstance(value, dict):
        text = ""
        for key in text_keys:
            if isinstance(value.get(key), str) and len(value[key].strip()) >= 80:
                text = value[key].strip()
                break
        if text:
            title_parts = [
                str(value[key]).strip()
                for key in title_keys
                if isinstance(value.get(key), (str, int)) and str(value[key]).strip()
            ]
            yield {"title": " - ".join(title_parts[:3]) or "CSE course material", "text": text}

        for nested in value.values():
            yield from extract_text_items(nested)

    elif isinstance(value, list):
        for item in value:
            yield from extract_text_items(item)


def parse_json_records(raw_text: str) -> List[Dict[str, str]]:
    try:
        return list(extract_text_items(json.loads(raw_text)))
    except json.JSONDecodeError:
        records: List[Dict[str, str]] = []
        for line in raw_text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.extend(extract_text_items(json.loads(line)))
            except json.JSONDecodeError:
                continue
        return records


def parse_csv_records(raw_text: str) -> List[Dict[str, str]]:
    rows = csv.DictReader(raw_text.splitlines())
    records: List[Dict[str, str]] = []
    text_columns = ("raw_text", "markdown", "text", "content", "abstract", "description")
    title_columns = ("title", "course", "course_id", "source_file", "doc_id", "name")
    for row in rows:
        text = next((row.get(col, "").strip() for col in text_columns if row.get(col, "").strip()), "")
        if len(text) < 80:
            continue
        title = " - ".join(row.get(col, "").strip() for col in title_columns if row.get(col, "").strip())
        records.append({"title": title or "CSE course material", "text": text})
    return records


def split_markdown_records(title: str, raw_text: str, max_items: int) -> List[Dict[str, str]]:
    sections = [section.strip() for section in re.split(r"\n(?=##+ )", raw_text) if len(section.strip()) >= 120]
    if not sections and len(raw_text.strip()) >= 120:
        sections = [raw_text.strip()]

    records: List[Dict[str, str]] = []
    for index, section in enumerate(sections[:max_items], start=1):
        heading_match = re.search(r"^#+\s+(.+)$", section, flags=re.MULTILINE)
        section_title = heading_match.group(1).strip() if heading_match else title
        records.append({"title": section_title or f"{title} section {index}", "text": section})
    return records


def fetch_cse_records(max_items: int, timeout: int) -> List[Dict[str, str]]:
    card_url = f"{HF_DATASET_URL}/{CSE_REPO}/raw/main/README.md"
    card_text = request_text(card_url, timeout=timeout)
    card_records = split_markdown_records("CSE Course RAG dataset card", card_text, max_items)
    for record in card_records:
        record["url"] = card_url
        record["path"] = "README.md"
    if len(card_records) >= max_items:
        return card_records[:max_items]

    tree_url = f"https://huggingface.co/api/datasets/{CSE_REPO}/tree/main"
    tree = request_json(tree_url, timeout=timeout)
    if not isinstance(tree, list):
        return []

    candidates = []
    for item in tree:
        path = str(item.get("path", ""))
        size = int(item.get("size") or 0)
        suffix = Path(path).suffix.lower()
        if item.get("type") != "file" or suffix not in {".json", ".jsonl", ".md", ".txt", ".csv"}:
            continue
        if size and size > 2_000_000:
            continue
        if any(part in path.lower() for part in ("/indices/", "/converted/", "/scratch/")):
            continue
        candidates.append(path)

    records: List[Dict[str, str]] = []
    for path in candidates:
        if len(records) >= max_items:
            break
        raw_url = f"{HF_DATASET_URL}/{CSE_REPO}/resolve/main/{quote(path, safe='/')}"
        try:
            raw_text = request_text(raw_url, timeout=timeout)
        except requests.RequestException:
            continue

        suffix = Path(path).suffix.lower()
        if suffix in {".json", ".jsonl"}:
            parsed = parse_json_records(raw_text)
        elif suffix == ".csv":
            parsed = parse_csv_records(raw_text)
        else:
            parsed = split_markdown_records(Path(path).stem, raw_text, max_items - len(records))

        for record in parsed:
            record["url"] = raw_url
            record["path"] = path
            records.append(record)
            if len(records) >= max_items:
                break

    if records:
        return records[:max_items]

    return (card_records + records)[:max_items]


def write_cse_documents(output_dir: Path, max_items: int, timeout: int, manifest: List[Dict[str, object]]) -> int:
    records = fetch_cse_records(max_items=max_items, timeout=timeout)
    written = 0
    for index, record in enumerate(records, start=1):
        title = record["title"][:120]
        path = output_dir / "cse_course_rag" / f"cse_{index:03d}_{slugify(title)}.md"
        metadata = {
            "Source": CSE_REPO,
            "License": "MIT",
            "URL": record.get("url", f"{HF_DATASET_URL}/{CSE_REPO}"),
            "Original path": record.get("path"),
        }
        write_markdown(path, title, metadata, record["text"])
        add_manifest_record(
            manifest,
            title=title,
            source=CSE_REPO,
            source_type="huggingface_dataset",
            license_name="MIT",
            url=str(record.get("url", f"{HF_DATASET_URL}/{CSE_REPO}")),
            output_path=path,
            extra={"original_path": record.get("path")},
        )
        written += 1
    return written


def get_hf_arxiv_rows(limit: int, timeout: int, categories: List[str]) -> List[Dict[str, object]]:
    split_info = request_json(f"{HF_DATASET_SERVER}/splits", timeout=timeout, params={"dataset": ARXIV_HF_REPO})
    splits = split_info.get("splits", []) if isinstance(split_info, dict) else []
    if not splits:
        return []

    config = splits[0]["config"]
    split = splits[0]["split"]
    selected: List[Dict[str, object]] = []
    page_length = min(max(limit * 20, 100), 500)
    for offset in range(0, 5000, page_length):
        rows_payload = request_json(
            f"{HF_DATASET_SERVER}/rows",
            timeout=timeout,
            params={
                "dataset": ARXIV_HF_REPO,
                "config": config,
                "split": split,
                "offset": offset,
                "length": page_length,
            },
        )
        rows = rows_payload.get("rows", []) if isinstance(rows_payload, dict) else []
        if not rows:
            break
        for item in rows:
            row = item.get("row", {}) if isinstance(item, dict) else {}
            row_categories = str(row.get("categories", ""))
            if any(category in row_categories.split() for category in categories):
                row["metadata_source"] = ARXIV_HF_REPO
                selected.append(row)
            if len(selected) >= limit:
                break
        if len(selected) >= limit:
            break
    return selected


def get_arxiv_api_rows(limit: int, timeout: int, categories: List[str]) -> List[Dict[str, object]]:
    query = " OR ".join(f"cat:{category}" for category in categories)
    response = requests.get(
        ARXIV_API_URL,
        params={
            "search_query": query,
            "start": 0,
            "max_results": limit,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        },
        timeout=timeout,
    )
    response.raise_for_status()

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ElementTree.fromstring(response.text)
    rows: List[Dict[str, object]] = []
    for entry in root.findall("atom:entry", ns):
        arxiv_id = entry.findtext("atom:id", default="", namespaces=ns).rsplit("/", 1)[-1]
        title = re.sub(r"\s+", " ", entry.findtext("atom:title", default="", namespaces=ns)).strip()
        abstract = re.sub(r"\s+", " ", entry.findtext("atom:summary", default="", namespaces=ns)).strip()
        authors = ", ".join(
            author.findtext("atom:name", default="", namespaces=ns)
            for author in entry.findall("atom:author", ns)
        )
        pdf_url = ""
        for link in entry.findall("atom:link", ns):
            if link.attrib.get("title") == "pdf":
                pdf_url = link.attrib.get("href", "")
                break
        rows.append(
            {
                "id": arxiv_id,
                "title": title,
                "authors": authors,
                "categories": " ".join(categories),
                "abstract": abstract,
                "license": "See arXiv record",
                "url": f"https://arxiv.org/abs/{arxiv_id}",
                "pdf_url": pdf_url or f"https://arxiv.org/pdf/{arxiv_id}",
                "metadata_source": "arXiv API",
            }
        )
    return rows[:limit]


def fetch_arxiv_records(limit: int, timeout: int, categories: List[str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    try:
        rows = get_hf_arxiv_rows(limit=limit, timeout=timeout, categories=categories)
    except requests.RequestException:
        pass

    for row in rows:
        arxiv_id = str(row.get("id", ""))
        row["url"] = f"https://arxiv.org/abs/{arxiv_id}"
        row["pdf_url"] = f"https://arxiv.org/pdf/{arxiv_id}"

    if len(rows) < limit:
        existing_ids = {str(row.get("id", "")) for row in rows}
        try:
            fallback_rows = get_arxiv_api_rows(limit=limit * 2, timeout=timeout, categories=categories)
        except requests.RequestException:
            fallback_rows = []

        for row in fallback_rows + ARXIV_SEED_RECORDS:
            arxiv_id = str(row.get("id", ""))
            if arxiv_id in existing_ids:
                continue
            if "metadata_source" not in row:
                row["metadata_source"] = "arXiv public seed"
            rows.append(row)
            existing_ids.add(arxiv_id)
            if len(rows) >= limit:
                break

    return rows[:limit]


def download_pdf(pdf_url: str, output_path: Path, timeout: int) -> bool:
    response = requests.get(pdf_url, timeout=timeout)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    if "pdf" not in content_type.lower() and not response.content.startswith(b"%PDF"):
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(response.content)
    return True


def write_arxiv_documents(
    output_dir: Path,
    max_items: int,
    timeout: int,
    categories: List[str],
    download_pdfs: bool,
    manifest: List[Dict[str, object]],
) -> int:
    rows = fetch_arxiv_records(limit=max_items, timeout=timeout, categories=categories)
    written = 0
    for index, row in enumerate(rows, start=1):
        arxiv_id = str(row.get("id", f"arxiv-{index}"))
        title = re.sub(r"\s+", " ", str(row.get("title", "Computer science paper"))).strip()[:160]
        authors = re.sub(r"\s+", " ", str(row.get("authors", ""))).strip()
        categories_text = str(row.get("categories", ""))
        abstract = re.sub(r"\s+", " ", str(row.get("abstract", ""))).strip()
        license_name = str(row.get("license") or "CC0 metadata; paper license follows arXiv record")
        url = str(row.get("url") or f"https://arxiv.org/abs/{arxiv_id}")
        pdf_url = str(row.get("pdf_url") or f"https://arxiv.org/pdf/{arxiv_id}")

        body = "\n\n".join(
            [
                f"Title: {title}",
                f"Authors: {authors}",
                f"Categories: {categories_text}",
                f"Abstract: {abstract}",
                f"Source URL: {url}",
            ]
        )
        path = output_dir / "arxiv" / f"arxiv_{index:03d}_{slugify(arxiv_id)}.md"
        metadata = {
            "Source": row.get("metadata_source", ARXIV_HF_REPO),
            "License": license_name,
            "URL": url,
            "PDF": pdf_url,
        }
        write_markdown(path, title or arxiv_id, metadata, body)

        extra = {"arxiv_id": arxiv_id, "categories": categories_text, "pdf_url": pdf_url}
        if download_pdfs:
            pdf_path = output_dir / "papers" / f"{slugify(arxiv_id)}.pdf"
            try:
                if download_pdf(pdf_url, pdf_path, timeout=timeout):
                    extra["pdf_output_path"] = str(pdf_path.relative_to(BASE_DIR))
            except requests.RequestException:
                extra["pdf_download_error"] = "download failed"

        add_manifest_record(
            manifest,
            title=title or arxiv_id,
            source=str(row.get("metadata_source", ARXIV_HF_REPO)),
            source_type="huggingface_arxiv_metadata" if row.get("metadata_source") == ARXIV_HF_REPO else "arxiv_api",
            license_name=license_name,
            url=url,
            output_path=path,
            extra=extra,
        )
        written += 1
        time.sleep(0.2)
    return written


def write_manifest(output_dir: Path, manifest: List[Dict[str, object]], args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_dir.relative_to(BASE_DIR)),
        "sources": [
            {
                "name": CSE_REPO,
                "type": "Hugging Face dataset",
                "url": f"{HF_DATASET_URL}/{CSE_REPO}",
                "license": "MIT",
            },
            {
                "name": ARXIV_HF_REPO,
                "type": "Hugging Face dataset",
                "url": f"{HF_DATASET_URL}/{ARXIV_HF_REPO}",
                "license": "CC0-1.0 metadata",
            },
            {
                "name": "arXiv Computer Science",
                "type": "paper metadata and optional PDF source",
                "url": "https://arxiv.org/archive/cs",
                "license": "Per-paper arXiv license",
            },
        ],
        "parameters": {
            "max_cse": args.max_cse,
            "max_arxiv": args.max_arxiv,
            "download_pdf": args.download_pdf,
            "categories": args.categories,
        },
        "documents": manifest,
    }
    (output_dir / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def clear_generated_markdown(output_dir: Path) -> None:
    for subdir in ("cse_course_rag", "arxiv"):
        target_dir = output_dir / subdir
        if not target_dir.exists():
            continue
        for path in target_dir.glob("*.md"):
            path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect public CS documents for the local RAG knowledge base.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directory for generated Markdown/PDF files.")
    parser.add_argument("--max-cse", type=int, default=50, help="Maximum CSE course material text items.")
    parser.add_argument("--max-arxiv", type=int, default=20, help="Maximum arXiv CS paper abstracts.")
    parser.add_argument("--categories", nargs="+", default=DEFAULT_CS_CATEGORIES, help="arXiv CS categories to keep.")
    parser.add_argument("--download-pdf", action="store_true", help="Download paper PDFs into data/public_kb/papers.")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout in seconds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else BASE_DIR / args.output_dir
    manifest: List[Dict[str, object]] = []

    clear_generated_markdown(output_dir)
    cse_count = write_cse_documents(output_dir, args.max_cse, args.timeout, manifest)
    arxiv_count = write_arxiv_documents(
        output_dir=output_dir,
        max_items=args.max_arxiv,
        timeout=args.timeout,
        categories=args.categories,
        download_pdfs=args.download_pdf,
        manifest=manifest,
    )
    write_manifest(output_dir, manifest, args)

    print(f"CSE documents written: {cse_count}")
    print(f"arXiv documents written: {arxiv_count}")
    print(f"Manifest: {output_dir / 'manifest.json'}")
    print(f"Total public documents: {len(manifest)}")


if __name__ == "__main__":
    main()
