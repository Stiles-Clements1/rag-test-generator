import argparse
import json
import os
import re
from getpass import getpass
from pathlib import Path
from typing import Any, Dict, List, Optional
from sentence_transformers import CrossEncoder

import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".pdf", ".r", ".xlsx", ".xls", ".csv"}


def resolve_openai_api_key(
    explicit_api_key: Optional[str] = None, prompt_if_missing: bool = False
) -> str:
    key = (explicit_api_key or os.getenv("OPENAI_API_KEY", "")).strip()
    if not key and prompt_if_missing:
        key = getpass("Enter your OpenAI API key: ").strip()
    if not key:
        raise ValueError(
            "OpenAI API key not found. Set OPENAI_API_KEY or provide one at runtime."
        )
    return key

"""Helpers:
    - infer document type using filename and extension
    - infer lecture day, so user can ask questions about specific lectures
    - infer topic
"""

def infer_document_type(file_name: str) -> str:
    name = file_name.lower()
    if name.endswith(".r"):
        return "code"
    if name.endswith(".csv") or name.endswith(".xlsx") or name.endswith(".xls"):
        return "tabular_data"
    if "slides" in name:
        return "slides"
    return "notes"

def infer_lecture_day(file_name: str) -> str:
    m = re.search(r"day_(\d+)", file_name.lower())
    if m:
        return f"day_{m.group(1).zfill(2)}"
    return "general"


def infer_topic(file_name: str) -> str:
    base = file_name.lower()
    for ext in [".pdf", ".r", ".xlsx", ".xls", ".csv"]:
        base = base.replace(ext, "")
    base = re.sub(r"day_\d+_", "", base)
    base = base.replace("_slides", "").replace("_theory", "")
    return base


def parse_query_filters(query: str) -> Dict[str, str]:
    query_lower = query.lower()
    metadata_filter: Dict[str, str] = {}

    if "r code" in query_lower or "coding test" in query_lower or "code" in query_lower or "script" in query_lower:
        metadata_filter["document_type"] = "code"
    elif "excel" in query_lower or "csv" in query_lower or "spreadsheet" in query_lower or "data file" in query_lower:
        metadata_filter["document_type"] = "tabular_data"
    elif "slides" in query_lower:
        metadata_filter["document_type"] = "slides"
    elif "notes" in query_lower:
        metadata_filter["document_type"] = "notes"

    day_match = re.search(r"(day|lecture)\s*(\d+)", query_lower)
    if day_match:
        day_num = int(day_match.group(2))
        metadata_filter["lecture_day"] = f"day_{day_num:02d}"

    return metadata_filter


def _normalize_text(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text).strip()


""" Store source, document type, lecture day, topic, file extension, sheet name,
    chunk id and the number of total chunks.
"""
def _chunk_text(
    text: str,
    splitter: RecursiveCharacterTextSplitter,
    source: str,
    document_type: str,
    file_extension: str,
    lecture_day: str,
    topic: str,
    sheet_name: Optional[str] = None,
) -> List[Document]:
    chunks = splitter.split_text(_normalize_text(text))
    docs: List[Document] = []
    for idx, chunk in enumerate(chunks):
        docs.append(
            Document(
                page_content=chunk,
                metadata={
                    "source": source,
                    "document_type": document_type,
                    "lecture_day": lecture_day,
                    "topic": topic,
                    "file_extension": file_extension,
                    "sheet_name": sheet_name,
                    "chunk_id": idx,
                    "total_chunks": len(chunks),
                },
            )
        )
    return docs

def process_course_materials(
    directory_path: str, chunk_size: int = 1000, chunk_overlap: int = 150
) -> List[Document]:
    materials_dir = Path(directory_path)
    if not materials_dir.exists() or not materials_dir.is_dir():
        raise FileNotFoundError(f"Materials directory not found: {materials_dir}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    files = sorted(
        [
            path
            for path in materials_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
    )
    if not files:
        raise ValueError(
            f"No supported files found in {materials_dir}. Supported: {SUPPORTED_EXTENSIONS}"
        )

    all_docs: List[Document] = []
    for file_path in files:
        ext = file_path.suffix.lower()
        document_type = infer_document_type(file_path.name)
        lecture_day = infer_lecture_day(file_path.name)
        topic = infer_topic(file_path.name)

        try:
            if ext == ".pdf":
                reader = PdfReader(str(file_path))
                text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
                if text.strip():
                    all_docs.extend(
                        _chunk_text(
                            text=text,
                            splitter=splitter,
                            source=file_path.name,
                            document_type=document_type,
                            file_extension=ext,
                            lecture_day=lecture_day,
                            topic=topic,
                        )
                    )

            elif ext == ".r":
                with file_path.open("r", encoding="utf-8", errors="replace") as handle:
                    code = handle.read()
                wrapped_code = (
                    f"Document type: R code\n"
                    f"Source file: {file_path.name}\n"
                    f"This file contains R code examples and implementation details.\n\n"
                    f"{code}"
                )
                all_docs.extend(
                    _chunk_text(
                        text=wrapped_code,
                        splitter=splitter,
                        source=file_path.name,
                        document_type=document_type,
                        file_extension=ext,
                        lecture_day=lecture_day,
                        topic=topic,
                    )
                )

            elif ext in {".xlsx", ".xls"}:
                sheets = pd.read_excel(file_path, sheet_name=None)
                for sheet_name, df in sheets.items():
                    sheet_text = (
                        f"Document type: spreadsheet\n"
                        f"Source file: {file_path.name}\n"
                        f"Sheet name: {sheet_name}\n\n"
                        f"{df.to_string(index=False)}"
                    )
                    all_docs.extend(
                        _chunk_text(
                            text=sheet_text,
                            splitter=splitter,
                            source=file_path.name,
                            document_type=document_type,
                            file_extension=ext,
                            lecture_day=lecture_day,
                            topic=topic,
                            sheet_name=str(sheet_name),
                        )
                    )

            elif ext == ".csv":
                df = pd.read_csv(file_path)
                csv_text = (
                    f"Document type: csv table\n"
                    f"Source file: {file_path.name}\n\n"
                    f"{df.to_string(index=False)}"
                )
                all_docs.extend(
                    _chunk_text(
                        text=csv_text,
                        splitter=splitter,
                        source=file_path.name,
                        document_type=document_type,
                        file_extension=ext,
                        lecture_day=lecture_day,
                        topic=topic,
                    )
                )

        except Exception as exc:
            print(f"Skipping {file_path.name} due to read error: {exc}")

    if not all_docs:
        raise ValueError("No readable content was extracted from materials.")
    return all_docs


def _extract_json(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise ValueError("Model response did not include valid JSON.")
        return json.loads(match.group(0))


def _normalize_options(raw_options: Any) -> List[Dict[str, str]]:
    options: List[Dict[str, str]] = []
    if not isinstance(raw_options, list):
        return options

    for idx, option in enumerate(raw_options):
        default_label = chr(65 + idx)
        if isinstance(option, dict):
            label = str(option.get("label", default_label)).strip() or default_label
            text = str(option.get("text", "")).strip()
        else:
            label = default_label
            text = str(option).strip()
        if text:
            options.append({"label": label[:1].upper(), "text": text})
    return options


def _normalize_test_payload(
    payload: Dict[str, Any], num_questions: int, sources: List[str]
) -> Dict[str, Any]:
    raw_questions = payload.get("questions", [])
    if not isinstance(raw_questions, list) or not raw_questions:
        raise ValueError("Generated test did not include a valid question list.")

    normalized_questions: List[Dict[str, Any]] = []
    for idx, question in enumerate(raw_questions[:num_questions], start=1):
        if not isinstance(question, dict):
            continue

        q_type = str(question.get("type", "short_answer")).strip().lower()
        if q_type not in {"multiple_choice", "short_answer"}:
            q_type = "short_answer"

        options = _normalize_options(question.get("options", []))
        if q_type == "multiple_choice" and len(options) < 2:
            q_type = "short_answer"

        answer = str(question.get("answer", "")).strip()
        if not answer:
            answer = "No answer provided."

        explanation = str(question.get("explanation", "")).strip()
        keywords = question.get("keywords", [])
        if not isinstance(keywords, list):
            keywords = []
        keywords = [str(item).strip().lower() for item in keywords if str(item).strip()]

        points = question.get("points", 1)
        try:
            points = max(1.0, float(points))
        except (TypeError, ValueError):
            points = 1.0

        raw_id = str(question.get("id", idx))
        sanitized_id = re.sub(r"[^a-zA-Z0-9_-]", "_", raw_id).strip("_") or str(idx)

        normalized_questions.append(
            {
                "id": sanitized_id,
                "type": q_type,
                "prompt": str(question.get("prompt", f"Question {idx}")).strip(),
                "options": options,
                "answer": answer,
                "keywords": keywords,
                "explanation": explanation,
                "points": points,
            }
        )

    if not normalized_questions:
        raise ValueError("Generated questions could not be normalized.")

    return {
        "title": str(payload.get("title", "Practice Test")).strip() or "Practice Test",
        "instructions": str(
            payload.get(
                "instructions",
                "Answer each question using only what you learned from class materials.",
            )
        ).strip(),
        "questions": normalized_questions,
        "sources": sorted(set(sources)),
    }


def strip_answers(test_payload: Dict[str, Any]) -> Dict[str, Any]:
    questions = []
    for question in test_payload.get("questions", []):
        question_copy = dict(question)
        question_copy.pop("answer", None)
        question_copy.pop("keywords", None)
        questions.append(question_copy)
    return {
        "title": test_payload.get("title", "Practice Test"),
        "instructions": test_payload.get("instructions", ""),
        "questions": questions,
        "sources": test_payload.get("sources", []),
    }


def _normalize_text_for_compare(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _grade_short_answer(question: Dict[str, Any], user_response: str) -> float:
    points = float(question.get("points", 1.0))
    if not user_response.strip():
        return 0.0

    user_normalized = _normalize_text_for_compare(user_response)
    answer_normalized = _normalize_text_for_compare(str(question.get("answer", "")))
    expected_tokens = set(answer_normalized.split())
    user_tokens = set(user_normalized.split())

    overlap_ratio = 0.0
    if expected_tokens:
        overlap_ratio = len(expected_tokens & user_tokens) / len(expected_tokens)

    keywords = [token for token in question.get("keywords", []) if token]
    keyword_hits = 0
    for keyword in keywords:
        if _normalize_text_for_compare(keyword) in user_normalized:
            keyword_hits += 1
    keyword_ratio = (keyword_hits / len(keywords)) if keywords else 0.0

    confidence = max(overlap_ratio, keyword_ratio)
    if confidence >= 0.75:
        return points
    if confidence >= 0.45:
        return round(points * 0.6, 2)
    if confidence >= 0.2:
        return round(points * 0.3, 2)
    return 0.0


def _grade_short_answers_with_llm(
    short_questions: List[Dict[str, Any]],
    responses: Dict[str, str],
    llm: Any,
) -> Dict[str, Dict[str, Any]]:
    if not short_questions:
        return {}

    question_payload = []
    max_points_by_id: Dict[str, float] = {}
    for question in short_questions:
        question_id = str(question.get("id", "")).strip()
        if not question_id:
            continue

        max_points = float(question.get("points", 1.0))
        max_points_by_id[question_id] = max_points
        question_payload.append(
            {
                "id": question_id,
                "prompt": str(question.get("prompt", "")).strip(),
                "reference_answer": str(question.get("answer", "")).strip(),
                "reference_explanation": str(question.get("explanation", "")).strip(),
                "max_points": max_points,
                "student_answer": str(responses.get(question_id, "")).strip(),
            }
        )

    if not question_payload:
        return {}

    rubric_prompt = f"""
You are grading short-answer questions for a university course.
Grade for semantic correctness and conceptual meaning, not keyword matching.

Scoring rubric:
- Full credit: answer is conceptually correct and complete.
- Partial credit: answer is partly correct but missing key details.
- Zero: answer is incorrect, irrelevant, or empty.

Return only valid JSON:
{{
  "results": [
    {{
      "id": "question id",
      "earned_points": 0.0,
      "feedback": "1-2 sentences on why this score was given"
    }}
  ]
}}

Rules:
- Grade each question independently.
- Keep earned_points between 0 and max_points.
- Use at most 2 decimal places.
- Do not include markdown or extra text.

Questions to grade:
{json.dumps(question_payload, ensure_ascii=True)}
"""
    try:
        model_output = llm.invoke(rubric_prompt)
        content = model_output.content if hasattr(model_output, "content") else str(model_output)
        parsed = _extract_json(content)
    except Exception:
        return {}

    rows = parsed.get("results", []) if isinstance(parsed, dict) else []
    if not isinstance(rows, list):
        return {}

    graded: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        question_id = str(row.get("id", "")).strip()
        if question_id not in max_points_by_id:
            continue

        max_points = max_points_by_id[question_id]
        try:
            raw_score = float(row.get("earned_points", 0.0))
        except (TypeError, ValueError):
            raw_score = 0.0

        score = round(max(0.0, min(max_points, raw_score)), 2)
        feedback = str(row.get("feedback", "")).strip() or "No feedback provided."
        graded[question_id] = {"earned_points": score, "feedback": feedback}

    return graded


def grade_test(
    test_payload: Dict[str, Any], responses: Dict[str, str], llm: Optional[Any] = None
) -> Dict[str, Any]:
    questions = test_payload.get("questions", [])
    results = []
    total_points = 0.0
    earned_points = 0.0
    short_questions = [q for q in questions if q.get("type", "short_answer") == "short_answer"]
    llm_grades = (
        _grade_short_answers_with_llm(short_questions, responses, llm)
        if llm is not None
        else {}
    )

    for question in questions:
        question_id = str(question.get("id"))
        points = float(question.get("points", 1.0))
        total_points += points

        user_response = str(responses.get(question_id, "")).strip()
        q_type = question.get("type", "short_answer")
        answer = str(question.get("answer", "")).strip()

        if q_type == "multiple_choice":
            normalized_user = _normalize_text_for_compare(user_response)
            normalized_answer = _normalize_text_for_compare(answer)

            answer_label = normalized_answer[:1]
            correct = normalized_user == normalized_answer or normalized_user == answer_label

            if not correct:
                for option in question.get("options", []):
                    if _normalize_text_for_compare(option.get("text", "")) == normalized_user:
                        if _normalize_text_for_compare(option.get("label", "")) == answer_label:
                            correct = True
                            break

            points_earned = points if correct else 0.0
            feedback = "Correct." if correct else "Incorrect."

        else:
            llm_grade = llm_grades.get(question_id)
            if llm_grade is not None:
                points_earned = float(llm_grade.get("earned_points", 0.0))
                feedback = str(llm_grade.get("feedback", ""))
            else:
                points_earned = _grade_short_answer(question, user_response)
                if points_earned == 0:
                    feedback = "Insufficient detail compared to expected answer."
                elif points_earned >= points * 0.99:
                    feedback = "Strong answer."
                else:
                    feedback = "Partially correct."
            correct = points_earned >= points * 0.99

        earned_points += points_earned
        results.append(
            {
                "id": question_id,
                "correct": correct,
                "earned_points": round(points_earned, 2),
                "max_points": round(points, 2),
                "feedback": feedback,
                "correct_answer": answer,
                "explanation": question.get("explanation", ""),
            }
        )

    percent = round((earned_points / total_points) * 100, 2) if total_points else 0.0
    return {
        "score": round(earned_points, 2),
        "total_points": round(total_points, 2),
        "percentage": percent,
        "results": results,
    }


class TestMakerRAG:
    def __init__(
        self,
        materials_dir: str = "Materials",
        persist_directory: str = "chroma_db",
        model_name: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        retrieval_k: int = 8,
        api_key: Optional[str] = None,
        rebuild_index: bool = False,
    ) -> None:
        self.materials_dir = str(materials_dir)
        self.persist_directory = str(persist_directory)
        self.retrieval_k = retrieval_k
        self.api_key = resolve_openai_api_key(api_key, prompt_if_missing=False)

        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            api_key=self.api_key,
        )
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.2,
            api_key=self.api_key,
        )

        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        self.vectorstore = self._build_or_load_vectorstore(rebuild_index=rebuild_index)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": retrieval_k})

    def _build_or_load_vectorstore(self, rebuild_index: bool) -> Chroma:
        persist_path = Path(self.persist_directory)
        persist_path.mkdir(parents=True, exist_ok=True)
        has_index = any(persist_path.iterdir())

        if rebuild_index or not has_index:
            docs = process_course_materials(self.materials_dir)
            vectorstore = Chroma.from_documents(
                documents=docs,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
            )
            if hasattr(vectorstore, "persist"):
                vectorstore.persist()
            return vectorstore

        return Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
        )


    def _format_context(self, docs: List[Document]) -> str:
        formatted_docs = []
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            doc_type = doc.metadata.get("document_type", "unknown")
            lecture_day = doc.metadata.get("lecture_day", "unknown")
            sheet_name = doc.metadata.get("sheet_name")

            header = f"[Source: {source} | Type: {doc_type} | Lecture: {lecture_day}"
            if sheet_name and str(sheet_name).lower() != "none":
                header += f" | Sheet: {sheet_name}"
            header += "]"

            formatted_docs.append(f"{header}\n{doc.page_content}")
        return "\n\n".join(formatted_docs)

    

    def _rerank_documents(self, query: str, docs: List[Document], top_k: int = 4) -> List[Document]:
        if not docs:
            return docs

        pairs = [[query, doc.page_content] for doc in docs]
        scores = self.reranker.predict(pairs)

        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[:top_k]]


    def retrieve(self, query: str, k: Optional[int] = None) -> List[Document]:
        if k is None:
            k = self.retrieval_k

        metadata_filter = parse_query_filters(query)

        if metadata_filter:
            docs = self.vectorstore.similarity_search(query, k=max(k, 10), filter=metadata_filter)
        else:
            docs = self.vectorstore.similarity_search(query, k=max(k, 10))

        docs = self._rerank_documents(query, docs, top_k=k)
        return docs

    def answer_query(self, query: str) -> str:
        docs = self.retrieve(query)
        context = self._format_context(docs)
        prompt = f"""
You are a test maker chatbot for a university course.
Use ONLY the retrieved course materials below to answer the request.
Do NOT use external knowledge.
If the answer is not present, say:
"I couldn't find that in the course materials."

When generating answers or questions:
Base them strictly on the context.

If the user asks for practice questions, a quiz, or a mini test:
Generate the requested number of questions.
Include both multiple choice and open-ended questions when possible.

If the user asks specifically for R code, answer only using code-related materials.
If the user asks for a specific lecture or day, answer only using materials from that lecture/day.

Context:
{context}

Question:
{query}

Answer clearly and then include:
Sources:
- <filename>
"""
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)

    def generate_test(self, request: str, num_questions: int = 8) -> Dict[str, Any]:
        docs = self.retrieve(request, k=max(self.retrieval_k, 10))
        context = self._format_context(docs)
        sources = [str(doc.metadata.get("source", "unknown")) for doc in docs]

        generation_prompt = f"""
You are a university test generator.
Create exactly {num_questions} questions based only on the provided context.
Use both multiple choice and short answer where reasonable.

Return only valid JSON with this schema:
{{
  "title": "string",
  "instructions": "string",
  "questions": [
    {{
      "id": "string",
      "type": "multiple_choice" or "short_answer",
      "prompt": "string",
      "options": [{{"label": "A", "text": "string"}}],
      "answer": "For multiple_choice use the correct label only, e.g., A",
      "keywords": ["keyword1", "keyword2"],
      "explanation": "string",
      "points": 1
    }}
  ]
}}

Rules:
- Every question must be answerable from context.
- If the request asks for R code or a coding test, generate questions only from code-related material.
- If the request asks for a specific lecture/day, use only materials from that lecture/day.
- For short_answer questions, include 3-6 grading keywords.
- No markdown, no code fences, no additional text.

Context:
{context}

User request:
{request}
"""
        model_output = self.llm.invoke(generation_prompt)
        content = model_output.content if hasattr(model_output, "content") else str(model_output)
        raw_payload = _extract_json(content)
        return _normalize_test_payload(raw_payload, num_questions=num_questions, sources=sources)


def _interactive_shell(rag: TestMakerRAG) -> None:
    print("Test Maker RAG is ready.")
    print("Type /test <request> to generate a practice test.")
    print("Type /ask <question> to ask a direct question.")
    print("Type exit to quit.")

    while True:
        query = input("\nYour input: ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break

        try:
            if query.startswith("/test "):
                request = query.replace("/test ", "", 1).strip()
                payload = rag.generate_test(request=request, num_questions=8)
                print(json.dumps(strip_answers(payload), indent=2))
                continue

            if query.startswith("/ask "):
                question = query.replace("/ask ", "", 1).strip()
                print(rag.answer_query(question))
                continue

            print("Use /test or /ask prefixes.")
        except Exception as exc:
            print(f"Error: {exc}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test Maker RAG")
    parser.add_argument(
        "--materials-dir",
        default=os.getenv("MATERIALS_DIR", "Materials"),
        help="Directory containing course materials.",
    )
    parser.add_argument(
        "--persist-directory",
        default=os.getenv("CHROMA_DIR", "chroma_db"),
        help="Directory for Chroma persistence.",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="OpenAI API key. If omitted, OPENAI_API_KEY is used or you will be prompted.",
    )
    parser.add_argument("--query", default="", help="Single QA query.")
    parser.add_argument("--test-request", default="", help="Single test generation request.")
    parser.add_argument("--num-questions", type=int, default=8, help="Questions in generated test.")
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Force re-indexing of all course materials.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    api_key = resolve_openai_api_key(args.api_key, prompt_if_missing=True)
    rag = TestMakerRAG(
        materials_dir=args.materials_dir,
        persist_directory=args.persist_directory,
        api_key=api_key,
        rebuild_index=args.rebuild_index,
    )

    if args.test_request:
        generated = rag.generate_test(args.test_request, num_questions=args.num_questions)
        print(json.dumps(strip_answers(generated), indent=2))
        return

    if args.query:
        print(rag.answer_query(args.query))
        return

    _interactive_shell(rag)


if __name__ == "__main__":
    main()
