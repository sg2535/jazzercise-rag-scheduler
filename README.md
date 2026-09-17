# Jazzercise Monthly Schedule Generator — RAG Application

## Problem Statement

Creating monthly Jazzercise schedules across multiple Honolulu locations
requires interpreting historical calendars, recurring class schedules,
room assignments, cancellations, and special scheduling exceptions.

One particularly important business rule occurs at the Waikiki Community
Center, where Tuesday classes normally use the Auditorium but move to
Room 205E on the second Tuesday of each month.

This project implements a Retrieval-Augmented Generation (RAG) application
that retrieves schedule information from historical calendars and explicit
business rules and generates grounded answers with source citations.

---

## Project Goals

- Convert historical Jazzercise Excel calendars into a searchable corpus.
- Preserve scheduling relationships through schedule-aware chunking.
- Add explicit room and exception rules as an additional knowledge source.
- Generate embeddings locally.
- Store and retrieve vectors using ChromaDB.
- Route exact-date and general questions through appropriate retrieval strategies.
- Generate grounded answers using a local LLM.
- Provide source citations.
- Avoid hallucination when the corpus does not contain an answer.
- Evaluate retrieval and answer quality.

---

## Corpus

The RAG corpus contains two source types.

### Historical Calendar

`Calendar_LHKH_online 2026.xlsx`

January through August 2026 historical schedules were extracted.

The calendars contain schedules for:

- Waikiki Community Center
- Palolo Hongwanji
- St. Peters

### Scheduling Rules

`waikiki_schedule_rules.md`

This source contains explicit Waikiki room-assignment rules and exceptions,
including:

- Monday 6:00 PM — Auditorium
- Tuesday 6:00 PM — Auditorium
- Second Tuesday exception — Room 205E
- Additional Thursday, Saturday, and Sunday room rules

---

## Chunking Strategy

The initial corpus treated each Excel calendar cell as one document.

This caused a chunking problem because a Monday cell could contain multiple
locations:

- Palolo — 8:30 AM
- St. Peters — 5:00 PM
- Waikiki — 6:00 PM

The corpus was redesigned using schedule-aware chunking.

Each historical schedule chunk represents:

`one date + one location + one time/status`

Scheduling rules are also separated into individual semantic chunks.

Final corpus:

- 368 historical calendar documents
- 6 scheduling-rule documents
- 374 total LangChain Documents

---

## Pre-Embedding Corpus Validation

Before generating embeddings, the corpus was validated for:

- Second-Tuesday Room 205E exception
- Duplicate records
- Required metadata
- Cancellation records
- Location normalization

Validation result:

- Second Tuesday rule: PASS
- Duplicate records: 0
- Missing required metadata: 0

---

## Technology Stack

| Layer | Technology |
|---|---|
| RAG framework | LangChain |
| Workflow orchestration | LangGraph |
| Embeddings | Hugging Face `all-MiniLM-L6-v2` |
| Embedding dimensions | 384 |
| Vector database | ChromaDB |
| Generation model | Llama 3.2 3B |
| Local model runtime | Ollama |
| Source processing | Python / openpyxl |
| Corpus formats | Excel, Markdown, JSON |

The application runs locally and does not require a paid OpenAI API.

---

## RAG Architecture

```text
Historical Excel Calendar
          +
Waikiki Scheduling Rules
          |
          v
Extract / Normalize
          |
          v
Schedule-Aware Chunking
          |
          v
374 LangChain Documents
          |
          v
Corpus Validation
          |
          v
Hugging Face Embeddings
all-MiniLM-L6-v2
          |
          v
       ChromaDB
          |
          v
      LangGraph
          |
     +----+----+
     |         |
 Exact Date   General
     |         |
 Metadata    Semantic
 Retrieval   Retrieval
     |         |
     +----+----+
          |
          v
 Retrieved Context
          |
          v
Ollama / Llama 3.2 3B
          |
          v
Grounded Answer + Citation