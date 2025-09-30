# LangExtract Architecture and Workflow

This document provides a detailed overview of how LangExtract works, including its architecture, workflow, and the role of Large Language Models (LLMs) in the extraction process.

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Core Components](#core-components)
4. [Workflow Process](#workflow-process)
5. [LLM Integration Points](#llm-integration-points)
6. [Data Flow Diagram](#data-flow-diagram)

## Overview

LangExtract is a Python library designed to extract structured information from unstructured text using Large Language Models. The system uses a few-shot learning approach where users provide examples to guide the extraction process. The library handles:

- Document chunking for long texts
- Parallel processing of chunks
- LLM-based information extraction
- Alignment of extracted entities with source text
- Multiple extraction passes for improved recall

## System Architecture

LangExtract follows a modular architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface                          │
│                    (extraction.extract())                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Orchestration Layer                        │
│                      (Annotator Class)                          │
│  • Document chunking                                            │
│  • Batch processing                                             │
│  • Multi-pass extraction coordination                           │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
┌─────────────────┐  ┌─────────────┐  ┌──────────────┐
│ Prompt Generator│  │ LLM Provider│  │   Resolver   │
│  (Prompting)    │  │ (Inference) │  │  (Parsing)   │
└─────────────────┘  └─────────────┘  └──────────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │  Provider System│
                     │  • Gemini       │
                     │  • OpenAI       │
                     │  • Ollama       │
                     │  • Custom       │
                     └─────────────────┘
```

## Core Components

### 1. **Extraction API** (`extraction.py`)

The main entry point for users. The `extract()` function:
- Validates input parameters and examples
- Configures the language model provider
- Sets up prompt templates
- Orchestrates the annotation process
- Returns structured results

**Key Responsibilities:**
- Input validation and preprocessing
- Model configuration via factory pattern
- Format handler setup
- Resolver configuration
- Error handling

### 2. **Annotator** (`annotation.py`)

The core orchestrator that manages the document processing pipeline:

**Key Classes:**
- `Annotator`: Main class that coordinates document annotation

**Key Methods:**
- `annotate_documents()`: Process multiple documents
- `annotate_text()`: Process single text string
- `_annotate_documents_single_pass()`: Single extraction pass
- `_annotate_documents_sequential_passes()`: Multiple extraction passes for improved recall

**Responsibilities:**
- Break documents into processable chunks
- Create batches for parallel processing
- Manage LLM inference calls
- Coordinate extraction-to-source alignment
- Merge results from multiple passes

### 3. **Chunking System** (`chunking.py`)

Handles breaking long documents into manageable pieces:

**Key Classes:**
- `TextChunk`: Represents a chunk with metadata
- `ChunkIterator`: Generates chunks from documents

**Features:**
- Respects `max_char_buffer` limits
- Maintains document and position metadata
- Enables parallel processing of chunks

### 4. **Prompting System** (`prompting.py`)

Constructs prompts for the LLM:

**Key Classes:**
- `PromptTemplateStructured`: Stores prompt description and examples
- `QAPromptGenerator`: Generates question-answer style prompts

**Prompt Structure:**
```
[Description/Instructions]

[Additional Context (optional)]

Examples
Q: [Example text]
A: [Example extractions in JSON/YAML]

Q: [Input text to process]
A: 
```

**🤖 LLM INTEGRATION POINT #1: Prompt Generation**
The prompt generator creates structured prompts with few-shot examples that guide the LLM on what to extract and how to format the output.

### 5. **Provider System** (`providers/`)

Abstraction layer for different LLM backends:

**Key Components:**
- `BaseLanguageModel`: Abstract interface all providers implement
- `router.py`: Routes model IDs to appropriate providers
- Built-in providers: Gemini, OpenAI, Ollama
- Plugin system for custom providers

**Provider Selection Flow:**
1. User specifies `model_id` (e.g., "gemini-2.5-flash")
2. Factory creates ModelConfig
3. Router matches pattern to provider
4. Provider instantiated with configuration
5. Schema constraints applied (if supported)

**🤖 LLM INTEGRATION POINT #2: Provider Interface**
Each provider implements the `infer()` method that sends prompts to the actual LLM and receives structured outputs.

### 6. **Resolver** (`resolver.py`)

Parses LLM output and aligns extractions with source text:

**Key Classes:**
- `AbstractResolver`: Base interface
- `Resolver`: Main implementation
- `WordAligner`: Aligns extracted text to source positions

**Key Responsibilities:**
- Parse JSON/YAML from LLM output
- Handle fenced code blocks (```json, ```yaml)
- Convert parsed data to `Extraction` objects
- Align extractions to character positions in source text
- Support fuzzy matching when exact match fails

**Alignment Process:**
1. Extract structured data from LLM output
2. For each extraction, find its position in source text
3. Use exact token matching first
4. Fall back to fuzzy matching if needed
5. Record character intervals for highlighting

### 7. **Format Handler** (`core/format_handler.py`)

Manages serialization formats (JSON/YAML):

**Features:**
- Format extraction examples for prompts
- Parse LLM outputs
- Handle attribute naming conventions
- Support fenced vs. raw output

### 8. **Schema System** (`core/schema.py`)

Enables structured output constraints for compatible models:

**Key Features:**
- Define extraction schema from examples
- Enforce output structure at LLM generation time
- Provider-specific schema implementations (Gemini, OpenAI)
- Validation of format compatibility

**🤖 LLM INTEGRATION POINT #3: Schema Constraints**
For supported models (e.g., Gemini), schemas constrain the LLM's output generation to match the expected structure, ensuring valid JSON/YAML.

### 9. **Factory Pattern** (`factory.py`)

Creates and configures language model instances:

**Key Functions:**
- `create_model()`: Instantiate provider from configuration
- Environment variable resolution for API keys
- Provider-specific defaults
- Schema application when enabled

## Workflow Process

### Step-by-Step Execution Flow

#### Phase 1: Initialization

```
User Code
   │
   └─> lx.extract(text, prompt, examples, model_id)
          │
          ├─> Validate examples (not empty)
          ├─> Validate prompt alignment (optional)
          ├─> Create PromptTemplateStructured
          ├─> Create ModelConfig
          └─> Create language model via factory
                 │
                 └─> Router selects provider
                 └─> Apply schema constraints (if enabled)
                 └─> Return BaseLanguageModel instance
```

**🤖 LLM Integration**: Model provider is selected and configured with API credentials.

#### Phase 2: Document Processing Setup

```
extract() function
   │
   ├─> Create FormatHandler (JSON/YAML)
   ├─> Create Resolver (parsing & alignment)
   ├─> Create QAPromptGenerator
   └─> Create Annotator
          │
          └─> annotate_text() or annotate_documents()
```

#### Phase 3: Chunking & Batch Creation

```
Annotator.annotate_documents()
   │
   ├─> Convert text to Document objects
   │      │
   │      └─> Tokenize text
   │
   ├─> Create TextChunks via ChunkIterator
   │      │
   │      ├─> Respect max_char_buffer limit
   │      ├─> Track token/char intervals
   │      └─> Maintain document metadata
   │
   └─> Group chunks into batches (batch_length)
```

**Why Chunking?**
- LLMs have context window limits
- Smaller chunks enable parallel processing
- Each chunk is processed independently

#### Phase 4: Prompt Generation & LLM Inference

```
For each batch of chunks:
   │
   ├─> For each chunk in batch:
   │      │
   │      └─> QAPromptGenerator.render(chunk_text)
   │             │
   │             ├─> Add description/instructions
   │             ├─> Add few-shot examples
   │             ├─> Add chunk text as question
   │             └─> Return complete prompt
   │
   └─> Language Model.infer(batch_prompts)  🤖 LLM CALL
          │
          ├─> Send prompts to LLM API
          ├─> Apply schema constraints (if enabled)
          ├─> Receive structured outputs
          └─> Return ScoredOutput objects
```

**🤖 LLM INTEGRATION POINT #4: Inference**
This is where the actual LLM processing happens. The provider sends HTTP requests to the LLM API (Gemini, OpenAI, Ollama, etc.) with the structured prompts and receives extraction results.

**What the LLM Does:**
1. Reads the prompt instructions
2. Studies the few-shot examples
3. Analyzes the input text chunk
4. Identifies entities matching the example patterns
5. Generates structured output (JSON/YAML)
6. Respects schema constraints (if enabled)

#### Phase 5: Output Resolution & Alignment

```
For each chunk result:
   │
   ├─> Get top-scored LLM output
   │
   ├─> Resolver.resolve(llm_output_text)
   │      │
   │      ├─> Extract content from fences (if needed)
   │      ├─> Parse JSON/YAML into dictionaries
   │      └─> Create Extraction objects
   │
   └─> Resolver.align(extractions, chunk_text, offsets)
          │
          ├─> For each extraction:
          │      │
          │      ├─> Find extraction_text in chunk
          │      ├─> Calculate character positions
          │      ├─> Use fuzzy matching if exact fails
          │      └─> Store CharInterval
          │
          └─> Return aligned Extraction objects
```

**Alignment Example:**
```
Chunk text: "Patient has diabetes and hypertension."
LLM output: {"extractions": [{"medical_condition": "diabetes"}]}

Alignment:
- Find "diabetes" at position 12-20
- Create CharInterval(start_pos=12, end_pos=20)
- This enables highlighting in visualization
```

#### Phase 6: Result Aggregation

```
Single Pass Mode:
   │
   └─> Collect extractions from all chunks
   └─> Return AnnotatedDocument

Multiple Pass Mode:
   │
   ├─> Run full extraction N times
   ├─> Collect extractions from each pass
   ├─> Merge non-overlapping extractions
   │      │
   │      ├─> First pass wins for overlaps
   │      └─> Add unique extractions from later passes
   │
   └─> Return AnnotatedDocument with merged results
```

**Multi-Pass Strategy:**
- Each pass processes the entire document independently
- Helps find entities missed in earlier passes
- Increases recall at the cost of more LLM calls
- Overlapping entities resolved by first-pass priority

#### Phase 7: Return Results

```
AnnotatedDocument
   │
   ├─> document_id: str
   ├─> text: str (original)
   └─> extractions: List[Extraction]
          │
          └─> Each Extraction contains:
                 ├─> extraction_text: str
                 ├─> extraction_class: str
                 ├─> attributes: Dict[str, Any]
                 └─> char_interval: CharInterval
                        ├─> start_pos: int
                        └─> end_pos: int
```

## LLM Integration Points

The LLM plays a crucial role at several points in the workflow:

### 🤖 Point #1: Schema Definition (Optional)
**Location:** `factory.create_model()` → `provider.apply_schema()`

**When:** During model initialization, if `use_schema_constraints=True`

**What:** 
- System analyzes the few-shot examples
- Extracts the output schema (entity types, attributes)
- Converts to provider-specific schema format
- Applies to LLM for constrained generation

**Impact:**
- Ensures LLM outputs valid JSON/YAML structure
- Reduces parsing errors
- Enforces expected field names and types

### 🤖 Point #2: Prompt Construction
**Location:** `QAPromptGenerator.render()`

**When:** For each text chunk before inference

**What:**
- Combines task description with few-shot examples
- Formats examples in JSON/YAML
- Adds the input chunk as a question
- Creates a structured prompt that guides LLM behavior

**Prompt Components:**
1. **Instructions**: What to extract and how to format
2. **Few-shot examples**: Input→Output demonstrations
3. **Input text**: The chunk to process
4. **Answer prefix**: Signals where LLM should start generating

### 🤖 Point #3: Inference (Main LLM Call)
**Location:** `BaseLanguageModel.infer()`

**When:** For each batch of chunks

**What:**
- Provider sends HTTP request to LLM API
- LLM processes the prompt
- LLM generates structured extraction output
- Provider receives and returns results

**LLM's Task:**
- Read and understand the task from prompt
- Learn patterns from few-shot examples
- Analyze the input text chunk
- Identify relevant entities and relationships
- Generate structured output matching example format
- Follow schema constraints if enabled

**Example LLM Input:**
```
Extract medical conditions from text. Output as JSON.

Examples
Q: Patient has diabetes.
A: ```json
{"extractions": [{"medical_condition": "diabetes"}]}
```

Q: Patient diagnosed with hypertension and asthma.
A: 
```

**Example LLM Output:**
```json
{
  "extractions": [
    {"medical_condition": "hypertension"},
    {"medical_condition": "asthma"}
  ]
}
```

### 🤖 Point #4: Output Validation
**Location:** `Resolver.resolve()`

**When:** After LLM returns output for each chunk

**What:**
- Parse LLM's JSON/YAML output
- Validate structure matches expected format
- Convert to internal Extraction objects
- Handle parsing errors gracefully

**LLM Output Quality Factors:**
- Proper JSON/YAML syntax
- Correct field names matching examples
- Extracted text appears in source
- Reasonable attribute values

## Data Flow Diagram

Here's a complete data flow showing how data transforms through the system:

```
Input Text (str)
   │
   ▼
[Document Object]
   • text: str
   • tokenized_text: TokenizedText
   • document_id: str
   │
   ▼
[Chunking]
   │
   ▼
[TextChunk] × N
   • chunk_text: str
   • token_interval: TokenInterval
   • char_interval: CharInterval
   • document_id: str
   │
   ▼
[Batch of Chunks] × M
   │
   ▼
[Prompt Generation]
   │
   ▼
[Prompts (str)] × N
   │
   │    🤖 LLM PROCESSING
   ▼         (API Call)
[LLM Inference]
   │
   ▼
[ScoredOutput] × N
   • output: str (JSON/YAML)
   • score: float
   │
   ▼
[Resolution]
   │
   ▼
[Extraction Dicts] × K
   • entity_type: str
   • entity_attributes: dict
   │
   ▼
[Extraction Objects] × K
   • extraction_text: str
   • extraction_class: str
   • attributes: dict
   │
   ▼
[Alignment]
   │
   ▼
[Aligned Extractions] × K
   • extraction_text: str
   • extraction_class: str
   • attributes: dict
   • char_interval: CharInterval
   │
   ▼
[AnnotatedDocument]
   • document_id: str
   • text: str
   • extractions: List[Extraction]
   │
   ▼
Output (returned to user)
```

## Key Design Decisions

### 1. **Chunking Strategy**
- **Why:** Handle documents larger than LLM context windows
- **How:** Break into overlapping or non-overlapping chunks
- **Trade-off:** May miss entities spanning chunk boundaries

### 2. **Batch Processing**
- **Why:** Improve throughput via parallelization
- **How:** Process multiple chunks simultaneously
- **Trade-off:** Higher memory usage, requires thread-safe providers

### 3. **Few-Shot Learning**
- **Why:** Guide LLM without fine-tuning
- **How:** Include input-output examples in prompts
- **Trade-off:** Consumes context window space

### 4. **Schema Constraints**
- **Why:** Ensure valid, parseable outputs
- **How:** Provider-specific structured generation
- **Trade-off:** Only supported by some providers

### 5. **Multi-Pass Extraction**
- **Why:** Improve recall, find missed entities
- **How:** Run extraction multiple times, merge results
- **Trade-off:** Increased API costs and latency

### 6. **Text Alignment**
- **Why:** Enable visualization and verification
- **How:** Match extracted text to source positions
- **Trade-off:** Fuzzy matching may be imprecise

## Performance Considerations

### Factors Affecting Speed
1. **Chunk size** (`max_char_buffer`): Smaller = more API calls
2. **Batch size** (`batch_length`): Larger = better parallelization
3. **Max workers** (`max_workers`): Provider-dependent parallelism
4. **Extraction passes** (`extraction_passes`): Linear cost multiplier
5. **LLM latency**: Network and model processing time

### Factors Affecting Cost
1. **Input tokens**: Prompt + examples + chunk text
2. **Output tokens**: JSON/YAML extraction results
3. **Number of chunks**: Total document length / chunk size
4. **Extraction passes**: Reprocesses all tokens N times
5. **API pricing**: Provider-specific (e.g., per 1M tokens)

### Optimization Tips
- Use larger `max_char_buffer` for fewer API calls
- Set `batch_length >= max_workers` for full parallelization
- Use `extraction_passes=1` unless recall is critical
- Choose efficient models (e.g., Gemini Flash vs. Pro)
- Enable schema constraints to reduce malformed outputs

## Error Handling

### Common Error Scenarios

1. **Invalid API Key**
   - Caught during model creation
   - Check environment variables

2. **LLM Output Parsing Failure**
   - Resolver catches malformed JSON/YAML
   - Falls back to empty extractions
   - Logged for debugging

3. **Alignment Failure**
   - Extracted text not found in source
   - Fuzzy matching attempts recovery
   - Logs warning, continues processing

4. **Rate Limiting**
   - Provider-specific handling
   - May require retry logic
   - Consider batch_length reduction

## Summary

LangExtract orchestrates a complex pipeline to extract structured information from text:

1. **Input Processing**: Validate inputs, configure models
2. **Chunking**: Break long documents into processable pieces
3. **Prompt Generation**: Create few-shot prompts with examples
4. **🤖 LLM Inference**: Send prompts to LLM, receive extractions
5. **Resolution**: Parse LLM output into structured objects
6. **Alignment**: Map extractions to source text positions
7. **Aggregation**: Merge results from multiple chunks/passes
8. **Output**: Return AnnotatedDocument with grounded extractions

**The LLM's role** is to analyze text chunks based on few-shot examples and generate structured extractions. The surrounding infrastructure handles chunking, batching, parsing, alignment, and visualization to create a robust, production-ready extraction system.

## Related Documentation

- [Provider System](../langextract/providers/README.md) - Details on LLM provider architecture
- [README.md](../README.md) - Quick start guide and examples
- [CONTRIBUTING.md](../CONTRIBUTING.md) - Development guidelines
