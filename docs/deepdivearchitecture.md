# LangExtract Deep Dive Architecture

This document provides an in-depth technical explanation of LangExtract's architecture, breaking down each component to its most granular mechanisms and detailing how they interact.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Extraction API Layer](#extraction-api-layer)
3. [Annotator Orchestration Layer](#annotator-orchestration-layer)
4. [Resolver System - Deep Dive](#resolver-system---deep-dive)
5. [Chunking System - Deep Dive](#chunking-system---deep-dive)
6. [Prompting System - Deep Dive](#prompting-system---deep-dive)
7. [Provider System - Deep Dive](#provider-system---deep-dive)
8. [Format Handler - Deep Dive](#format-handler---deep-dive)
9. [Schema System - Deep Dive](#schema-system---deep-dive)
10. [Data Structures and Transformations](#data-structures-and-transformations)
11. [Component Interaction Patterns](#component-interaction-patterns)

---

## Architecture Overview

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          USER APPLICATION LAYER                         │
│                         lx.extract(text, ...)                           │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    EXTRACTION API (extraction.py)                       │
│  • Input validation & normalization                                     │
│  • Model configuration via Factory                                      │
│  • Component initialization (FormatHandler, Resolver, Annotator)        │
│  • Example validation                                                   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
        ┌────────────────┐  ┌─────────────┐  ┌──────────────┐
        │ FormatHandler  │  │   Factory   │  │   Resolver   │
        └────────────────┘  └─────────────┘  └──────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                 ANNOTATOR ORCHESTRATION (annotation.py)                 │
│  • Document iteration & chunking                                        │
│  • Batch formation & parallel processing                                │
│  • Multi-pass extraction coordination                                   │
│  • Result aggregation & merging                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            ▼                       ▼                       ▼
    ┌───────────────┐      ┌──────────────┐      ┌─────────────────┐
    │   Chunking    │      │  Prompting   │      │    Provider     │
    │   System      │      │   System     │      │     System      │
    └───────────────┘      └──────────────┘      └─────────────────┘
            │                      │                       │
            │                      │              ┌────────┴────────┐
            ▼                      ▼              ▼                 ▼
    ┌───────────────┐      ┌──────────────┐  ┌─────────┐    ┌──────────┐
    │  TextChunk    │      │ QAPrompt     │  │ Router  │    │ Gemini/  │
    │  Iterator     │      │ Generator    │  │         │    │ OpenAI/  │
    └───────────────┘      └──────────────┘  └─────────┘    │ Ollama   │
                                                             └──────────┘
                                    │
                                    ▼ LLM INFERENCE
                              (External API Call)
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   RESOLVER & ALIGNMENT (resolver.py)                    │
│  • Format parsing (JSON/YAML)                                           │
│  • Fence extraction                                                     │
│  • Extraction object creation                                           │
│  • Token-based alignment (WordAligner)                                  │
│  • Fuzzy matching fallback                                              │
│  • Character interval mapping                                           │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         ANNOTATED DOCUMENT                              │
│  • document_id: str                                                     │
│  • text: str (original)                                                 │
│  • extractions: List[Extraction]                                        │
│    - extraction_text: str                                               │
│    - extraction_class: str                                              │
│    - attributes: Dict                                                   │
│    - char_interval: CharInterval (start_pos, end_pos)                  │
│    - alignment_status: MATCH_EXACT | MATCH_LESSER | MATCH_FUZZY        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Extraction API Layer

### Location
`langextract/extraction.py` - The `extract()` function

### Purpose
The main entry point that orchestrates the entire extraction pipeline. It validates inputs, configures components, and delegates to the Annotator.

### Detailed Mechanism

#### Step 1: Input Validation
```python
def extract(text_or_documents, prompt_description, examples, model_id, ...):
    # 1.1: Validate examples are provided
    if not examples:
        raise ValueError("Examples are required...")
```

**What happens:**
- Checks that `examples` list is non-empty
- Examples are critical for few-shot learning
- Without examples, the LLM has no pattern to follow

#### Step 2: Prompt Alignment Validation
```python
    # 1.2: Validate alignment of examples (optional pre-flight check)
    if prompt_validation_level is not OFF:
        report = validate_prompt_alignment(examples, aligner, policy)
        handle_alignment_report(report, level, strict)
```

**What happens:**
- Creates a `WordAligner` instance
- Checks if example extractions can be aligned with their source text
- Reports misalignments (MATCH_FUZZY, MATCH_LESSER, no match)
- Can raise errors or warnings based on validation level
- Helps catch problematic examples before expensive LLM calls

#### Step 3: URL Handling
```python
    # 1.3: Download text from URL if needed
    if fetch_urls and isinstance(text_or_documents, str) and is_url(text_or_documents):
        text_or_documents = download_text_from_url(text_or_documents)
```

**What happens:**
- Detects HTTP/HTTPS URLs in text input
- Downloads content via HTTP request
- Converts URL to actual text for processing
- Can be disabled with `fetch_urls=False`

#### Step 4: Prompt Template Creation
```python
    # 1.4: Create structured prompt template
    prompt_template = PromptTemplateStructured(description=prompt_description)
    prompt_template.examples.extend(examples)
```

**What happens:**
- Wraps user's description and examples in a structured object
- `description` becomes the instruction text
- `examples` are `ExampleData` objects with text + extractions
- This template will be used by `QAPromptGenerator` later

#### Step 5: Model Configuration & Creation
```python
    # 1.5: Create or use provided language model
    if model:
        language_model = model
    elif config:
        language_model = factory.create_model(config, examples, use_schema_constraints)
    else:
        # Build config from parameters
        config = ModelConfig(model_id=model_id, provider_kwargs={...})
        language_model = factory.create_model(config, examples, use_schema_constraints)
```

**What happens:**
- Three paths: explicit model, explicit config, or build config
- `factory.create_model()` orchestrates:
  - Provider resolution via `router.resolve(model_id)`
  - API key resolution from environment
  - Schema generation from examples (if `use_schema_constraints=True`)
  - Provider instantiation with kwargs
- Returns a `BaseLanguageModel` instance ready for inference

**Schema Application (if enabled):**
```python
    if use_schema_constraints and examples:
        schema_class = provider_class.get_schema_class()
        schema_instance = schema_class.from_examples(examples)
        language_model.apply_schema(schema_instance)
```

**What happens:**
- Analyzes example extractions to determine structure
- Creates provider-specific schema (GeminiSchema, OpenAISchema)
- Applies to model for constrained generation
- Schema ensures LLM outputs match expected format

#### Step 6: FormatHandler Creation
```python
    # 1.6: Create format handler for prompt generation and parsing
    format_handler = FormatHandler.from_resolver_params(
        resolver_params=resolver_params,
        base_format_type=format_type,
        base_use_fences=language_model.requires_fence_output,
        base_attribute_suffix=ATTRIBUTE_SUFFIX,
        base_use_wrapper=True,
        base_wrapper_key=EXTRACTIONS_KEY,
    )
```

**What happens:**
- Normalizes format configuration
- Determines if fences (```json/```yaml) are needed
- Sets attribute suffix (e.g., "_attributes")
- Configures wrapper key (e.g., "extractions")
- Validates schema compatibility with format

#### Step 7: Resolver Creation
```python
    # 1.7: Create resolver for parsing LLM output
    alignment_kwargs = {
        'enable_fuzzy_alignment': True,
        'fuzzy_alignment_threshold': 0.75,
        'accept_match_lesser': True,
    }
    resolver = Resolver(format_handler=format_handler, **remaining_params)
```

**What happens:**
- Creates resolver with format handler
- Extracts alignment parameters from `resolver_params`
- Resolver will parse LLM output and align extractions

#### Step 8: Annotator Creation & Execution
```python
    # 1.8: Create annotator and process documents
    annotator = Annotator(
        language_model=language_model,
        prompt_template=prompt_template,
        format_handler=format_handler,
    )
    
    # 1.9: Execute annotation
    if isinstance(text_or_documents, str):
        return annotator.annotate_text(text, resolver, max_char_buffer, ...)
    else:
        return annotator.annotate_documents(documents, resolver, ...)
```

**What happens:**
- Creates orchestrator with all configured components
- Delegates to appropriate annotation method
- Returns `AnnotatedDocument` or iterator of `AnnotatedDocument`

---

## Annotator Orchestration Layer

### Location
`langextract/annotation.py` - The `Annotator` class

### Purpose
Orchestrates the entire document processing pipeline: chunking, batching, prompting, inference, resolution, and alignment.

### Key Components

#### Component 1: Annotator Initialization
```python
class Annotator:
    def __init__(self, language_model, prompt_template, format_handler):
        self._language_model = language_model
        self._prompt_generator = QAPromptGenerator(
            template=prompt_template,
            format_handler=format_handler,
        )
```

**What happens:**
- Stores the language model for inference
- Creates prompt generator with template and format handler
- Prompt generator will render prompts for each chunk

### Detailed Flow: annotate_documents()

#### Phase 1: Document Preparation
```python
def annotate_documents(self, documents, resolver, max_char_buffer, batch_length, ...):
    # Create two iterators from documents
    doc_iter, doc_iter_for_chunks = itertools.tee(documents, 2)
    curr_document = next(doc_iter, None)
```

**What happens:**
- Creates two independent iterators from input documents
- `doc_iter`: tracks current document for result aggregation
- `doc_iter_for_chunks`: consumed by chunking system
- `curr_document`: current document being processed

#### Phase 2: Chunking
```python
    # Create chunk iterator
    chunk_iter = _document_chunk_iterator(doc_iter_for_chunks, max_char_buffer)
```

**What happens:**
- `_document_chunk_iterator()` wraps multiple documents
- For each document:
  - Tokenizes text via `tokenizer.TokenizedText(text)`
  - Creates `ChunkIterator(tokenized_text, max_char_buffer)`
  - Yields `TextChunk` objects with token/char intervals
- Maintains document boundaries and IDs
- Prevents duplicate document IDs (raises `DocumentRepeatError`)

#### Phase 3: Batch Formation
```python
    # Group chunks into batches
    batches = chunking.make_batches_of_textchunk(chunk_iter, batch_length)
```

**What happens:**
- Uses `more_itertools.batched()` to group chunks
- Each batch contains up to `batch_length` chunks
- Chunks can come from different documents
- Enables parallel processing of multiple chunks

#### Phase 4: Progress Bar Setup
```python
    model_info = progress.get_model_info(language_model)
    progress_bar = create_extraction_progress_bar(batches, model_info, disable=not show_progress)
```

**What happens:**
- Extracts model name/info for display
- Creates tqdm progress bar
- Tracks batch processing progress
- Shows estimated time and throughput

#### Phase 5: Batch Processing Loop
```python
    for index, batch in enumerate(progress_bar):
        # 5.1: Render prompts for each chunk
        batch_prompts = []
        for text_chunk in batch:
            prompt = self._prompt_generator.render(
                question=text_chunk.chunk_text,
                additional_context=text_chunk.additional_context,
            )
            batch_prompts.append(prompt)
```

**What happens:**
- Iterates over each batch of chunks
- For each chunk, generates a prompt
- `render()` combines:
  - Task description
  - Few-shot examples
  - Chunk text as question
  - Answer prefix
- Collects all prompts for batch

**Example prompt structure:**
```
Extract medical conditions from text. Output as JSON.

Examples
Q: Patient has diabetes.
A: ```json
{"extractions": [{"medical_condition": "diabetes"}]}
```

Q: <chunk_text>
A: 
```

#### Phase 6: LLM Inference
```python
        # 5.2: Call language model
        batch_scored_outputs = self._language_model.infer(
            batch_prompts=batch_prompts,
            **kwargs,
        )
```

**What happens:**
- Sends batch of prompts to LLM provider
- Provider handles:
  - API authentication
  - Request formatting
  - Schema constraints (if enabled)
  - Parallel/sequential processing
  - Error handling & retries
- Returns `List[List[ScoredOutput]]`
- Each `ScoredOutput` has:
  - `output`: str (LLM's text response)
  - `score`: float (confidence, usually 1.0)

**LLM Processing:**
1. Reads instruction and examples
2. Learns extraction pattern
3. Analyzes chunk text
4. Identifies matching entities
5. Generates structured output (JSON/YAML)
6. Respects schema constraints

#### Phase 7: Resolution & Alignment
```python
        # 5.3: Process each chunk's results
        for text_chunk, scored_outputs in zip(batch, batch_scored_outputs):
            # Get top result
            top_inference_result = scored_outputs[0].output
            
            # 5.3a: Parse LLM output into Extraction objects
            annotated_chunk_extractions = resolver.resolve(
                top_inference_result, debug=debug, **kwargs
            )
```

**Resolution mechanism (detailed in Resolver section):**
1. Extract content from fences if present
2. Parse JSON/YAML into dictionaries
3. Create `Extraction` objects
4. Sort by index if configured

```python
            # 5.3b: Align extractions with source text
            chunk_text = text_chunk.chunk_text
            token_offset = text_chunk.token_interval.start_index
            char_offset = text_chunk.char_interval.start_pos
            
            aligned_extractions = resolver.align(
                annotated_chunk_extractions,
                chunk_text,
                token_offset,
                char_offset,
                **kwargs,
            )
```

**Alignment mechanism (detailed in Resolver section):**
1. Tokenize source chunk
2. For each extraction, find position in source
3. Set token intervals and char intervals
4. Use fuzzy matching if exact fails
5. Mark alignment status

#### Phase 8: Document Boundary Handling
```python
            # 5.4: Check if we've finished current document
            while curr_document.document_id != text_chunk.document_id:
                # Yield completed document
                annotated_doc = AnnotatedDocument(
                    document_id=curr_document.document_id,
                    extractions=annotated_extractions,
                    text=curr_document.text,
                )
                yield annotated_doc
                annotated_extractions.clear()
                curr_document = next(doc_iter, None)
```

**What happens:**
- Compares chunk's document ID with current document
- When IDs differ, current document is complete
- Yields `AnnotatedDocument` with all extractions
- Clears extraction list for next document
- Advances to next document

#### Phase 9: Accumulation
```python
            # 5.5: Accumulate extractions for current document
            annotated_extractions.extend(aligned_extractions)
```

**What happens:**
- Appends chunk's extractions to document's list
- Extractions accumulate across all chunks of a document
- Maintains order (chunk order within document)

#### Phase 10: Final Document
```python
    # After all batches processed
    if curr_document is not None:
        annotated_doc = AnnotatedDocument(
            document_id=curr_document.document_id,
            extractions=annotated_extractions,
            text=curr_document.text,
        )
        yield annotated_doc
```

**What happens:**
- Yields the final document's results
- Ensures last document isn't dropped
- Completes iteration

### Multi-Pass Extraction

#### Mechanism: Sequential Passes
```python
def _annotate_documents_sequential_passes(self, documents, resolver, extraction_passes, ...):
    document_list = list(documents)
    document_extractions_by_pass = {}
    
    # Run extraction N times
    for pass_num in range(extraction_passes):
        for annotated_doc in self._annotate_documents_single_pass(document_list, ...):
            doc_id = annotated_doc.document_id
            if doc_id not in document_extractions_by_pass:
                document_extractions_by_pass[doc_id] = []
            document_extractions_by_pass[doc_id].append(annotated_doc.extractions)
```

**What happens:**
- Converts documents to list (for re-iteration)
- Runs single-pass extraction multiple times
- Stores extractions from each pass separately
- Each pass is independent (no state sharing)

#### Merging Strategy
```python
    # Merge results
    for doc_id, all_pass_extractions in document_extractions_by_pass.items():
        merged = _merge_non_overlapping_extractions(all_pass_extractions)
        yield AnnotatedDocument(doc_id, merged, text)
```

**Merge algorithm:**
```python
def _merge_non_overlapping_extractions(all_extractions):
    merged = list(all_extractions[0])  # Start with first pass
    
    for pass_extractions in all_extractions[1:]:
        for extraction in pass_extractions:
            overlaps = False
            if extraction.char_interval:
                for existing in merged:
                    if _extractions_overlap(extraction, existing):
                        overlaps = True
                        break
            if not overlaps:
                merged.append(extraction)
    return merged
```

**What happens:**
- Starts with all extractions from first pass
- For each subsequent pass:
  - Check each extraction for overlap with merged set
  - Overlap = character intervals intersect
  - Add only non-overlapping extractions
- First pass wins for overlapping entities
- Improves recall by finding missed entities

---

## Resolver System - Deep Dive

### Location
`langextract/resolver.py` - `Resolver` and `WordAligner` classes

### Purpose
Parse LLM's textual output into structured `Extraction` objects and align them with source text positions.

### Component 1: Initialization

```python
class Resolver(AbstractResolver):
    def __init__(self, format_handler=None, extraction_index_suffix=None, **kwargs):
        # Handle legacy parameters
        if format_handler is None:
            format_handler = FormatHandler.from_kwargs(**kwargs)
        
        self.format_handler = format_handler
        self.extraction_index_suffix = extraction_index_suffix  # e.g., "_index"
        self._constraint = constraint or Constraint()
```

**What happens:**
- Stores format handler for parsing
- Stores index suffix for sorting (optional)
- Supports legacy parameter conversion
- Stores constraint for strict mode

### Component 2: resolve() - Parsing LLM Output

#### Step 1: Parse Output
```python
def resolve(self, input_text, suppress_parse_errors=False, **kwargs):
    # 2.1: Parse LLM output string
    constraint = getattr(self, "_constraint", Constraint())
    strict = getattr(constraint, "strict", False)
    extraction_data = self.format_handler.parse_output(input_text, strict=strict)
```

**Format handler parsing (detailed in FormatHandler section):**
1. Strip whitespace from input
2. Extract content from fences if present (```json...```)
3. Parse JSON or YAML into Python objects
4. Validate structure (requires list of dicts or dict with wrapper key)
5. Return `Sequence[Mapping[str, ExtractionValueType]]`

**Example:**
```
Input: '```json\n{"extractions": [{"condition": "diabetes"}]}\n```'
Output: [{"condition": "diabetes"}]
```

#### Step 2: Create Extraction Objects
```python
    # 2.2: Convert dictionaries to Extraction objects
    processed_extractions = self.extract_ordered_extractions(extraction_data)
    return processed_extractions
```

### Component 3: extract_ordered_extractions() - Object Creation

#### Mechanism
```python
def extract_ordered_extractions(self, extraction_data):
    processed_extractions = []
    extraction_index = 0
    index_suffix = self.extraction_index_suffix  # e.g., "_index"
    attributes_suffix = self.format_handler.attribute_suffix  # e.g., "_attributes"
    
    for group_index, group in enumerate(extraction_data):
        for extraction_class, extraction_value in group.items():
            # 3.1: Skip index keys
            if index_suffix and extraction_class.endswith(index_suffix):
                continue
            
            # 3.2: Skip attribute keys
            if attributes_suffix and extraction_class.endswith(attributes_suffix):
                continue
            
            # 3.3: Validate extraction value is string/int/float
            if not isinstance(extraction_value, (str, int, float)):
                raise ValueError("Extraction text must be string, int, or float")
            
            if not isinstance(extraction_value, str):
                extraction_value = str(extraction_value)
            
            # 3.4: Determine extraction index
            if index_suffix:
                index_key = extraction_class + index_suffix
                extraction_index = group.get(index_key, None)
                if extraction_index is None:
                    continue  # Skip extractions without index
            else:
                extraction_index += 1  # Sequential
            
            # 3.5: Extract attributes
            attributes = None
            if attributes_suffix:
                attributes_key = extraction_class + attributes_suffix
                attributes = group.get(attributes_key, None)
            
            # 3.6: Create Extraction object
            processed_extractions.append(
                Extraction(
                    extraction_class=extraction_class,
                    extraction_text=extraction_value,
                    extraction_index=extraction_index,
                    group_index=group_index,
                    attributes=attributes,
                )
            )
    
    # 3.7: Sort by index
    processed_extractions.sort(key=operator.attrgetter("extraction_index"))
    return processed_extractions
```

**What happens:**
- Iterates over parsed dictionaries
- For each key-value pair:
  - Checks if it's an index key (skip)
  - Checks if it's an attribute key (skip)
  - Validates value type
  - Extracts index if configured
  - Extracts attributes if configured
  - Creates `Extraction` object
- Sorts by `extraction_index`
- Returns ordered list

**Example transformation:**
```python
Input: [
    {
        "condition": "diabetes",
        "condition_index": 4,
        "condition_attributes": {"severity": "type2"}
    },
    {
        "condition": "hypertension",
        "condition_index": 9
    }
]

Output: [
    Extraction(
        extraction_class="condition",
        extraction_text="diabetes",
        extraction_index=4,
        attributes={"severity": "type2"},
    ),
    Extraction(
        extraction_class="condition",
        extraction_text="hypertension",
        extraction_index=9,
        attributes=None,
    )
]
```

### Component 4: align() - Alignment Orchestration

```python
def align(self, extractions, source_text, token_offset, char_offset,
          enable_fuzzy_alignment=True, fuzzy_alignment_threshold=0.75,
          accept_match_lesser=True, **kwargs):
    
    if not extractions:
        return
    
    # 4.1: Create aligner
    aligner = WordAligner()
    
    # 4.2: Perform alignment
    aligned_groups = aligner.align_extractions(
        extraction_groups=[extractions],
        source_text=source_text,
        token_offset=token_offset,
        char_offset=char_offset,
        enable_fuzzy_alignment=enable_fuzzy_alignment,
        fuzzy_alignment_threshold=fuzzy_alignment_threshold,
        accept_match_lesser=accept_match_lesser,
    )
    
    # 4.3: Yield aligned extractions
    for extraction in itertools.chain(*aligned_groups):
        yield extraction
```

**What happens:**
- Creates `WordAligner` instance
- Delegates to aligner's `align_extractions()`
- Flattens results and yields

### Component 5: WordAligner - The Core Alignment Engine

#### Purpose
Match extraction text to positions in source text using token-level alignment.

#### Initialization
```python
class WordAligner:
    def __init__(self):
        self.matcher = difflib.SequenceMatcher(autojunk=False)
        self.source_tokens = None
        self.extraction_tokens = None
```

**What happens:**
- Creates `SequenceMatcher` from Python's difflib
- `autojunk=False`: don't skip "popular" elements
- Will compare token sequences

#### Alignment Strategy Overview

```
┌────────────────────────────────────────────────────────────┐
│                  WordAligner.align_extractions()           │
└────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Tokenize    │    │  Tokenize    │    │   Prepare    │
│   Source     │    │ Extractions  │    │  Delimiter   │
│   Text       │    │   with       │    │   Mapping    │
│              │    │  Delimiters  │    │              │
└──────────────┘    └──────────────┘    └──────────────┘
        │                   │                   │
        └───────────────────┴───────────────────┘
                            │
                            ▼
        ┌────────────────────────────────────────┐
        │    difflib.SequenceMatcher.            │
        │     get_matching_blocks()              │
        │  Returns: [(i, j, n), ...]            │
        │    where source[i:i+n] == extract[j:j+n]│
        └────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ EXACT MATCH  │    │ LESSER MATCH │    │  UNMATCHED   │
│ extraction   │    │  extraction  │    │ extractions  │
│ length ==    │    │  length >    │    │  (no block   │
│ block size   │    │  block size  │    │   match)     │
└──────────────┘    └──────────────┘    └──────────────┘
        │                   │                   │
        ▼                   ▼                   ▼
  Set intervals       Set intervals        Unaligned
  status=EXACT        status=LESSER        status=None
        │                   │                   │
        └───────────────────┴───────────────────┘
                            │
                            ▼
            ┌───────────────────────────────┐
            │  Fuzzy Alignment (optional)   │
            │  For unaligned extractions    │
            └───────────────────────────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │ Sliding Window Search │
                │  with SequenceMatcher │
                └───────────────────────┘
                            │
                            ▼
                  Set intervals if ratio
                  >= fuzzy_threshold
                  status=MATCH_FUZZY
```

#### Detailed Mechanism: align_extractions()

##### Step 1: Tokenization
```python
def align_extractions(self, extraction_groups, source_text, token_offset, char_offset,
                      delim="\u241F", ...):
    # 1.1: Tokenize source text
    source_tokens = list(_tokenize_with_lowercase(source_text))
```

**`_tokenize_with_lowercase()`:**
```python
def _tokenize_with_lowercase(text):
    tokenized_pb2 = tokenizer.tokenize(text)
    for token in tokenized_pb2.tokens:
        start = token.char_interval.start_pos
        end = token.char_interval.end_pos
        token_str = tokenized_pb2.text[start:end].lower()
        yield token_str
```

**What happens:**
- Uses `tokenizer.tokenize()` (sentence tokenizer)
- Extracts each token's text using character positions
- Lowercases for case-insensitive matching
- Returns iterator of lowercase token strings

**Example:**
```
Input: "Patient has Diabetes."
Tokens: ["patient", "has", "diabetes", "."]
```

##### Step 2: Prepare Extraction Tokens with Delimiters
```python
    # 1.2: Concatenate all extractions with delimiter
    extraction_tokens = _tokenize_with_lowercase(
        f" {delim} ".join(
            extraction.extraction_text
            for extraction in itertools.chain(*extraction_groups)
        )
    )
```

**What happens:**
- Joins all extraction texts with special delimiter (`\u241F`)
- Tokenizes the concatenated string
- Delimiter must be single token (validated)
- Creates mapping from token positions to extractions

**Example:**
```
Extractions: ["diabetes", "hypertension"]
Concatenated: "diabetes \u241F hypertension"
Tokens: ["diabetes", "\u241F", "hypertension"]
```

##### Step 3: Create Extraction Index Mapping
```python
    # 1.3: Build index to extraction mapping
    index_to_extraction_group = {}
    extraction_index = 0
    for group_index, group in enumerate(extraction_groups):
        for extraction in group:
            index_to_extraction_group[extraction_index] = (extraction, group_index)
            extraction_text_tokens = list(_tokenize_with_lowercase(extraction.extraction_text))
            extraction_index += len(extraction_text_tokens) + delim_len
```

**What happens:**
- Tracks which extraction each token position corresponds to
- `extraction_index` is position in concatenated token sequence
- Skips delimiter length between extractions
- Enables reverse lookup: token position → extraction object

**Example:**
```
Token position 0: extraction "diabetes"
Token position 2: extraction "hypertension" (after delimiter at position 1)
```

##### Step 4: Set Sequences for Matching
```python
    # 1.4: Configure SequenceMatcher
    self._set_seqs(source_tokens, extraction_tokens)
```

**What happens:**
- Stores source and extraction tokens
- Configures `difflib.SequenceMatcher`
- Prepares for matching block retrieval

##### Step 5: Exact Matching Phase
```python
    # 1.5: Get matching blocks from difflib
    aligned_extractions = []
    exact_matches = 0
    lesser_matches = 0
    
    for i, j, n in self._get_matching_blocks()[:-1]:  # Skip dummy block
        # 5.1: Find extraction at position j
        extraction, _ = index_to_extraction_group.get(j, (None, None))
        if extraction is None:
            continue  # j is not start of an extraction
```

**`get_matching_blocks()` returns:**
```python
[(i, j, n), ...]  # source[i:i+n] == extraction[j:j+n]
```

**What happens:**
- Difflib finds all contiguous matching sequences
- `i`: position in source tokens
- `j`: position in extraction tokens
- `n`: length of match
- Only process blocks starting at extraction boundaries

**Example:**
```
Source tokens: ["patient", "has", "diabetes", "and", "hypertension"]
Extract tokens: ["diabetes", "\u241F", "hypertension"]

Matching blocks:
  (2, 0, 1) -> source[2:3]="diabetes" matches extract[0:1]="diabetes"
  (4, 2, 1) -> source[4:5]="hypertension" matches extract[2:3]="hypertension"
```

##### Step 6: Set Token and Char Intervals
```python
        # 5.2: Set token interval
        extraction.token_interval = TokenInterval(
            start_index=i + token_offset,
            end_index=i + n + token_offset,
        )
        
        # 5.3: Set char interval
        tokenized_text = tokenizer.tokenize(source_text)
        start_token = tokenized_text.tokens[i]
        end_token = tokenized_text.tokens[i + n - 1]
        extraction.char_interval = CharInterval(
            start_pos=char_offset + start_token.char_interval.start_pos,
            end_pos=char_offset + end_token.char_interval.end_pos,
        )
```

**What happens:**
- Token interval: marks token positions in source
- Adds `token_offset` (chunk's position in document)
- Char interval: marks character positions
- Uses tokenized text to get char positions
- Adds `char_offset` (chunk's char position in document)

**Example:**
```
Match: source[2:3] = "diabetes"
Token interval: [2, 3) (relative to chunk)
If token_offset=10: [12, 13) (absolute in document)

Char positions from tokenized text:
  Token 2: char_interval = [12, 20)
Char interval: [12, 20) (relative to chunk)
If char_offset=100: [112, 120) (absolute in document)
```

##### Step 7: Classify Match Type
```python
        # 5.4: Determine match type
        extraction_text_len = len(list(_tokenize_with_lowercase(extraction.extraction_text)))
        
        if extraction_text_len == n:
            # Perfect match
            extraction.alignment_status = AlignmentStatus.MATCH_EXACT
            exact_matches += 1
            aligned_extractions.append(extraction)
        elif extraction_text_len > n:
            # Partial match (extraction longer than matched block)
            if accept_match_lesser:
                extraction.alignment_status = AlignmentStatus.MATCH_LESSER
                lesser_matches += 1
                aligned_extractions.append(extraction)
            else:
                # Reset intervals
                extraction.token_interval = None
                extraction.char_interval = None
                extraction.alignment_status = None
```

**Match types:**
- **MATCH_EXACT**: Extraction length == block length
  - Example: extraction="diabetes", matched 1 token
- **MATCH_LESSER**: Extraction length > block length
  - Example: extraction="type 2 diabetes", matched 1 token "diabetes"
  - Partial match acceptable if `accept_match_lesser=True`
- **Unmatched**: No intervals set

##### Step 8: Collect Unaligned Extractions
```python
    # 1.6: Find extractions without matches
    unaligned_extractions = []
    for extraction, _ in index_to_extraction_group.values():
        if extraction not in aligned_extractions:
            unaligned_extractions.append(extraction)
```

**What happens:**
- Identifies extractions with no exact/lesser match
- These will attempt fuzzy alignment

#### Fuzzy Alignment - The Fallback Mechanism

##### Purpose
Find approximate matches when exact token matching fails.

##### Mechanism: _fuzzy_align_extraction()
```python
def _fuzzy_align_extraction(self, extraction, source_tokens, tokenized_text,
                            token_offset, char_offset, fuzzy_alignment_threshold=0.75):
    # 1: Tokenize and normalize extraction
    extraction_tokens = list(_tokenize_with_lowercase(extraction.extraction_text))
    extraction_tokens_norm = [_normalize_token(t) for t in extraction_tokens]
```

**`_normalize_token()`:**
```python
@functools.lru_cache(maxsize=10000)
def _normalize_token(token):
    token = token.lower()
    # Light stemming: remove trailing 's' (plurals)
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        token = token[:-1]
    return token
```

**What happens:**
- Lowercases token
- Removes plural 's' (light stemming)
- Cached for performance
- Makes "conditions" match "condition"

##### Sliding Window Search
```python
    # 2: Initialize sliding window search
    best_ratio = 0.0
    best_span = None  # (start_idx, window_size)
    
    len_e = len(extraction_tokens)
    max_window = len(source_tokens)
    
    extraction_counts = Counter(extraction_tokens_norm)
    min_overlap = int(len_e * fuzzy_alignment_threshold)
    
    matcher = difflib.SequenceMatcher(autojunk=False, b=extraction_tokens_norm)
```

**What happens:**
- Prepares to scan all possible windows in source
- Tracks best matching window
- Pre-counts extraction tokens for optimization
- Calculates minimum overlap needed

##### Window Iteration with Optimization
```python
    # 3: Scan all window sizes and positions
    for window_size in range(len_e, max_window + 1):
        if window_size > len(source_tokens):
            break
        
        # Initialize sliding window
        window_deque = collections.deque(source_tokens[0:window_size])
        window_counts = Counter([_normalize_token(t) for t in window_deque])
        
        for start_idx in range(len(source_tokens) - window_size + 1):
            # Optimization: fast pre-check
            if (extraction_counts & window_counts).total() >= min_overlap:
                # Expensive sequence matching
                window_tokens_norm = [_normalize_token(t) for t in window_deque]
                matcher.set_seq1(window_tokens_norm)
                matches = sum(size for _, _, size in matcher.get_matching_blocks())
                ratio = matches / len_e if len_e > 0 else 0.0
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_span = (start_idx, window_size)
            
            # Slide window right
            if start_idx + window_size < len(source_tokens):
                old_token = window_deque.popleft()
                window_counts[_normalize_token(old_token)] -= 1
                if window_counts[_normalize_token(old_token)] == 0:
                    del window_counts[_normalize_token(old_token)]
                
                new_token = source_tokens[start_idx + window_size]
                window_deque.append(new_token)
                window_counts[_normalize_token(new_token)] += 1
```

**Optimization strategy:**
1. **Fast pre-check**: Count token overlap using Counter intersection
   - If `(extraction_counts & window_counts).total() < min_overlap`, skip expensive matching
   - Eliminates >90% of windows early
2. **Incremental window update**: Slide window by adjusting counts
   - Remove old token from counts
   - Add new token to counts
   - O(1) per slide instead of O(window_size)
3. **SequenceMatcher only when promising**: Run difflib only on candidates

**Example:**
```
Source: "patient diagnosed with type 2 diabetes mellitus"
Extraction: "diabetes"

Window scan:
  window[0:1] = "patient" -> ratio=0.0
  window[1:2] = "diagnosed" -> ratio=0.0
  ...
  window[5:6] = "diabetes" -> ratio=1.0 ✓ BEST
  window[5:7] = "diabetes mellitus" -> ratio=0.5
```

##### Accept or Reject Fuzzy Match
```python
    # 4: Apply threshold
    if best_span and best_ratio >= fuzzy_alignment_threshold:
        start_idx, window_size = best_span
        
        # Set intervals
        extraction.token_interval = TokenInterval(
            start_index=start_idx + token_offset,
            end_index=start_idx + window_size + token_offset,
        )
        
        start_token = tokenized_text.tokens[start_idx]
        end_token = tokenized_text.tokens[start_idx + window_size - 1]
        extraction.char_interval = CharInterval(
            start_pos=char_offset + start_token.char_interval.start_pos,
            end_pos=char_offset + end_token.char_interval.end_pos,
        )
        
        extraction.alignment_status = AlignmentStatus.MATCH_FUZZY
        return extraction
    
    return None  # Alignment failed
```

**What happens:**
- If best ratio meets threshold (default 0.75), accept match
- Set token and char intervals
- Mark status as `MATCH_FUZZY`
- Otherwise, return None (alignment failed)

#### Alignment Status Summary

| Status | Meaning | How Achieved |
|--------|---------|--------------|
| `MATCH_EXACT` | Perfect token-level match | Extraction tokens exactly match source tokens via difflib |
| `MATCH_LESSER` | Partial exact match | Extraction longer than matched block, but block matches |
| `MATCH_FUZZY` | Approximate match | Sliding window search found match above threshold |
| `None` | No alignment | All alignment attempts failed |

#### Performance Characteristics

**Exact matching:**
- O(n + m) where n=source length, m=extraction length
- difflib.SequenceMatcher is efficient
- Typically aligns 70-90% of extractions

**Fuzzy matching:**
- O(n * w * e) where w=window sizes, e=extraction length
- Pre-check reduces constant factor significantly
- Only runs on unaligned extractions (10-30%)
- Most expensive operation in resolver

---

## Chunking System - Deep Dive

### Location
`langextract/chunking.py` - `ChunkIterator`, `SentenceIterator`, `TextChunk`

### Purpose
Break long documents into manageable chunks that fit within LLM context limits while respecting sentence boundaries.

### Component 1: TextChunk Data Structure

```python
@dataclasses.dataclass
class TextChunk:
    token_interval: TokenInterval  # start_index, end_index
    document: Document | None = None
    _chunk_text: str | None = None  # Cached
    _sanitized_chunk_text: str | None = None  # Cached
    _char_interval: CharInterval | None = None  # Cached
```

**Properties:**
- `document_id`: From source document
- `document_text`: Tokenized text from document
- `chunk_text`: Extracted text for this chunk (cached)
- `sanitized_chunk_text`: Whitespace normalized (cached)
- `char_interval`: Character positions (computed on access)
- `additional_context`: From document

**Lazy evaluation:**
- Text and intervals computed only when accessed
- Caches to avoid recomputation
- Efficient memory usage for large documents

### Component 2: SentenceIterator

#### Purpose
Iterate through sentences in tokenized text, starting from any token position.

#### Mechanism
```python
class SentenceIterator:
    def __init__(self, tokenized_text, curr_token_pos=0):
        self.tokenized_text = tokenized_text
        self.token_len = len(tokenized_text.tokens)
        self.curr_token_pos = curr_token_pos
    
    def __next__(self):
        if self.curr_token_pos == self.token_len:
            raise StopIteration
        
        # Find sentence containing current token
        sentence_range = tokenizer.find_sentence_range(
            self.tokenized_text.text,
            self.tokenized_text.tokens,
            self.curr_token_pos,
        )
        
        # Start from current position (may be mid-sentence)
        sentence_range = create_token_interval(
            self.curr_token_pos,
            sentence_range.end_index
        )
        
        self.curr_token_pos = sentence_range.end_index
        return sentence_range
```

**What happens:**
- Uses `tokenizer.find_sentence_range()` to find sentence boundaries
- Sentence determined by punctuation and capitalization
- If mid-sentence, returns from current position to sentence end
- Advances position to end of sentence
- Enables resumption after breaking large sentences

**Example:**
```
Text: "First sentence. Second sentence. Third sentence."
Tokens: 0-2, 3-5, 6-8

Initial position 0:
  Sentence 1: [0, 3)
  Sentence 2: [3, 6)
  Sentence 3: [6, 9)

Mid-sentence position 4:
  Remainder: [4, 6)
  Next sentence: [6, 9)
```

### Component 3: ChunkIterator - The Main Chunking Algorithm

#### Initialization
```python
class ChunkIterator:
    def __init__(self, text, max_char_buffer, document=None):
        if isinstance(text, str):
            text = tokenizer.TokenizedText(text=text)
        
        self.tokenized_text = text
        self.max_char_buffer = max_char_buffer
        self.sentence_iter = SentenceIterator(self.tokenized_text)
        self.broken_sentence = False
        self.document = document or Document(text=text.text)
```

**What happens:**
- Converts string to `TokenizedText` if needed
- Tokenization happens via `tokenizer.tokenize()`
- Creates sentence iterator
- Tracks whether previous chunk broke a sentence

#### Chunking Strategy Decision Tree

```
                        ┌─────────────────────────┐
                        │   Get Next Sentence     │
                        └───────────┬─────────────┘
                                    │
                        ┌───────────▼───────────┐
                        │ Does single token     │
                        │ exceed max_buffer?    │
                        └─────┬────────────┬────┘
                              │ YES        │ NO
                              │            │
                ┌─────────────▼──┐    ┌────▼────────────────────┐
                │ Return single  │    │ Try to fit sentence     │
                │ token as chunk │    │ within max_buffer       │
                └────────────────┘    └────┬────────────────────┘
                                           │
                            ┌──────────────▼──────────────┐
                            │ Does sentence fit?          │
                            └───┬────────────────────┬────┘
                                │ YES                │ NO
                                │                    │
                ┌───────────────▼──┐    ┌────────────▼────────────┐
                │ Try to add more  │    │ Break at token boundary │
                │ sentences        │    │ (prefer newline)        │
                └───┬──────────────┘    └─────────────────────────┘
                    │
        ┌───────────▼───────────┐
        │ Fits with next        │
        │ sentence?             │
        └───┬───────────────┬───┘
            │ YES           │ NO
            │               │
    ┌───────▼───┐    ┌──────▼──────┐
    │ Add to    │    │ Return      │
    │ chunk     │    │ chunk       │
    └───────────┘    └─────────────┘
```

#### Detailed Algorithm: __next__()

##### Step 1: Get Next Sentence
```python
def __next__(self):
    sentence = next(self.sentence_iter)  # TokenInterval
```

**What happens:**
- Gets next sentence from sentence iterator
- May be full sentence or sentence fragment
- Raises `StopIteration` when document exhausted

##### Step 2: Handle Oversized Single Token
```python
    # Create chunk with just first token
    curr_chunk = create_token_interval(sentence.start_index, sentence.start_index + 1)
    
    if self._tokens_exceed_buffer(curr_chunk):
        # Single token exceeds buffer - must be its own chunk
        self.sentence_iter = SentenceIterator(self.tokenized_text, sentence.start_index + 1)
        self.broken_sentence = (curr_chunk.end_index < sentence.end_index)
        return TextChunk(token_interval=curr_chunk, document=self.document)
```

**`_tokens_exceed_buffer()`:**
```python
def _tokens_exceed_buffer(self, token_interval):
    char_interval = get_char_interval(self.tokenized_text, token_interval)
    return (char_interval.end_pos - char_interval.start_pos) > self.max_char_buffer
```

**What happens:**
- Tests if first token alone exceeds buffer
- Examples: very long URLs, chemical names, IDs
- If yes:
  - Makes single token the entire chunk
  - Resets sentence iterator to next token
  - Sets `broken_sentence=True`
  - Returns chunk immediately

**Example:**
```
Text: "This is antidisestablishmentarianism."
max_char_buffer: 20

Token "antidisestablishmentarianism" (28 chars) exceeds buffer
Chunks:
  1. "This is" (7 chars)
  2. "antidisestablishmentarianism" (28 chars) - oversized
  3. "." (1 char)
```

##### Step 3: Grow Chunk Within Sentence
```python
    # Try to fit as much of sentence as possible
    start_of_new_line = -1
    for token_index in range(curr_chunk.start_index, sentence.end_index):
        # Track newline positions
        if self.tokenized_text.tokens[token_index].first_token_after_newline:
            start_of_new_line = token_index
        
        # Test adding this token
        test_chunk = create_token_interval(curr_chunk.start_index, token_index + 1)
        
        if self._tokens_exceed_buffer(test_chunk):
            # Would exceed buffer
            # Prefer breaking at newline if available
            if start_of_new_line > 0 and start_of_new_line > curr_chunk.start_index:
                curr_chunk = create_token_interval(curr_chunk.start_index, start_of_new_line)
            # else: keep curr_chunk as is (up to token_index - 1)
            
            self.sentence_iter = SentenceIterator(self.tokenized_text, curr_chunk.end_index)
            self.broken_sentence = True
            return TextChunk(token_interval=curr_chunk, document=self.document)
        else:
            # Fits, extend chunk
            curr_chunk = test_chunk
```

**What happens:**
- Iterates through sentence tokens
- Tracks newline positions (for better breaks)
- Tests each token addition
- If adding token would exceed buffer:
  - Break at most recent newline (if available and after start)
  - Otherwise break at previous token
  - Reset iterator to resume from break point
  - Mark sentence as broken
  - Return chunk
- If loop completes, entire sentence fits

**Example with newline breaking:**
```
Text: "First line\nSecond line\nThird line"
max_char_buffer: 25

Sentence fits, but has newlines:
  After "First line\n" (11 chars) - newline at token 3
  After "Second line\n" (23 chars) - newline at token 6
  After "Third line" (33 chars) - exceeds!

Chunks:
  1. "First line\nSecond line\n" (23 chars) - breaks at newline 6
  2. "Third line" (10 chars)
```

##### Step 4: Pack Multiple Sentences
```python
    # Sentence fit - try to add more sentences
    if self.broken_sentence:
        self.broken_sentence = False
    else:
        for sentence in self.sentence_iter:
            test_chunk = create_token_interval(curr_chunk.start_index, sentence.end_index)
            
            if self._tokens_exceed_buffer(test_chunk):
                # Next sentence doesn't fit
                self.sentence_iter = SentenceIterator(
                    self.tokenized_text,
                    curr_chunk.end_index
                )
                return TextChunk(token_interval=curr_chunk, document=self.document)
            else:
                # Fits, add to chunk
                curr_chunk = test_chunk
    
    # Return final chunk (end of document or sentences)
    return TextChunk(token_interval=curr_chunk, document=self.document)
```

**What happens:**
- If previous chunk broke sentence, skip packing
- Otherwise, try to add complete sentences
- For each subsequent sentence:
  - Test if adding it would exceed buffer
  - If yes: stop, return current chunk
  - If no: extend chunk to include sentence
- Continue until buffer full or document end

**Example:**
```
Sentences:
  1. "Roses are red." (15 chars)
  2. "Violets are blue." (17 chars)
  3. "Flowers are nice." (17 chars)
  4. "And so are you." (15 chars)

max_char_buffer: 35

Chunks:
  1. "Roses are red. Violets are blue." (32 chars) - 2 sentences
  2. "Flowers are nice." (17 chars)
  3. "And so are you." (15 chars)

max_char_buffer: 60

Chunks:
  1. "Roses are red. Violets are blue. Flowers are nice." (50 chars) - 3 sentences
  2. "And so are you." (15 chars)
```

### Component 4: Document Chunking

#### _document_chunk_iterator()
```python
def _document_chunk_iterator(documents, max_char_buffer, restrict_repeats=True):
    visited_ids = set()
    
    for document in documents:
        tokenized_text = document.tokenized_text
        document_id = document.document_id
        
        # Check for duplicate IDs
        if restrict_repeats and document_id in visited_ids:
            raise DocumentRepeatError(f"Document id {document_id} is already visited.")
        
        # Create chunk iterator for this document
        chunk_iter = ChunkIterator(
            text=tokenized_text,
            max_char_buffer=max_char_buffer,
            document=document,
        )
        visited_ids.add(document_id)
        
        # Yield all chunks from this document
        yield from chunk_iter
```

**What happens:**
- Wraps multiple documents
- Ensures unique document IDs
- Creates `ChunkIterator` for each document
- Yields chunks sequentially from all documents
- Maintains document metadata in chunks

#### make_batches_of_textchunk()
```python
def make_batches_of_textchunk(chunk_iter, batch_length):
    for batch in more_itertools.batched(chunk_iter, batch_length):
        yield list(batch)
```

**What happens:**
- Groups chunks into batches
- Uses `more_itertools.batched()` for efficient batching
- Each batch: up to `batch_length` chunks
- Chunks can come from different documents
- Enables parallel LLM inference

### Chunking Properties

**Guarantees:**
1. No chunk exceeds `max_char_buffer` (except single oversized tokens)
2. Sentence boundaries respected when possible
3. Newline boundaries preferred for mid-sentence breaks
4. Multiple sentences packed when they fit
5. Document metadata preserved in chunks

**Trade-offs:**
- Smaller chunks = more API calls, better parallelization
- Larger chunks = fewer API calls, more context per call
- Breaking sentences = potential information loss at boundaries
- Overlapping not implemented (future enhancement)

---


## Prompting System - Deep Dive

### Location
`langextract/prompting.py` - `QAPromptGenerator`, `PromptTemplateStructured`

### Purpose
Transform task descriptions and examples into structured prompts that guide LLM extraction behavior.

### Component 1: PromptTemplateStructured

```python
@dataclasses.dataclass
class PromptTemplateStructured:
    description: str  # Task instructions
    examples: list[ExampleData] = field(default_factory=list)
```

**What it stores:**
- `description`: Natural language instructions for the LLM
- `examples`: List of `ExampleData` objects (text + extractions)

**Creation:**
```python
prompt_template = PromptTemplateStructured(
    description="Extract medical conditions from clinical notes. Output as JSON.",
    examples=[
        ExampleData(
            text="Patient has diabetes.",
            extractions=[
                Extraction(extraction_class="condition", extraction_text="diabetes")
            ]
        )
    ]
)
```

### Component 2: QAPromptGenerator

#### Purpose
Generate question-answer style prompts with few-shot examples.

#### Initialization
```python
@dataclasses.dataclass
class QAPromptGenerator:
    template: PromptTemplateStructured
    format_handler: FormatHandler
    examples_heading: str = "Examples"
    question_prefix: str = "Q: "
    answer_prefix: str = "A: "
```

**What it stores:**
- `template`: The structured prompt template
- `format_handler`: Handles JSON/YAML formatting
- Customizable prefixes for Q&A format

#### Mechanism: render()

```python
def render(self, question: str, additional_context: str | None = None) -> str:
    prompt_lines = [f"{self.template.description}\n"]
    
    if additional_context:
        prompt_lines.append(f"{additional_context}\n")
    
    if self.template.examples:
        prompt_lines.append(self.examples_heading)
        for ex in self.template.examples:
            prompt_lines.append(self.format_example_as_text(ex))
    
    prompt_lines.append(f"{self.question_prefix}{question}")
    prompt_lines.append(self.answer_prefix)
    
    return "\n".join(prompt_lines)
```

**Structure:**
```
[Description]

[Additional Context (optional)]

Examples
Q: [Example 1 text]
A: [Example 1 extractions in JSON/YAML]

Q: [Example 2 text]
A: [Example 2 extractions in JSON/YAML]

Q: [Input text to process]
A: 
```

**What happens:**
1. Starts with task description
2. Adds additional context if provided
3. Adds "Examples" heading
4. Formats each example using format handler
5. Adds input text as question
6. Ends with answer prefix (LLM continues from here)

#### Mechanism: format_example_as_text()

```python
def format_example_as_text(self, example: ExampleData) -> str:
    question = example.text
    answer = self.format_handler.format_extraction_example(example.extractions)
    
    return "\n".join([
        f"{self.question_prefix}{question}",
        f"{self.answer_prefix}{answer}\n",
    ])
```

**What happens:**
- Gets example text
- Formats extractions via format handler
- Combines into Q&A format
- Adds blank line after each example

**Format handler formatting (see FormatHandler section):**
- Converts `Extraction` objects to dictionaries
- Adds attributes with suffix
- Wraps in container key if configured
- Serializes to JSON or YAML
- Adds fences if configured

**Example output (JSON with fences):**
```
Q: Patient has diabetes and hypertension.
A: ```json
{
  "extractions": [
    {
      "condition": "diabetes",
      "condition_attributes": {}
    },
    {
      "condition": "hypertension",
      "condition_attributes": {}
    }
  ]
}
```

### Prompt Engineering Principles

#### 1. Clear Task Description
```python
description = """Extract medical conditions from clinical notes.
- Include diagnosed conditions
- Include suspected conditions marked with "?"
- Exclude historical conditions marked as "no longer present"
Output as JSON with extraction class "condition"."""
```

**Impact:**
- Explicit instructions reduce ambiguity
- Examples reinforce the rules
- Clear output format expectation

#### 2. Consistent Example Format
```python
format_handler = FormatHandler(
    format_type=FormatType.JSON,
    use_wrapper=True,
    wrapper_key="extractions",
    use_fences=True,
)
```

**Impact:**
- All examples formatted identically
- LLM learns consistent structure
- Parser knows what to expect

#### 3. Representative Examples
```python
examples = [
    # Simple case
    ExampleData(text="Patient has diabetes.", extractions=[...]),
    
    # Multiple extractions
    ExampleData(text="Diagnosed with hypertension and asthma.", extractions=[...]),
    
    # With attributes
    ExampleData(text="Type 2 diabetes diagnosed in 2020.", extractions=[
        Extraction(
            extraction_class="condition",
            extraction_text="diabetes",
            attributes={"type": "type 2", "year": "2020"}
        )
    ]),
]
```

**Impact:**
- Covers various input patterns
- Demonstrates attribute usage
- Shows how to handle multiple entities

#### 4. Additional Context Usage
```python
additional_context = "Patient is 65-year-old female with history of cardiovascular disease."
prompt = generator.render(chunk_text, additional_context=additional_context)
```

**Impact:**
- Provides document-level context
- Helps with ambiguous cases
- Can include terminology definitions

### Prompt Optimization Strategies

#### Strategy 1: Few-Shot Count
- **Too few (0-1)**: LLM may not understand pattern
- **Optimal (2-5)**: Good pattern recognition, fits context
- **Too many (>10)**: Wastes context window, slower inference

#### Strategy 2: Example Diversity
- Include simple and complex cases
- Show various attribute combinations
- Demonstrate edge cases

#### Strategy 3: Format Consistency
- Same format_type across all examples
- Consistent attribute naming
- Same wrapper structure

---

## Provider System - Deep Dive

### Location
`langextract/providers/` - Router, factory, individual providers

### Architecture

```
┌────────────────────────────────────────────────────────┐
│                    FACTORY LAYER                       │
│              (factory.py: create_model)                │
│  • Environment variable resolution                     │
│  • Provider instantiation                              │
│  • Schema application                                  │
└───────────────────┬────────────────────────────────────┘
                    │
        ┌───────────▼────────────┐
        │    ROUTER LAYER        │
        │   (router.py)          │
        │  • Pattern matching    │
        │  • Provider selection  │
        └───────────┬────────────┘
                    │
    ┌───────────────┼───────────────┐
    ▼               ▼               ▼
┌─────────┐   ┌──────────┐   ┌──────────┐
│ Gemini  │   │ OpenAI   │   │ Ollama   │
│Provider │   │ Provider │   │ Provider │
└─────────┘   └──────────┘   └──────────┘
    │               │               │
    ▼               ▼               ▼
┌─────────┐   ┌──────────┐   ┌──────────┐
│ Gemini  │   │ OpenAI   │   │ Ollama   │
│  API    │   │   API    │   │   API    │
└─────────┘   └──────────┘   └──────────┘
```

### Component 1: ModelConfig

```python
@dataclasses.dataclass(frozen=True)
class ModelConfig:
    model_id: str | None = None  # e.g., "gemini-2.5-flash"
    provider: str | None = None  # e.g., "GeminiLanguageModel"
    provider_kwargs: dict[str, Any] = field(default_factory=dict)
```

**Purpose:**
- Encapsulates model configuration
- Frozen dataclass (immutable)
- Explicit or inferred provider

**Usage:**
```python
# By model ID (provider inferred)
config = ModelConfig(
    model_id="gemini-2.5-flash",
    provider_kwargs={"temperature": 0.7}
)

# By explicit provider
config = ModelConfig(
    provider="GeminiLanguageModel",
    provider_kwargs={"api_key": "...", "model_id": "gemini-2.5-flash"}
)
```

### Component 2: Factory - create_model()

#### Step 1: Load Providers
```python
def create_model(config, examples=None, use_schema_constraints=False, fence_output=None):
    # Load built-in and plugin providers
    providers.load_builtins_once()
    providers.load_plugins_once()
```

**What happens:**
- Loads built-in providers (Gemini, OpenAI, Ollama)
- Discovers and loads plugin providers
- Uses global registry to avoid reloading

#### Step 2: Resolve Provider
```python
    # Resolve provider class
    if config.provider:
        provider_class = router.resolve_provider(config.provider)
    else:
        provider_class = router.resolve(config.model_id)
```

**Router resolution (detailed below):**
- Matches `model_id` against registered patterns
- Returns provider class
- Raises error if no match

#### Step 3: Environment Defaults
```python
    # Add environment-based defaults
    kwargs = _kwargs_with_environment_defaults(config.model_id, config.provider_kwargs)
```

**`_kwargs_with_environment_defaults()`:**
```python
def _kwargs_with_environment_defaults(model_id, kwargs):
    resolved = dict(kwargs)
    
    # API key resolution
    if "api_key" not in resolved:
        model_lower = model_id.lower()
        
        if "gemini" in model_lower:
            for env_var in ["GEMINI_API_KEY", "LANGEXTRACT_API_KEY"]:
                api_key = os.getenv(env_var)
                if api_key:
                    resolved["api_key"] = api_key
                    break
        
        elif "gpt" in model_lower:
            for env_var in ["OPENAI_API_KEY", "LANGEXTRACT_API_KEY"]:
                api_key = os.getenv(env_var)
                if api_key:
                    resolved["api_key"] = api_key
                    break
    
    # Ollama URL resolution
    if "ollama" in model_id.lower() and "base_url" not in resolved:
        resolved["base_url"] = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    return resolved
```

**What happens:**
- Checks for API key in kwargs
- If missing, tries provider-specific env vars
- Falls back to generic `LANGEXTRACT_API_KEY`
- Sets Ollama base URL from env or default
- Doesn't override explicit kwargs

#### Step 4: Schema Generation (if enabled)
```python
    # Generate schema from examples
    if use_schema_constraints and examples:
        schema_class = provider_class.get_schema_class()
        if schema_class:
            schema_instance = schema_class.from_examples(
                examples,
                format_type=kwargs.get("format_type", FormatType.JSON)
            )
```

**What happens:**
- Asks provider for schema class
- Not all providers support schemas (returns None)
- Creates schema instance from examples
- Schema analyzes extraction structure

**Schema generation (see Schema System section):**
- Extracts extraction classes from examples
- Determines attribute structure
- Creates JSON Schema or provider-specific format

#### Step 5: Provider Instantiation
```python
    # Instantiate provider
    try:
        model = provider_class(model_id=config.model_id, **kwargs)
    except Exception as e:
        raise InferenceConfigError(f"Failed to create provider: {e}") from e
```

**What happens:**
- Calls provider's `__init__`
- Passes model_id and all kwargs
- Provider initializes API client
- Stores configuration

#### Step 6: Schema Application
```python
    # Apply schema to provider
    if schema_instance:
        model.apply_schema(schema_instance)
```

**What happens:**
- Calls provider's `apply_schema()` method
- Provider stores schema for inference
- Provider may adjust inference parameters

#### Step 7: Fence Output Configuration
```python
    # Configure fence output
    if fence_output is not None:
        model.set_fence_output(fence_output)
```

**What happens:**
- Explicit override if provided
- Otherwise computed from schema
- Determines if LLM should generate fences

### Component 3: Router - Provider Resolution

#### Registry Structure
```python
# Global registry
_PROVIDER_REGISTRY = {
    "pattern": {
        r"^gemini": GeminiLanguageModel,
        r"^gpt": OpenAILanguageModel,
        r"^ollama": OllamaLanguageModel,
    },
    "name": {
        "GeminiLanguageModel": GeminiLanguageModel,
        "OpenAILanguageModel": OpenAILanguageModel,
        "OllamaLanguageModel": OllamaLanguageModel,
    }
}
```

**What it stores:**
- `pattern`: Regex → provider class mappings
- `name`: Name → provider class mappings

#### Resolution Mechanism: resolve()
```python
def resolve(model_id: str) -> Type[BaseLanguageModel]:
    for pattern, provider_class in _PROVIDER_REGISTRY["pattern"].items():
        if re.match(pattern, model_id, re.IGNORECASE):
            return provider_class
    
    raise ValueError(f"No provider registered for model_id: {model_id}")
```

**What happens:**
- Iterates through registered patterns
- Tests each regex against model_id
- Returns first matching provider class
- Case-insensitive matching
- Raises error if no match

**Example matches:**
- `"gemini-2.5-flash"` → `GeminiLanguageModel`
- `"gpt-4o"` → `OpenAILanguageModel`
- `"ollama:llama3"` → `OllamaLanguageModel`

#### Resolution by Name: resolve_provider()
```python
def resolve_provider(provider_name: str) -> Type[BaseLanguageModel]:
    provider_class = _PROVIDER_REGISTRY["name"].get(provider_name)
    if provider_class:
        return provider_class
    
    raise ValueError(f"Provider not found: {provider_name}")
```

**What happens:**
- Direct lookup by class name
- Useful for explicit provider specification
- Bypasses pattern matching

### Component 4: BaseLanguageModel Interface

```python
class BaseLanguageModel(ABC):
    def __init__(self, constraint=None, **kwargs):
        self._constraint = constraint or Constraint()
        self._schema = None
        self._fence_output_override = None
        self._extra_kwargs = kwargs.copy()
    
    @abstractmethod
    def infer(self, batch_prompts, **kwargs):
        """Run inference on batch of prompts.
        
        Args:
            batch_prompts: List of prompt strings
            **kwargs: Additional inference parameters
        
        Yields:
            List[ScoredOutput] for each prompt
        """
        pass
    
    @classmethod
    def get_schema_class(cls):
        """Return schema class this provider supports."""
        return None
    
    def apply_schema(self, schema_instance):
        """Apply schema instance to provider."""
        self._schema = schema_instance
    
    @property
    def schema(self):
        """Current schema instance."""
        return self._schema
    
    @property
    def requires_fence_output(self):
        """Whether model requires fenced output."""
        if self._fence_output_override is not None:
            return self._fence_output_override
        
        if self._schema is None:
            return True  # Default to fences
        
        return not self._schema.requires_raw_output
```

**Contract:**
- `infer()`: Must be implemented by all providers
- `get_schema_class()`: Optional, return None if no schema support
- `apply_schema()`: Store schema for inference
- `requires_fence_output`: Computed from schema or override

### Component 5: Example Provider - GeminiLanguageModel

#### Initialization
```python
class GeminiLanguageModel(BaseLanguageModel):
    def __init__(self, model_id="gemini-2.5-flash", api_key=None,
                 temperature=None, max_workers=10, **kwargs):
        super().__init__(**kwargs)
        
        self.model_id = model_id
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY required")
        
        self.temperature = temperature
        self.max_workers = max_workers
        
        # Initialize Gemini client
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        self.client = genai.GenerativeModel(model_id)
```

**What happens:**
- Stores configuration
- Resolves API key
- Imports Gemini SDK
- Configures SDK with API key
- Creates model client

#### Schema Support
```python
    @classmethod
    def get_schema_class(cls):
        from langextract.providers.schemas.gemini import GeminiSchema
        return GeminiSchema
    
    def apply_schema(self, schema_instance):
        super().apply_schema(schema_instance)
        if schema_instance:
            config = schema_instance.to_provider_config()
            self.response_schema = config.get("response_schema")
            self.structured_output = config.get("structured_output", False)
        else:
            self.response_schema = None
            self.structured_output = False
```

**What happens:**
- Returns `GeminiSchema` class
- Stores schema configuration
- Extracts Gemini-specific response schema
- Enables structured output mode

#### Inference
```python
    def infer(self, batch_prompts, **kwargs):
        # Parallel processing with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for prompt in batch_prompts:
                future = executor.submit(self._infer_single, prompt, **kwargs)
                futures.append(future)
            
            for future in futures:
                result = future.result()
                yield result
    
    def _infer_single(self, prompt, **kwargs):
        generation_config = {}
        if self.temperature is not None:
            generation_config["temperature"] = self.temperature
        
        if self.response_schema:
            generation_config["response_schema"] = self.response_schema
        
        response = self.client.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        return [ScoredOutput(output=response.text, score=1.0)]
```

**What happens:**
- Uses thread pool for parallel requests
- Submits each prompt as separate task
- Configures temperature and schema
- Calls Gemini API
- Wraps response in `ScoredOutput`
- Yields results in submission order

### Component 6: Plugin System

#### Discovery
```python
def load_plugins_once():
    global _plugins_loaded
    if _plugins_loaded:
        return
    
    # Discover entry points
    for entry_point in metadata.entry_points().select(group="langextract.providers"):
        try:
            provider_class = entry_point.load()
            register_provider(
                provider_class=provider_class,
                patterns=getattr(provider_class, "PATTERNS", []),
            )
        except Exception as e:
            logging.warning(f"Failed to load plugin {entry_point.name}: {e}")
    
    _plugins_loaded = True
```

**What happens:**
- Uses Python entry points
- Discovers packages with `langextract.providers` entry point
- Loads provider classes
- Registers patterns
- Logs errors but continues

#### Plugin Structure
```python
# In plugin package: setup.py or pyproject.toml
entry_points={
    "langextract.providers": [
        "myprovider = langextract_myprovider:MyProvider"
    ]
}

# In langextract_myprovider/__init__.py
class MyProvider(BaseLanguageModel):
    PATTERNS = [r"^myprovider:"]
    
    def infer(self, batch_prompts, **kwargs):
        # Implementation
        pass
```

**What happens:**
- Package declares entry point
- Entry point points to provider class
- Provider class has PATTERNS attribute
- Router registers patterns automatically

---

## Format Handler - Deep Dive

### Location
`langextract/core/format_handler.py` - `FormatHandler` class

### Purpose
Centralize all format-specific logic for JSON/YAML handling in prompts and parsing.

### Component 1: Configuration

```python
class FormatHandler:
    def __init__(
        self,
        format_type=FormatType.JSON,
        use_wrapper=True,
        wrapper_key=None,
        use_fences=True,
        attribute_suffix="_attributes",
        strict_fences=False,
        allow_top_level_list=True,
    ):
        self.format_type = format_type
        self.use_wrapper = use_wrapper
        self.wrapper_key = wrapper_key if use_wrapper else None
        self.use_fences = use_fences
        self.attribute_suffix = attribute_suffix
        self.strict_fences = strict_fences
        self.allow_top_level_list = allow_top_level_list
```

**Parameters explained:**
- `format_type`: JSON or YAML
- `use_wrapper`: Wrap in `{"extractions": [...]}`  vs `[...]`
- `wrapper_key`: Custom key (default "extractions")
- `use_fences`: Generate/expect ` ```json` fences
- `attribute_suffix`: Suffix for attributes (e.g., "_attributes")
- `strict_fences`: Strict fence parsing vs lenient
- `allow_top_level_list`: Allow `[...]` without wrapper

### Component 2: Formatting for Prompts

#### Mechanism: format_extraction_example()
```python
def format_extraction_example(self, extractions: list[Extraction]) -> str:
    # Convert extractions to dictionaries
    items = [
        {
            ext.extraction_class: ext.extraction_text,
            f"{ext.extraction_class}{self.attribute_suffix}": ext.attributes or {},
        }
        for ext in extractions
    ]
    
    # Wrap if configured
    if self.use_wrapper and self.wrapper_key:
        payload = {self.wrapper_key: items}
    else:
        payload = items
    
    # Serialize to JSON or YAML
    if self.format_type == FormatType.YAML:
        formatted = yaml.safe_dump(payload, default_flow_style=False, sort_keys=False)
    else:
        formatted = json.dumps(payload, indent=2, ensure_ascii=False)
    
    # Add fences if configured
    return self._add_fences(formatted) if self.use_fences else formatted
```

**Step-by-step transformation:**

```python
# Input: Extraction objects
extractions = [
    Extraction(
        extraction_class="condition",
        extraction_text="diabetes",
        attributes={"type": "type2"}
    )
]

# Step 1: Convert to dicts
items = [
    {
        "condition": "diabetes",
        "condition_attributes": {"type": "type2"}
    }
]

# Step 2: Wrap (if use_wrapper=True)
payload = {
    "extractions": [
        {
            "condition": "diabetes",
            "condition_attributes": {"type": "type2"}
        }
    ]
}

# Step 3: Serialize (JSON)
formatted = """{
  "extractions": [
    {
      "condition": "diabetes",
      "condition_attributes": {
        "type": "type2"
      }
    }
  ]
}"""

# Step 4: Add fences (if use_fences=True)
result = """```json
{
  "extractions": [
    {
      "condition": "diabetes",
      "condition_attributes": {
        "type": "type2"
      }
    }
  ]
}
```"""
```

#### Mechanism: _add_fences()
```python
def _add_fences(self, content: str) -> str:
    lang = "json" if self.format_type == FormatType.JSON else "yaml"
    return f"```{lang}\n{content}\n```"
```

**What happens:**
- Determines language tag
- Wraps content in code fences
- Adds newlines for proper formatting

### Component 3: Parsing LLM Output

#### Mechanism: parse_output()
```python
def parse_output(self, text: str, *, strict: bool = None) -> Sequence[Mapping[str, ExtractionValueType]]:
    # Step 1: Extract from fences if present
    text = text.strip()
    if self.use_fences or "```" in text:
        text = self._extract_from_fences(text, strict=strict or self.strict_fences)
    
    # Step 2: Parse JSON or YAML
    try:
        if self.format_type == FormatType.JSON:
            parsed = json.loads(text)
        else:
            parsed = yaml.safe_load(text)
    except Exception as e:
        raise FormatError(f"Failed to parse {self.format_type.value}: {e}") from e
    
    # Step 3: Unwrap if needed
    if isinstance(parsed, dict) and self.wrapper_key in parsed:
        items = parsed[self.wrapper_key]
    elif isinstance(parsed, list):
        if not self.allow_top_level_list and (strict or self.strict_fences):
            raise FormatError("Top-level list not allowed in strict mode")
        items = parsed
    else:
        raise FormatError(f"Expected list or dict with '{self.wrapper_key}' key")
    
    # Step 4: Validate structure
    if not isinstance(items, list):
        raise FormatError("Extractions must be a list")
    
    for item in items:
        if not isinstance(item, dict):
            raise FormatError("Each extraction must be a dictionary")
    
    return items
```

**Step-by-step parsing:**

```python
# Input: LLM output
text = """```json
{
  "extractions": [
    {
      "condition": "diabetes",
      "condition_attributes": {"type": "type2"}
    }
  ]
}
```"""

# Step 1: Extract from fences
text = """{
  "extractions": [
    {
      "condition": "diabetes",
      "condition_attributes": {"type": "type2"}
    }
  ]
}"""

# Step 2: Parse JSON
parsed = {
    "extractions": [
        {
            "condition": "diabetes",
            "condition_attributes": {"type": "type2"}
        }
    ]
}

# Step 3: Unwrap
items = [
    {
        "condition": "diabetes",
        "condition_attributes": {"type": "type2"}
    }
]

# Step 4: Validate (passes)
return items
```

#### Mechanism: _extract_from_fences()
```python
def _extract_from_fences(self, text: str, strict: bool = False) -> str:
    # Regex pattern for code fences
    pattern = r"```(?P<lang>[A-Za-z0-9_+-]+)?(?:\s*\n)?(?P<body>[\s\S]*?)```"
    
    matches = list(re.finditer(pattern, text, re.MULTILINE))
    
    if not matches:
        if strict:
            raise FormatError("No code fences found in strict mode")
        return text  # Return as-is in lenient mode
    
    # Find matching language tag
    target_lang = "json" if self.format_type == FormatType.JSON else "yaml"
    
    for match in matches:
        lang = match.group("lang")
        if lang and lang.lower() in [target_lang, "yml" if target_lang == "yaml" else ""]:
            return match.group("body").strip()
    
    # No matching language, use first fence
    return matches[0].group("body").strip()
```

**What happens:**
- Uses regex to find all code fences
- Extracts language tag and body
- Prefers matching language tag
- Falls back to first fence
- Lenient mode returns text as-is if no fences

**Regex breakdown:**
```
```(?P<lang>[A-Za-z0-9_+-]+)?   # Optional language tag
(?:\s*\n)?                        # Optional whitespace/newline
(?P<body>[\s\S]*?)               # Body (non-greedy)
```                               # Closing fence
```

**Example matches:**
- ` ```json\n{...}\n``` ` → body=`{...}`
- ` ```\n{...}\n``` ` → body=`{...}` (no lang)
- ` ```yaml\n...\n``` ` → body=`...`

### Component 4: from_resolver_params()

#### Purpose
Create FormatHandler from legacy resolver_params.

```python
@classmethod
def from_resolver_params(
    cls,
    resolver_params=None,
    base_format_type=FormatType.JSON,
    base_use_fences=True,
    base_attribute_suffix="_attributes",
    base_use_wrapper=True,
    base_wrapper_key="extractions",
):
    params = resolver_params or {}
    
    # Extract format-specific params
    format_type = params.pop("format_type", base_format_type)
    use_fences = params.pop("fence_output", base_use_fences)
    attribute_suffix = params.pop("attribute_suffix", base_attribute_suffix)
    use_wrapper = params.pop("require_extractions_key", base_use_wrapper)
    strict_fences = params.pop("strict_fences", False)
    
    # Create handler
    handler = cls(
        format_type=format_type,
        use_wrapper=use_wrapper,
        wrapper_key=base_wrapper_key if use_wrapper else None,
        use_fences=use_fences,
        attribute_suffix=attribute_suffix,
        strict_fences=strict_fences,
    )
    
    # Return handler and remaining params
    return handler, params
```

**What happens:**
- Extracts format-related parameters
- Uses defaults for missing parameters
- Creates FormatHandler instance
- Returns handler and remaining parameters
- Remaining parameters used for alignment config

---


## Schema System - Deep Dive

### Location
`langextract/core/schema.py` and `langextract/providers/schemas/`

### Purpose
Enable structured output constraints for LLMs that support them, ensuring outputs match expected format without manual parsing.

### Component 1: BaseSchema Interface

```python
class BaseSchema(ABC):
    @property
    @abstractmethod
    def schema_dict(self) -> dict:
        """Return the schema as a dictionary."""
        pass
    
    @classmethod
    @abstractmethod
    def from_examples(cls, examples_data, attribute_suffix="_attributes"):
        """Build schema from example extractions."""
        pass
    
    def to_provider_config(self) -> dict:
        """Convert schema to provider-specific configuration."""
        return {"response_schema": self.schema_dict}
    
    @property
    def requires_raw_output(self) -> bool:
        """Whether this schema requires raw output (no fences)."""
        return True  # Most schemas require raw JSON/YAML
    
    def validate_format(self, format_handler):
        """Validate format handler compatibility."""
        pass
```

### Component 2: Schema Generation from Examples

#### Mechanism: from_examples()
```python
@classmethod
def from_examples(cls, examples_data, attribute_suffix="_attributes"):
    # Analyze all extractions across examples
    extraction_types = {}
    
    for example in examples_data:
        for extraction in example.extractions:
            class_name = extraction.extraction_class
            
            if class_name not in extraction_types:
                extraction_types[class_name] = set()
            
            # Collect attribute keys
            if extraction.attributes:
                extraction_types[class_name].update(extraction.attributes.keys())
    
    # Build JSON Schema
    schema_dict = {
        "type": "object",
        "properties": {
            "extractions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        },
        "required": ["extractions"]
    }
    
    # Add properties for each extraction type
    properties = schema_dict["properties"]["extractions"]["items"]["properties"]
    
    for class_name, attribute_keys in extraction_types.items():
        properties[class_name] = {"type": "string"}
        
        if attribute_keys:
            attr_key = f"{class_name}{attribute_suffix}"
            properties[attr_key] = {
                "type": "object",
                "properties": {key: {"type": "string"} for key in attribute_keys}
            }
    
    return cls(schema_dict)
```

**What happens:**
- Scans all examples to find extraction classes
- Collects attribute keys for each class
- Builds JSON Schema structure
- Specifies required fields
- Defines property types

### Component 3: GeminiSchema

```python
class GeminiSchema(BaseSchema):
    def __init__(self, schema_dict: dict):
        self._schema_dict = schema_dict
    
    @property
    def schema_dict(self) -> dict:
        return self._schema_dict
    
    def to_provider_config(self) -> dict:
        # Convert to Gemini's response_schema format
        return {
            "response_schema": self._schema_dict,
            "structured_output": True
        }
    
    @property
    def requires_raw_output(self) -> bool:
        return True  # Gemini with schema outputs raw JSON
```

### Schema Benefits

1. **Guaranteed Valid Output**: LLM cannot generate malformed JSON
2. **Reduced Parsing Errors**: No need to handle fence extraction
3. **Faster Inference**: Constrained generation can be faster
4. **Type Safety**: Schema enforces types at generation time

---

## Data Structures and Transformations

### Data Flow Through System

```
1. INPUT: String
   "Patient has diabetes."

2. DOCUMENT: data.Document
   {
       text: "Patient has diabetes.",
       document_id: "doc_001",
       tokenized_text: TokenizedText(...)
   }

3. CHUNKS: TextChunk
   {
       token_interval: [0, 4),
       char_interval: [0, 23),
       chunk_text: "Patient has diabetes.",
       document: Document(...)
   }

4. PROMPTS: String
   "Extract conditions...\n\nExamples\n...\n\nQ: Patient has diabetes.\nA: "

5. LLM OUTPUT: String
   '```json\n{"extractions": [{"condition": "diabetes"}]}\n```'

6. PARSED: List[Dict]
   [{"condition": "diabetes"}]

7. EXTRACTIONS: List[Extraction]
   [
       Extraction(
           extraction_class="condition",
           extraction_text="diabetes",
           extraction_index=1
       )
   ]

8. ALIGNED: List[Extraction]
   [
       Extraction(
           extraction_class="condition",
           extraction_text="diabetes",
           token_interval=[2, 3),
           char_interval=[12, 20),
           alignment_status=MATCH_EXACT
       )
   ]

9. OUTPUT: AnnotatedDocument
   {
       document_id: "doc_001",
       text: "Patient has diabetes.",
       extractions: [...]
   }
```

### Key Data Structures

#### TokenInterval
```python
@dataclass
class TokenInterval:
    start_index: int  # Inclusive
    end_index: int    # Exclusive
```
**Purpose**: Mark token positions in document

#### CharInterval
```python
@dataclass
class CharInterval:
    start_pos: int  # Inclusive
    end_pos: int    # Exclusive
```
**Purpose**: Mark character positions for highlighting

#### Extraction
```python
@dataclass
class Extraction:
    extraction_class: str
    extraction_text: str
    extraction_index: int = 0
    group_index: int = 0
    attributes: dict | None = None
    token_interval: TokenInterval | None = None
    char_interval: CharInterval | None = None
    alignment_status: AlignmentStatus | None = None
```
**Purpose**: Represent a single extracted entity with position and metadata

#### ScoredOutput
```python
@dataclass
class ScoredOutput:
    output: str    # LLM's text response
    score: float   # Confidence (usually 1.0)
```
**Purpose**: Wrap LLM output with confidence score

---

## Component Interaction Patterns

### Pattern 1: Dependency Injection

**Annotator receives configured components:**
```python
annotator = Annotator(
    language_model=model,      # Pre-configured provider
    prompt_template=template,  # Pre-loaded examples
    format_handler=handler,    # Pre-configured format
)
```

**Benefits:**
- Testability: Mock any component
- Flexibility: Swap implementations
- Clarity: Explicit dependencies

### Pattern 2: Iterator Chaining

**Documents → Chunks → Batches → Results:**
```python
documents → ChunkIterator → make_batches → process → AnnotatedDocuments
```

**Benefits:**
- Memory efficient: No full document in memory
- Lazy evaluation: Process on demand
- Composable: Chain transformations

### Pattern 3: Factory Pattern

**Model creation abstraction:**
```python
config = ModelConfig(model_id="gemini-2.5-flash")
model = factory.create_model(config)  # Returns GeminiLanguageModel
```

**Benefits:**
- Encapsulation: Hide provider details
- Extensibility: Add providers without changing client code
- Configuration: Centralized model setup

### Pattern 4: Strategy Pattern

**Alignment strategies:**
```python
# Exact matching via difflib
aligned = aligner.align_extractions(..., enable_fuzzy_alignment=False)

# Fuzzy matching fallback
aligned = aligner.align_extractions(..., enable_fuzzy_alignment=True)
```

**Benefits:**
- Flexibility: Choose strategy at runtime
- Extensibility: Add new strategies
- Testability: Test strategies independently

### Pattern 5: Template Method

**Annotation flow:**
```python
class Annotator:
    def annotate_documents(self, ...):
        if extraction_passes == 1:
            return self._annotate_documents_single_pass(...)
        else:
            return self._annotate_documents_sequential_passes(...)
```

**Benefits:**
- Code reuse: Shared single-pass logic
- Extensibility: Override for custom behavior
- Clarity: Separate concerns

---

## Performance Considerations

### Bottlenecks

1. **LLM Inference**: 80-90% of total time
   - Network latency to API
   - Model generation time
   - Rate limiting

2. **Fuzzy Alignment**: 5-10% of time (when many unaligned)
   - O(n * w * e) complexity
   - Only runs on unaligned extractions
   - Pre-check optimization reduces impact

3. **Parsing**: <5% of time
   - JSON/YAML parsing is fast
   - Fence extraction is negligible

### Optimization Strategies

#### 1. Parallel Processing
```python
# Batch processing with ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(infer, prompt) for prompt in batch]
    results = [f.result() for f in futures]
```

**Impact**: Near-linear speedup with workers

#### 2. Caching
```python
@functools.lru_cache(maxsize=10000)
def _normalize_token(token: str) -> str:
    # Expensive stemming cached
    ...
```

**Impact**: 10-20% faster alignment

#### 3. Lazy Evaluation
```python
@property
def chunk_text(self) -> str:
    if self._chunk_text is None:
        self._chunk_text = get_token_interval_text(...)
    return self._chunk_text
```

**Impact**: Reduced memory usage, faster iteration

#### 4. Early Termination
```python
# Fast pre-check in fuzzy alignment
if (extraction_counts & window_counts).total() < min_overlap:
    continue  # Skip expensive matching
```

**Impact**: 90% of windows skipped

---

## Error Handling Strategies

### 1. Validation Errors (Early Detection)
```python
if not examples:
    raise ValueError("Examples are required")
```

**When**: Input validation
**Recovery**: User must provide valid inputs

### 2. Parsing Errors (Recoverable)
```python
try:
    extraction_data = format_handler.parse_output(text)
except FormatError as e:
    if suppress_parse_errors:
        logging.exception("Parse failed: %s", e)
        return []  # Continue with empty
    raise
```

**When**: LLM output parsing
**Recovery**: Log and continue or raise

### 3. Alignment Failures (Graceful Degradation)
```python
aligned = fuzzy_align_extraction(...)
if aligned:
    extraction.alignment_status = MATCH_FUZZY
else:
    extraction.alignment_status = None  # Unaligned but included
```

**When**: Cannot find text in source
**Recovery**: Include extraction without position

### 4. API Errors (Propagate)
```python
try:
    response = self.client.generate_content(prompt)
except Exception as e:
    raise InferenceError(f"API call failed: {e}") from e
```

**When**: LLM API calls
**Recovery**: Propagate to caller for retry logic

---

## Summary

LangExtract's architecture is built on several key principles:

1. **Modularity**: Clear separation of concerns (chunking, prompting, inference, resolution)
2. **Extensibility**: Plugin system for providers, strategy pattern for algorithms
3. **Robustness**: Multiple alignment strategies, error handling at each layer
4. **Efficiency**: Parallel processing, lazy evaluation, caching
5. **Flexibility**: Configurable formats, schemas, alignment parameters

**The resolver is the most complex component**, implementing:
- Two-phase alignment (exact then fuzzy)
- Token-based matching with difflib
- Sliding window search with optimizations
- Character interval computation
- Multiple alignment status levels

**The system achieves high accuracy** through:
- Few-shot learning with examples
- Schema-constrained generation (when supported)
- Fuzzy alignment for robustness
- Multi-pass extraction for recall

**The architecture scales** via:
- Document chunking for large texts
- Batch processing for parallelization
- Stateless components for thread safety
- Iterator-based processing for memory efficiency

---

## Related Documentation

- [architecture.md](architecture.md) - High-level overview and workflow
- [Provider System README](../langextract/providers/README.md) - Provider development guide
- [CONTRIBUTING.md](../CONTRIBUTING.md) - Development guidelines
- [README.md](../README.md) - Quick start and examples

