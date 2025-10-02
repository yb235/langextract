# API Reference

Complete reference for LangExtract's public API.

## Table of Contents

1. [Core Functions](#core-functions)
2. [Data Classes](#data-classes)
3. [Configuration](#configuration)
4. [I/O Operations](#io-operations)
5. [Visualization](#visualization)
6. [Provider System](#provider-system)
7. [Advanced Components](#advanced-components)

---

## Core Functions

### `lx.extract()`

Main extraction function that processes text and returns structured extractions.

```python
lx.extract(
    text_or_documents,
    prompt_description=None,
    examples=None,
    model_id="gemini-2.5-flash",
    api_key=None,
    **kwargs
) -> AnnotatedDocument | Iterable[AnnotatedDocument]
```

#### Parameters

##### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `text_or_documents` | `str` \| `Iterable[Document]` | Input text, URL (if `fetch_urls=True`), or collection of Document objects |
| `prompt_description` | `str` | Instructions describing what to extract from the text |
| `examples` | `Sequence[ExampleData]` | List of examples showing desired extraction format |

##### Model Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_id` | `str` | `"gemini-2.5-flash"` | Model identifier (e.g., "gemini-2.5-flash", "gpt-4o", "gemma2:2b") |
| `api_key` | `str` \| `None` | `None` | API key for cloud models (or use `LANGEXTRACT_API_KEY` env var) |
| `model_url` | `str` \| `None` | `None` | URL for model endpoint (primarily for Ollama: "http://localhost:11434") |
| `temperature` | `float` \| `None` | `None` | Sampling temperature (0.0=deterministic, higher=more random) |
| `language_model_params` | `dict` \| `None` | `None` | Provider-specific parameters (e.g., `{"vertexai": True}`) |

##### Processing Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_char_buffer` | `int` | `1000` | Maximum characters per chunk for long documents |
| `batch_length` | `int` | `10` | Number of chunks to process in each batch |
| `max_workers` | `int` | `10` | Number of parallel workers for processing |
| `extraction_passes` | `int` | `1` | Number of times to process document (higher=better recall, more cost) |
| `show_progress` | `bool` | `True` | Whether to display progress bar during processing |

##### Format Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `format_type` | `FormatType` | `None` | Output format: `FormatType.JSON` or `FormatType.YAML` |
| `fence_output` | `bool` \| `None` | `None` | Whether to wrap output in \\`\\`\\`json/yaml markers (auto-detected) |
| `use_schema_constraints` | `bool` | `True` | Enable strict schema validation (provider-dependent) |

##### Advanced Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `additional_context` | `str` \| `None` | `None` | Extra context to append to each prompt |
| `resolver_params` | `dict` \| `None` | `None` | Parameters for the Resolver component |
| `debug` | `bool` | `False` | Enable debug mode with additional information |
| `fetch_urls` | `bool` | `True` | Auto-download URLs starting with http:// or https:// |
| `prompt_validation_level` | `PromptValidationLevel` | `WARNING` | Validation level: `OFF`, `WARNING`, or `ERROR` |
| `prompt_validation_strict` | `bool` | `False` | Strict validation mode (with `ERROR` level) |

#### Returns

- **Single input**: `AnnotatedDocument` with all extractions
- **Multiple inputs**: `Iterable[AnnotatedDocument]`, one per input document

#### Raises

- `ValueError`: If examples is None or empty
- `ValueError`: If no API key provided (for cloud models)
- `InferenceConfigError`: If model configuration is invalid
- `ResolverParsingError`: If LLM output cannot be parsed

#### Examples

**Basic usage:**
```python
result = lx.extract(
    text_or_documents="Your input text here",
    prompt_description="Extract person names",
    examples=[
        lx.data.ExampleData(
            text="John works here",
            extractions=[lx.data.Extraction("person", "John")]
        )
    ],
)
```

**Long document with multiple passes:**
```python
result = lx.extract(
    text_or_documents=long_text,
    prompt_description="Extract entities",
    examples=examples,
    model_id="gemini-2.5-flash",
    max_char_buffer=800,
    extraction_passes=3,
    max_workers=20,
)
```

**Using OpenAI:**
```python
result = lx.extract(
    text_or_documents=text,
    prompt_description="Extract info",
    examples=examples,
    model_id="gpt-4o",
    api_key=os.environ["OPENAI_API_KEY"],
    fence_output=True,
    use_schema_constraints=False,
)
```

**From URL:**
```python
result = lx.extract(
    text_or_documents="https://example.com/document.txt",
    prompt_description="Extract data",
    examples=examples,
    fetch_urls=True,
)
```

### `lx.visualize()`

Generate interactive HTML visualization of extractions.

```python
lx.visualize(
    data_source,
    *,
    animation_speed=1.0,
    show_legend=True,
    gif_optimized=True
) -> HTML | str
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_source` | `AnnotatedDocument` \| `str` \| `Path` | Required | Document or path to .jsonl file |
| `animation_speed` | `float` | `1.0` | Animation speed multiplier (higher=faster) |
| `show_legend` | `bool` | `True` | Whether to show extraction class legend |
| `gif_optimized` | `bool` | `True` | Apply GIF-friendly styling |

#### Returns

- `HTML` object (in Jupyter/Colab) or `str` (elsewhere)

#### Example

```python
# From AnnotatedDocument
html = lx.visualize(result)

# From file
html = lx.visualize("extractions.jsonl")

# Save to file
with open("viz.html", "w") as f:
    if hasattr(html, 'data'):
        f.write(html.data)
    else:
        f.write(html)
```

---

## Data Classes

### `ExampleData`

Represents a training example for the extraction task.

```python
lx.data.ExampleData(
    text: str,
    extractions: Sequence[Extraction],
    document_id: str | None = None
)
```

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `text` | `str` | Example input text |
| `extractions` | `Sequence[Extraction]` | List of expected extractions from this text |
| `document_id` | `str` \| `None` | Optional identifier for the example |

#### Example

```python
example = lx.data.ExampleData(
    text="Apple released iPhone 15 in 2023.",
    extractions=[
        lx.data.Extraction(
            extraction_class="company",
            extraction_text="Apple"
        ),
        lx.data.Extraction(
            extraction_class="product",
            extraction_text="iPhone 15",
            attributes={"year": "2023"}
        ),
    ]
)
```

### `Extraction`

Represents a single extracted piece of information.

```python
lx.data.Extraction(
    extraction_class: str,
    extraction_text: str,
    extraction_index: int = 0,
    attributes: dict | None = None,
    char_interval: CharInterval | None = None,
    token_interval: TokenInterval | None = None,
    alignment_status: str | None = None
)
```

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `extraction_class` | `str` | Category/type of extraction (e.g., "person", "location") |
| `extraction_text` | `str` | The actual extracted text |
| `extraction_index` | `int` | Order of appearance (0-based) |
| `attributes` | `dict` \| `None` | Additional key-value metadata |
| `char_interval` | `CharInterval` \| `None` | Character positions in source text |
| `token_interval` | `TokenInterval` \| `None` | Token positions in source text |
| `alignment_status` | `str` \| `None` | Alignment quality: "MATCH_EXACT", "MATCH_FUZZY", "MATCH_LESSER" |

#### Example

```python
extraction = lx.data.Extraction(
    extraction_class="medication",
    extraction_text="Aspirin",
    extraction_index=0,
    attributes={
        "dosage": "100mg",
        "frequency": "daily"
    },
    char_interval=lx.data.CharInterval(45, 52),
    alignment_status="MATCH_EXACT"
)

# Access attributes
print(extraction.extraction_text)  # "Aspirin"
print(extraction.attributes["dosage"])  # "100mg"
print(extraction.char_interval.start_pos)  # 45
```

### `Document`

Represents an input document to be processed.

```python
lx.data.Document(
    text: str,
    document_id: str | None = None,
    additional_context: str | None = None
)
```

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `text` | `str` | The document text content |
| `document_id` | `str` \| `None` | Unique identifier for the document |
| `additional_context` | `str` \| `None` | Additional context for extraction |

#### Example

```python
doc = lx.data.Document(
    text="Patient presents with fever and cough.",
    document_id="patient_001",
    additional_context="Emergency room visit 2024-01-15"
)

# Batch processing
documents = [
    lx.data.Document(text="Doc 1 text", document_id="doc1"),
    lx.data.Document(text="Doc 2 text", document_id="doc2"),
]
results = lx.extract(text_or_documents=documents, ...)
```

### `AnnotatedDocument`

Result of extraction containing the original text and all extractions.

```python
class AnnotatedDocument:
    document_id: str
    text: str
    extractions: Sequence[Extraction]
    tokenized_text: TokenizedText | None
```

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `document_id` | `str` | Unique identifier |
| `text` | `str` | Original input text |
| `extractions` | `Sequence[Extraction]` | All extracted information |
| `tokenized_text` | `TokenizedText` \| `None` | Tokenization information |

#### Methods

```python
# Iterate over extractions
for extraction in result.extractions:
    print(extraction.extraction_text)

# Access by index
first = result.extractions[0]

# Count extractions
count = len(result.extractions)

# Filter by class
medications = [e for e in result.extractions if e.extraction_class == "medication"]
```

### `CharInterval`

Represents character position range in text.

```python
lx.data.CharInterval(start_pos: int, end_pos: int)
```

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `start_pos` | `int` | Starting character position (inclusive) |
| `end_pos` | `int` | Ending character position (exclusive) |

#### Example

```python
text = "Hello, World!"
interval = lx.data.CharInterval(7, 12)
extracted = text[interval.start_pos:interval.end_pos]  # "World"
```

---

## Configuration

### `ModelConfig`

Configuration for model instantiation.

```python
lx.factory.ModelConfig(
    model_id: str | None = None,
    provider: str | None = None,
    provider_kwargs: dict | None = None
)
```

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `model_id` | `str` \| `None` | Model identifier |
| `provider` | `str` \| `None` | Explicit provider name or class |
| `provider_kwargs` | `dict` | Provider-specific parameters |

#### Example

```python
# Auto-detect provider
config = lx.factory.ModelConfig(model_id="gemini-2.5-flash")

# Explicit provider
config = lx.factory.ModelConfig(
    model_id="gpt-4o",
    provider="OpenAILanguageModel",
    provider_kwargs={"api_key": "..."}
)

# Create model
model = lx.factory.create_model(config)
```

### `PromptValidationLevel`

Controls pre-flight validation of examples.

```python
from langextract.prompt_validation import PromptValidationLevel

PromptValidationLevel.OFF      # No validation
PromptValidationLevel.WARNING  # Log warnings, continue (default)
PromptValidationLevel.ERROR    # Raise on validation failures
```

#### Example

```python
result = lx.extract(
    text_or_documents=text,
    prompt_description=prompt,
    examples=examples,
    prompt_validation_level=PromptValidationLevel.ERROR,
    prompt_validation_strict=True,  # Strict mode
)
```

---

## I/O Operations

### `lx.io.save_annotated_documents()`

Save extraction results to JSONL format.

```python
lx.io.save_annotated_documents(
    documents: Sequence[AnnotatedDocument],
    output_name: str = "annotated_documents.jsonl",
    output_dir: str = "."
) -> None
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `documents` | `Sequence[AnnotatedDocument]` | Required | Documents to save |
| `output_name` | `str` | `"annotated_documents.jsonl"` | Output filename |
| `output_dir` | `str` | `"."` | Output directory |

#### Example

```python
# Single document
lx.io.save_annotated_documents([result], "output.jsonl", "./results")

# Multiple documents
lx.io.save_annotated_documents(results, "batch_results.jsonl")
```

### `lx.io.load_annotated_documents()`

Load extraction results from JSONL format.

```python
lx.io.load_annotated_documents(
    file_path: str | Path
) -> Sequence[AnnotatedDocument]
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `file_path` | `str` \| `Path` | Path to JSONL file |

#### Returns

List of `AnnotatedDocument` objects

#### Example

```python
# Load results
documents = lx.io.load_annotated_documents("output.jsonl")

# Access loaded data
for doc in documents:
    print(f"Document {doc.document_id}: {len(doc.extractions)} extractions")
```

---

## Visualization

The visualization system generates interactive HTML for reviewing extractions.

### Features

- **Animated playback**: Step through extractions sequentially
- **Highlighting**: Color-coded extraction classes
- **Context view**: See extractions in their surrounding text
- **Controls**: Play/pause, step forward/back, speed control
- **Legend**: Visual key for extraction classes
- **Position info**: Character positions for each extraction

### Customization

```python
# Adjust animation speed
html = lx.visualize(result, animation_speed=2.0)  # 2x faster

# Disable legend
html = lx.visualize(result, show_legend=False)

# Optimize for GIF recording
html = lx.visualize(result, gif_optimized=True)
```

---

## Provider System

### Overview

LangExtract supports multiple LLM providers through a plugin system.

### Built-in Providers

| Provider | Models | Installation |
|----------|--------|--------------|
| Gemini | gemini-2.5-flash, gemini-2.5-pro, etc. | Included |
| Ollama | gemma2:2b, llama3, etc. | Included |
| OpenAI | gpt-4o, gpt-4-turbo, etc. | `pip install langextract[openai]` |

### Provider Selection

```python
# Automatic (by model_id pattern)
lx.extract(..., model_id="gemini-2.5-flash")  # → GeminiLanguageModel

# Explicit
config = lx.factory.ModelConfig(
    model_id="gpt-4o",
    provider="OpenAILanguageModel"
)
model = lx.factory.create_model(config)
```

### Environment Variables

| Provider | Environment Variable | Default |
|----------|---------------------|---------|
| Gemini | `GEMINI_API_KEY` or `LANGEXTRACT_API_KEY` | None |
| OpenAI | `OPENAI_API_KEY` or `LANGEXTRACT_API_KEY` | None |
| Ollama | `OLLAMA_BASE_URL` | `http://localhost:11434` |

### Custom Providers

See [Provider System Documentation](../langextract/providers/README.md) for creating custom providers.

---

## Advanced Components

### Annotator

The core orchestrator for document processing.

```python
from langextract.annotation import Annotator

annotator = Annotator(
    qa_prompt_generator=prompt_generator,
    language_model=model
)

result = annotator.annotate_text(
    text=input_text,
    resolver=resolver,
    max_char_buffer=1000,
    batch_length=10,
    extraction_passes=1,
)
```

### Resolver

Parses LLM output into structured extractions.

```python
from langextract.resolver import Resolver
from langextract.core.format_handler import FormatHandler

# Create format handler
format_handler = FormatHandler(
    format_type="json",
    use_fences=True,
    require_extractions_key=True,
)

# Create resolver
resolver = Resolver(
    format_handler=format_handler,
    extraction_index_suffix="_index",
)

# Parse LLM output
extractions = resolver.resolve(llm_output_string)
```

### Prompt Generation

Create few-shot prompts for LLM.

```python
from langextract.prompting import QAPromptGenerator, PromptTemplateStructured

# Create template
template = PromptTemplateStructured(
    prompt_description="Extract entities",
    examples_data=examples,
)

# Create generator
generator = QAPromptGenerator(
    template=template,
    additional_context="Process medical records",
)

# Generate prompt
prompt = generator.create_qa_prompt(
    document_text=text_chunk,
    output_format="json",
)
```

### Chunking

Break long documents into processable pieces.

```python
from langextract.chunking import chunk_documents

chunks = chunk_documents(
    documents=[document],
    max_char_buffer=1000,
)

for chunk in chunks:
    print(f"Chunk: {chunk.chunk_text[:50]}...")
    print(f"Position: {chunk.char_interval}")
```

---

## Error Handling

### Common Exceptions

| Exception | Cause | Solution |
|-----------|-------|----------|
| `ValueError` | Empty examples or missing API key | Provide valid examples and API key |
| `InferenceConfigError` | Invalid model configuration | Check model_id and provider settings |
| `ResolverParsingError` | Cannot parse LLM output | Check prompt and examples |
| `FormatError` | Invalid JSON/YAML format | Verify output format configuration |

### Example

```python
try:
    result = lx.extract(
        text_or_documents=text,
        prompt_description=prompt,
        examples=examples,
    )
except ValueError as e:
    print(f"Configuration error: {e}")
except lx.exceptions.InferenceConfigError as e:
    print(f"Model configuration error: {e}")
except lx.resolver.ResolverParsingError as e:
    print(f"Failed to parse LLM output: {e}")
```

---

## Type Annotations

LangExtract includes full type annotations for IDE support:

```python
from langextract import extract
from langextract.core.data import AnnotatedDocument, ExampleData, Extraction

def my_extraction(text: str) -> AnnotatedDocument:
    examples: list[ExampleData] = [...]
    result: AnnotatedDocument = extract(
        text_or_documents=text,
        prompt_description="...",
        examples=examples,
    )
    return result
```

---

## Version Compatibility

This API reference is for LangExtract 1.x.

Breaking changes planned for 2.0:
- Removal of deprecated `language_model_type` parameter
- Updated resolver parameter handling
- Enhanced schema system

---

## See Also

- [Getting Started Guide](getting-started.md)
- [Tutorial](tutorial.md)
- [Architecture Documentation](architecture.md)
- [Examples](examples/)
- [GitHub Repository](https://github.com/google/langextract)
