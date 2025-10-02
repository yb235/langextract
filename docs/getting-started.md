# Getting Started with LangExtract

Welcome! This guide will help you get started with LangExtract, a Python library that extracts structured information from unstructured text using Large Language Models (LLMs).

## What is LangExtract?

LangExtract is designed to solve a common problem: extracting structured, meaningful information from unstructured text documents. Instead of writing complex regular expressions or rules, you simply provide a few examples of what you want to extract, and LangExtract uses AI to find similar patterns in your text.

### What Makes LangExtract Special?

1. **Source Grounding**: Every extracted piece of information is mapped to its exact location in the original text
2. **Structured Output**: Get reliable, consistent JSON/YAML results based on your examples
3. **Long Document Support**: Efficiently process large documents (novels, reports, etc.) with automatic chunking
4. **Interactive Visualization**: Review extractions with a beautiful HTML viewer
5. **Flexible Model Support**: Use Gemini, OpenAI, local models via Ollama, or custom providers

## Prerequisites

- **Python 3.10 or higher** (Python 3.11 recommended)
- **API Key** (for cloud models) or **Local LLM** (via Ollama)
- **Basic Python knowledge** (ability to write simple scripts)

## Installation

### Step 1: Set Up a Virtual Environment (Recommended)

Creating a virtual environment keeps your project dependencies isolated:

```bash
# Create a new directory for your project
mkdir my-langextract-project
cd my-langextract-project

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Step 2: Install LangExtract

```bash
# Basic installation (includes Gemini and Ollama support)
pip install langextract

# For OpenAI support:
pip install langextract[openai]

# For development (includes testing and linting tools):
pip install langextract[dev]
```

### Step 3: Get an API Key

If you're using cloud-hosted models like Gemini or OpenAI, you'll need an API key.

#### For Gemini Models (Recommended for Beginners)

1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Click "Get API Key" or "Create API Key"
3. Copy your API key

#### For OpenAI Models

1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)
2. Sign in or create an account
3. Generate a new API key
4. Copy your API key

### Step 4: Configure Your API Key

You have several options for providing your API key:

#### Option A: Environment Variable (Recommended)

```bash
export LANGEXTRACT_API_KEY="your-api-key-here"
```

To make this permanent, add it to your shell profile (~/.bashrc, ~/.zshrc, etc.).

#### Option B: .env File (Best for Projects)

Create a `.env` file in your project directory:

```bash
# Create .env file
echo "LANGEXTRACT_API_KEY=your-api-key-here" > .env

# Keep it secure - add to .gitignore
echo ".env" >> .gitignore
```

LangExtract will automatically read this file.

#### Option C: Direct in Code (Not Recommended for Production)

```python
result = lx.extract(
    text=your_text,
    prompt_description="...",
    examples=[...],
    api_key="your-api-key-here"  # Only for testing!
)
```

## Your First Extraction

Let's create a simple extraction task to get familiar with LangExtract.

### Step 1: Import LangExtract

Create a new Python file (e.g., `first_extraction.py`):

```python
import langextract as lx
```

### Step 2: Define What You Want to Extract

Think about your task:
- What kind of information do you want to find?
- What attributes should each extraction have?
- How should the information be structured?

For this example, let's extract movie information:

```python
import textwrap

# Describe your extraction task clearly
prompt = textwrap.dedent("""\
    Extract movies mentioned in the text.
    For each movie, extract:
    - The movie title
    - The year it was released (if mentioned)
    - Any genre information (if mentioned)
    
    Use exact text from the document. Do not infer information not present.""")
```

### Step 3: Provide Examples

Examples guide the AI. Provide at least one high-quality example:

```python
examples = [
    lx.data.ExampleData(
        text="I watched The Matrix (1999), a groundbreaking sci-fi film.",
        extractions=[
            lx.data.Extraction(
                extraction_class="movie",
                extraction_text="The Matrix",
                attributes={
                    "year": "1999",
                    "genre": "sci-fi"
                }
            ),
        ]
    )
]
```

**Understanding the Example:**
- `text`: Sample input text
- `extractions`: List of things to extract from that text
- `extraction_class`: Category/type of extraction (e.g., "movie", "person", "date")
- `extraction_text`: The exact text to extract from the document
- `attributes`: Additional information about the extraction (key-value pairs)

### Step 4: Run the Extraction

Now apply your task to new text:

```python
# Your input text
input_text = """
Yesterday I finally watched Inception (2010), an incredible thriller 
that kept me on the edge of my seat. My friend recommended Parasite, 
the 2019 thriller that won Best Picture.
"""

# Run the extraction
result = lx.extract(
    text_or_documents=input_text,
    prompt_description=prompt,
    examples=examples,
    model_id="gemini-2.5-flash",  # Fast and cost-effective
)

# Print results
print(f"Found {len(result.extractions)} extractions:")
for extraction in result.extractions:
    print(f"\n  Class: {extraction.extraction_class}")
    print(f"  Text: {extraction.extraction_text}")
    print(f"  Attributes: {extraction.attributes}")
    print(f"  Position: chars {extraction.char_interval.start_pos}-{extraction.char_interval.end_pos}")
```

### Step 5: View Results

Run your script:

```bash
python first_extraction.py
```

Expected output:
```
Found 3 extractions:

  Class: movie
  Text: Inception
  Attributes: {'year': '2010', 'genre': 'thriller'}
  Position: chars 25-34

  Class: movie
  Text: Parasite
  Attributes: {'year': '2019', 'genre': 'thriller'}
  Position: chars 142-150
...
```

### Step 6: Visualize Results

Create an interactive HTML visualization:

```python
# Save results to a file
lx.io.save_annotated_documents([result], output_name="my_extraction.jsonl", output_dir=".")

# Generate visualization
html_content = lx.visualize("my_extraction.jsonl")

# Save to HTML file
with open("visualization.html", "w") as f:
    if hasattr(html_content, 'data'):
        f.write(html_content.data)
    else:
        f.write(html_content)

print("Open 'visualization.html' in your browser to view results!")
```

## Understanding the Basics

### Core Concepts

#### 1. ExampleData
Examples teach the AI what to extract and how to structure it:

```python
example = lx.data.ExampleData(
    text="Sample input text here",
    extractions=[
        lx.data.Extraction(
            extraction_class="category",      # What type of thing
            extraction_text="exact text",     # What to extract
            attributes={"key": "value"}       # Additional info
        ),
    ]
)
```

#### 2. Extraction
The result of extraction contains:

```python
extraction = lx.data.Extraction(
    extraction_class="movie",           # Category
    extraction_text="The Matrix",       # Extracted text
    extraction_index=0,                 # Order of appearance
    attributes={"year": "1999"},        # Extra information
    char_interval=CharInterval(10, 20), # Position in source text
    alignment_status="MATCH_EXACT"      # How well it was found
)
```

#### 3. AnnotatedDocument
The complete result containing all extractions:

```python
result = lx.extract(...)
# result is an AnnotatedDocument with:
result.text          # Original input text
result.document_id   # Unique identifier
result.extractions   # List of Extraction objects
```

### Parameters Explained

Common parameters for `lx.extract()`:

- **text_or_documents**: Your input text, URL, or list of documents
- **prompt_description**: Instructions for what to extract
- **examples**: List of ExampleData showing what you want
- **model_id**: Which AI model to use (default: "gemini-2.5-flash")
- **api_key**: Your API key (or use environment variable)
- **max_char_buffer**: Maximum chunk size for long documents (default: 1000)
- **batch_length**: Number of chunks to process in parallel (default: 10)
- **max_workers**: Number of parallel workers (default: 10)
- **extraction_passes**: Number of times to process document (default: 1, more = better recall)
- **temperature**: Randomness of model output (0.0 = deterministic, higher = more variation)
- **fence_output**: Whether to wrap output in ```json or ```yaml markers
- **use_schema_constraints**: Enable strict schema validation (default: True for Gemini)

## Common Use Cases

### 1. Extract Named Entities

```python
prompt = "Extract person names, organizations, and locations."
examples = [
    lx.data.ExampleData(
        text="John Smith works at Google in California.",
        extractions=[
            lx.data.Extraction(extraction_class="person", extraction_text="John Smith"),
            lx.data.Extraction(extraction_class="organization", extraction_text="Google"),
            lx.data.Extraction(extraction_class="location", extraction_text="California"),
        ]
    )
]
```

### 2. Extract Structured Information

```python
prompt = "Extract product information including price and specifications."
examples = [
    lx.data.ExampleData(
        text="The iPhone 15 Pro costs $999 and has 256GB storage.",
        extractions=[
            lx.data.Extraction(
                extraction_class="product",
                extraction_text="iPhone 15 Pro",
                attributes={"price": "$999", "storage": "256GB"}
            ),
        ]
    )
]
```

### 3. Extract from Long Documents

```python
result = lx.extract(
    text_or_documents="https://www.gutenberg.org/files/1513/1513-0.txt",  # Romeo & Juliet
    prompt_description=prompt,
    examples=examples,
    model_id="gemini-2.5-flash",
    max_char_buffer=1000,      # Smaller chunks for better accuracy
    extraction_passes=3,       # Multiple passes for better recall
    max_workers=20,            # Parallel processing for speed
)
```

## Choosing a Model

### Recommended Models

| Model | Best For | Speed | Cost | Quality |
|-------|----------|-------|------|---------|
| `gemini-2.5-flash` | General use | Fast | Low | Good |
| `gemini-2.5-pro` | Complex tasks | Slower | Higher | Excellent |
| `gpt-4o` | OpenAI users | Medium | Medium | Excellent |
| `gemma2:2b` (Ollama) | Local/offline | Fast | Free | Good |

### Model Selection Guide

**Start with `gemini-2.5-flash`** - It offers the best balance of speed, cost, and quality for most tasks.

**Use `gemini-2.5-pro`** when:
- Your extraction task is complex
- You need higher accuracy
- Cost is not a primary concern

**Use local models (Ollama)** when:
- You need offline capability
- You have privacy/security requirements
- You want to avoid API costs
- You're experimenting/learning

## Next Steps

Now that you have the basics:

1. **Try different extraction tasks** - Experiment with your own data
2. **Read the [Tutorial](tutorial.md)** - Step-by-step walkthroughs
3. **Explore [API Reference](api-reference.md)** - Complete parameter documentation
4. **Check [Examples](examples/)** - Real-world use cases
5. **Learn about [Architecture](architecture.md)** - Understand how it works

## Getting Help

- **[FAQ](faq.md)** - Common questions and answers
- **[Troubleshooting](troubleshooting.md)** - Fix common issues
- **[GitHub Issues](https://github.com/google/langextract/issues)** - Report bugs or request features
- **[Examples](examples/)** - See working code for various scenarios

## Quick Reference Card

```python
import langextract as lx

# 1. Define task
prompt = "Extract [what you want]"
examples = [
    lx.data.ExampleData(
        text="example input",
        extractions=[
            lx.data.Extraction(
                extraction_class="type",
                extraction_text="text to extract",
                attributes={"key": "value"}
            ),
        ]
    )
]

# 2. Run extraction
result = lx.extract(
    text_or_documents=your_text,
    prompt_description=prompt,
    examples=examples,
    model_id="gemini-2.5-flash",
)

# 3. Use results
for extraction in result.extractions:
    print(extraction.extraction_text)

# 4. Visualize
lx.io.save_annotated_documents([result], "output.jsonl", ".")
html = lx.visualize("output.jsonl")
with open("viz.html", "w") as f:
    f.write(html if isinstance(html, str) else html.data)
```

---

**Ready to dive deeper?** Continue to the [Tutorial](tutorial.md) for hands-on walkthroughs!
