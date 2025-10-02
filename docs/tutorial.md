# LangExtract Tutorial

This tutorial provides step-by-step walkthroughs for common extraction tasks. Each example builds on previous concepts while introducing new features.

## Table of Contents

1. [Basic Extraction](#1-basic-extraction)
2. [Adding Attributes](#2-adding-attributes)
3. [Multiple Extraction Classes](#3-multiple-extraction-classes)
4. [Processing Long Documents](#4-processing-long-documents)
5. [Working with URLs](#5-working-with-urls)
6. [Using Local Models](#6-using-local-models)
7. [Batch Processing Multiple Documents](#7-batch-processing-multiple-documents)
8. [Improving Recall with Multiple Passes](#8-improving-recall-with-multiple-passes)

---

## 1. Basic Extraction

**Goal**: Extract character names from a story.

### Step 1: Set Up

```python
import langextract as lx
import textwrap

# Define what to extract
prompt = textwrap.dedent("""\
    Extract all character names mentioned in the text.
    Use the exact name as it appears in the text.""")
```

### Step 2: Create Examples

```python
examples = [
    lx.data.ExampleData(
        text="Alice met Bob at the park. They talked for hours.",
        extractions=[
            lx.data.Extraction(
                extraction_class="character",
                extraction_text="Alice"
            ),
            lx.data.Extraction(
                extraction_class="character",
                extraction_text="Bob"
            ),
        ]
    )
]
```

### Step 3: Run Extraction

```python
story = """
Once upon a time, Emma lived in a small village. 
She had a friend named Thomas who loved adventures.
"""

result = lx.extract(
    text_or_documents=story,
    prompt_description=prompt,
    examples=examples,
    model_id="gemini-2.5-flash",
)

# Display results
for extraction in result.extractions:
    print(f"Found character: {extraction.extraction_text}")
```

**Output:**
```
Found character: Emma
Found character: Thomas
```

### Key Takeaway
Basic extraction requires just three things: a prompt, examples, and input text.

---

## 2. Adding Attributes

**Goal**: Extract characters with their roles and traits.

### Adding Context with Attributes

Attributes let you capture additional information about each extraction:

```python
prompt = textwrap.dedent("""\
    Extract characters and describe their role and personality.
    Include role (e.g., protagonist, villain) and key personality traits.""")

examples = [
    lx.data.ExampleData(
        text="The brave knight Sir Lancelot fought against the evil sorcerer.",
        extractions=[
            lx.data.Extraction(
                extraction_class="character",
                extraction_text="Sir Lancelot",
                attributes={
                    "role": "protagonist",
                    "personality": "brave"
                }
            ),
            lx.data.Extraction(
                extraction_class="character",
                extraction_text="sorcerer",
                attributes={
                    "role": "antagonist",
                    "personality": "evil"
                }
            ),
        ]
    )
]

story = """
Princess Isabella was known for her wisdom and kindness.
The cruel dragon Smaug terrorized the kingdom for years.
"""

result = lx.extract(
    text_or_documents=story,
    prompt_description=prompt,
    examples=examples,
    model_id="gemini-2.5-flash",
)

# Display with attributes
for extraction in result.extractions:
    print(f"\nCharacter: {extraction.extraction_text}")
    print(f"  Role: {extraction.attributes.get('role', 'N/A')}")
    print(f"  Personality: {extraction.attributes.get('personality', 'N/A')}")
```

**Output:**
```
Character: Princess Isabella
  Role: protagonist
  Personality: wise, kind

Character: dragon Smaug
  Role: antagonist
  Personality: cruel
```

### Key Takeaway
Attributes enrich extractions with structured metadata.

---

## 3. Multiple Extraction Classes

**Goal**: Extract different types of information from the same text.

### Extracting Various Entity Types

```python
prompt = textwrap.dedent("""\
    Extract the following from the text:
    - Characters (people or beings)
    - Locations (places)
    - Items (objects mentioned)
    
    For each extraction, use the exact text and provide relevant context.""")

examples = [
    lx.data.ExampleData(
        text="Robin Hood stole a golden arrow from the Sheriff in Nottingham.",
        extractions=[
            lx.data.Extraction(
                extraction_class="character",
                extraction_text="Robin Hood",
                attributes={"description": "legendary outlaw"}
            ),
            lx.data.Extraction(
                extraction_class="character",
                extraction_text="Sheriff",
                attributes={"description": "antagonist"}
            ),
            lx.data.Extraction(
                extraction_class="location",
                extraction_text="Nottingham",
                attributes={"type": "city"}
            ),
            lx.data.Extraction(
                extraction_class="item",
                extraction_text="golden arrow",
                attributes={"owner": "Sheriff"}
            ),
        ]
    )
]

story = """
In the mystical land of Eldoria, the wizard Merlin searched for 
the Crystal of Power hidden in the Dark Forest.
"""

result = lx.extract(
    text_or_documents=story,
    prompt_description=prompt,
    examples=examples,
    model_id="gemini-2.5-flash",
)

# Organize by class
from collections import defaultdict
by_class = defaultdict(list)
for extraction in result.extractions:
    by_class[extraction.extraction_class].append(extraction)

# Display grouped results
for class_name, extractions in by_class.items():
    print(f"\n{class_name.upper()}:")
    for ext in extractions:
        print(f"  - {ext.extraction_text}")
        if ext.attributes:
            print(f"    {ext.attributes}")
```

**Output:**
```
CHARACTER:
  - wizard Merlin
    {'description': 'wizard'}

LOCATION:
  - Eldoria
    {'type': 'mystical land'}
  - Dark Forest
    {'type': 'forest'}

ITEM:
  - Crystal of Power
    {'status': 'hidden'}
```

### Key Takeaway
Use multiple extraction classes to capture diverse information types from the same text.

---

## 4. Processing Long Documents

**Goal**: Extract information from a document that exceeds the model's context window.

### Understanding Chunking

Long documents are automatically split into chunks. Key parameters:

- **max_char_buffer**: Maximum characters per chunk (default: 1000)
- **batch_length**: Chunks processed in parallel (default: 10)
- **max_workers**: Parallel processing threads (default: 10)

```python
prompt = "Extract all scientific terms and their definitions."

examples = [
    lx.data.ExampleData(
        text="Photosynthesis is the process by which plants convert light into energy.",
        extractions=[
            lx.data.Extraction(
                extraction_class="term",
                extraction_text="Photosynthesis",
                attributes={"definition": "process by which plants convert light into energy"}
            ),
        ]
    )
]

# Long document (imagine this is much longer)
long_document = """
Mitochondria are the powerhouse of the cell, producing ATP through cellular respiration.
DNA (deoxyribonucleic acid) stores genetic information in living organisms.
RNA plays a crucial role in protein synthesis and gene expression.
""" * 100  # Repeat to make it longer

result = lx.extract(
    text_or_documents=long_document,
    prompt_description=prompt,
    examples=examples,
    model_id="gemini-2.5-flash",
    max_char_buffer=800,      # Smaller chunks
    batch_length=15,          # Process 15 chunks at a time
    max_workers=20,           # Use 20 parallel workers
    show_progress=True,       # Show progress bar
)

print(f"Extracted {len(result.extractions)} terms from long document")
```

### Tuning Parameters for Long Documents

| Parameter | Small Documents | Large Documents | Very Large Documents |
|-----------|----------------|-----------------|---------------------|
| max_char_buffer | 1500-2000 | 800-1000 | 500-800 |
| batch_length | 5-10 | 10-20 | 20-30 |
| max_workers | 5-10 | 10-20 | 20-50 |

**Trade-offs:**
- **Smaller chunks** (low max_char_buffer) → Better accuracy, more API calls, higher cost
- **Larger chunks** → Faster, cheaper, but may miss entities
- **More workers** → Faster processing, no extra cost

### Key Takeaway
LangExtract automatically handles long documents with chunking and parallel processing.

---

## 5. Working with URLs

**Goal**: Extract information directly from web pages.

### Automatic URL Fetching

LangExtract can download and process content from URLs:

```python
prompt = "Extract book titles and authors mentioned in the text."

examples = [
    lx.data.ExampleData(
        text="Pride and Prejudice by Jane Austen is a classic novel.",
        extractions=[
            lx.data.Extraction(
                extraction_class="book",
                extraction_text="Pride and Prejudice",
                attributes={"author": "Jane Austen"}
            ),
        ]
    )
]

# Extract from Project Gutenberg
result = lx.extract(
    text_or_documents="https://www.gutenberg.org/files/1513/1513-0.txt",
    prompt_description=prompt,
    examples=examples,
    model_id="gemini-2.5-flash",
    fetch_urls=True,  # Enable URL fetching (default)
)

print(f"Extracted from URL: {len(result.extractions)} items")
```

### Disabling URL Fetching

If you want to treat URLs as literal text:

```python
result = lx.extract(
    text_or_documents="Check out https://example.com for more info",
    prompt_description="Extract URLs",
    examples=[...],
    fetch_urls=False,  # Treat URL as literal text
)
```

### Key Takeaway
LangExtract can directly process web content without manual downloading.

---

## 6. Using Local Models

**Goal**: Run extractions offline using Ollama.

### Setup Ollama

1. Install Ollama from [ollama.com](https://ollama.com/)
2. Pull a model:
   ```bash
   ollama pull gemma2:2b
   ```
3. Start Ollama:
   ```bash
   ollama serve
   ```

### Using Ollama in LangExtract

```python
prompt = "Extract product names and prices."

examples = [
    lx.data.ExampleData(
        text="The laptop costs $999 and the mouse is $25.",
        extractions=[
            lx.data.Extraction(
                extraction_class="product",
                extraction_text="laptop",
                attributes={"price": "$999"}
            ),
            lx.data.Extraction(
                extraction_class="product",
                extraction_text="mouse",
                attributes={"price": "$25"}
            ),
        ]
    )
]

text = "The smartphone is priced at $599 while the tablet costs $399."

# Use local model
result = lx.extract(
    text_or_documents=text,
    prompt_description=prompt,
    examples=examples,
    model_id="gemma2:2b",              # Ollama model
    model_url="http://localhost:11434", # Ollama server
    fence_output=False,                 # Ollama doesn't use fences
    use_schema_constraints=False,       # Disable for Ollama
)

for extraction in result.extractions:
    print(f"{extraction.extraction_text}: {extraction.attributes.get('price')}")
```

### Key Takeaway
Use Ollama for offline, private, and cost-free extraction.

---

## 7. Batch Processing Multiple Documents

**Goal**: Process multiple documents efficiently.

### Processing Document Collections

```python
prompt = "Extract key topics discussed in each document."

examples = [
    lx.data.ExampleData(
        text="This report discusses climate change impacts on agriculture.",
        extractions=[
            lx.data.Extraction(
                extraction_class="topic",
                extraction_text="climate change"
            ),
            lx.data.Extraction(
                extraction_class="topic",
                extraction_text="agriculture"
            ),
        ]
    )
]

# Create document objects
documents = [
    lx.data.Document(
        text="Artificial intelligence is transforming healthcare and medical diagnosis.",
        document_id="doc_001"
    ),
    lx.data.Document(
        text="Renewable energy sources like solar and wind are becoming more efficient.",
        document_id="doc_002"
    ),
    lx.data.Document(
        text="Quantum computing promises to revolutionize cryptography and drug discovery.",
        document_id="doc_003"
    ),
]

# Process all documents
results = lx.extract(
    text_or_documents=documents,
    prompt_description=prompt,
    examples=examples,
    model_id="gemini-2.5-flash",
)

# Results is an iterable of AnnotatedDocuments
for doc in results:
    print(f"\nDocument {doc.document_id}:")
    for extraction in doc.extractions:
        print(f"  - {extraction.extraction_text}")
```

**Output:**
```
Document doc_001:
  - Artificial intelligence
  - healthcare
  - medical diagnosis

Document doc_002:
  - Renewable energy
  - solar
  - wind

Document doc_003:
  - Quantum computing
  - cryptography
  - drug discovery
```

### Key Takeaway
Process multiple documents in a single call for efficient batch operations.

---

## 8. Improving Recall with Multiple Passes

**Goal**: Increase the number of extractions found by processing the document multiple times.

### Why Multiple Passes?

Sometimes a single pass misses some entities due to:
- Long documents with many entities
- Complex or ambiguous text
- Model limitations on context window

Multiple passes re-process the document with different contexts, improving recall.

```python
prompt = "Extract all character names from this novel excerpt."

examples = [
    lx.data.ExampleData(
        text="Elizabeth and Darcy walked through the garden.",
        extractions=[
            lx.data.Extraction(
                extraction_class="character",
                extraction_text="Elizabeth"
            ),
            lx.data.Extraction(
                extraction_class="character",
                extraction_text="Darcy"
            ),
        ]
    )
]

long_story = """
In the bustling city of London, Sarah met James at a café. 
Later that evening, she encountered Michael at the theater.
The next day, Elizabeth introduced her to Thomas and Margaret.
At the party, William, Catherine, and Robert were also present.
""" * 10  # Repeat to create a longer text

# Single pass
result_single = lx.extract(
    text_or_documents=long_story,
    prompt_description=prompt,
    examples=examples,
    model_id="gemini-2.5-flash",
    extraction_passes=1,
)

# Multiple passes
result_multi = lx.extract(
    text_or_documents=long_story,
    prompt_description=prompt,
    examples=examples,
    model_id="gemini-2.5-flash",
    extraction_passes=3,  # Process 3 times
)

print(f"Single pass: {len(result_single.extractions)} characters")
print(f"Three passes: {len(result_multi.extractions)} characters")
print(f"Improvement: {len(result_multi.extractions) - len(result_single.extractions)} more found")
```

### Cost vs. Recall Trade-off

| Passes | Recall | API Calls | Cost | Use When |
|--------|--------|-----------|------|----------|
| 1 | Good | 1x | Low | Testing, short docs |
| 2 | Better | 2x | Medium | Important extractions |
| 3 | Best | 3x | High | Critical tasks, long docs |

### Key Takeaway
Use multiple passes when recall is critical, accepting higher API costs.

---

## Advanced: Combining Techniques

Here's a real-world example combining multiple techniques:

```python
import langextract as lx

# Complex extraction task
prompt = """
Extract medical information from clinical notes:
- Medications (drug name, dosage, frequency)
- Conditions (diagnosis, severity)
- Procedures (name, date if mentioned)
"""

examples = [
    lx.data.ExampleData(
        text="Patient prescribed Lisinopril 10mg daily for hypertension.",
        extractions=[
            lx.data.Extraction(
                extraction_class="medication",
                extraction_text="Lisinopril",
                attributes={
                    "dosage": "10mg",
                    "frequency": "daily"
                }
            ),
            lx.data.Extraction(
                extraction_class="condition",
                extraction_text="hypertension",
                attributes={"severity": "moderate"}
            ),
        ]
    )
]

# Process from URL with optimal settings
result = lx.extract(
    text_or_documents="https://example.com/clinical-notes.txt",
    prompt_description=prompt,
    examples=examples,
    model_id="gemini-2.5-pro",  # Use Pro for medical accuracy
    max_char_buffer=800,         # Smaller chunks for precision
    batch_length=20,             # Parallel processing
    max_workers=30,              # High parallelism
    extraction_passes=3,         # Multiple passes for recall
    temperature=0.0,             # Deterministic for medical
    show_progress=True,
)

# Save results
lx.io.save_annotated_documents([result], "medical_extractions.jsonl", ".")

# Generate visualization
html = lx.visualize("medical_extractions.jsonl")
with open("medical_viz.html", "w") as f:
    f.write(html if isinstance(html, str) else html.data)

print(f"Extracted {len(result.extractions)} medical entities")
print("Visualization saved to medical_viz.html")
```

---

## Best Practices Summary

### 1. Writing Good Prompts
- Be specific about what to extract
- Specify whether to use exact text or allow inference
- Describe the expected attributes clearly
- Provide context about the domain

### 2. Creating Quality Examples
- Use realistic text similar to your actual data
- Include edge cases (complex entities, multiple words, etc.)
- Show all extraction classes you want
- Include representative attributes
- Start with 1-3 examples, add more if needed

### 3. Choosing Parameters
- Start with defaults, then optimize
- Use smaller chunks for better accuracy
- Increase workers for faster processing
- Use multiple passes for critical tasks
- Set temperature=0.0 for deterministic output

### 4. Model Selection
- **gemini-2.5-flash**: Default choice, fast and cheap
- **gemini-2.5-pro**: Complex tasks, higher accuracy needed
- **Local (Ollama)**: Privacy, offline, free
- **OpenAI**: If you're already using OpenAI

### 5. Testing and Iteration
- Test on small samples first
- Review results and refine examples
- Adjust parameters based on performance
- Use visualization to verify quality

---

## Next Steps

- **[API Reference](api-reference.md)** - Complete parameter documentation
- **[Examples](examples/)** - More real-world scenarios
- **[Architecture](architecture.md)** - Understand the internals
- **[FAQ](faq.md)** - Common questions
- **[Troubleshooting](troubleshooting.md)** - Fix issues

---

Happy extracting! 🎉
