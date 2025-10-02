# Frequently Asked Questions (FAQ)

Common questions and answers about LangExtract.

## Table of Contents

- [General Questions](#general-questions)
- [Getting Started](#getting-started)
- [Models and Providers](#models-and-providers)
- [Extraction Quality](#extraction-quality)
- [Performance and Cost](#performance-and-cost)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)

---

## General Questions

### What is LangExtract?

LangExtract is a Python library that uses Large Language Models (LLMs) to extract structured information from unstructured text. Instead of writing complex rules or regular expressions, you provide a few examples of what you want to extract, and LangExtract uses AI to find similar patterns.

### What makes LangExtract different from other extraction tools?

1. **Source Grounding**: Every extraction is mapped to its exact location in the source text
2. **Few-shot Learning**: Learn from examples without training or fine-tuning
3. **Long Document Support**: Automatically handles documents of any length with chunking
4. **Interactive Visualization**: Beautiful HTML viewer for reviewing results
5. **Flexible**: Works with multiple LLM providers (Gemini, OpenAI, Ollama, custom)

### Is LangExtract free to use?

Yes, the library itself is free and open-source (Apache 2.0 license). However:
- **Cloud models** (Gemini, OpenAI) charge for API usage based on tokens
- **Local models** (Ollama) are completely free but require running locally
- See your LLM provider's pricing for costs

### Can I use LangExtract offline?

Yes! Use Ollama with local models:

```python
result = lx.extract(
    text_or_documents=text,
    prompt_description=prompt,
    examples=examples,
    model_id="gemma2:2b",
    model_url="http://localhost:11434",
    fence_output=False,
    use_schema_constraints=False,
)
```

See [Ollama setup guide](../examples/ollama/README.md).

### What programming languages does LangExtract support?

LangExtract is written in Python and currently only has a Python API. The library requires Python 3.10 or higher.

---

## Getting Started

### How do I install LangExtract?

```bash
# Basic installation
pip install langextract

# With OpenAI support
pip install langextract[openai]

# With development tools
pip install langextract[dev]
```

### Do I need an API key?

- **Yes** for cloud models (Gemini, OpenAI, etc.)
- **No** for local models (Ollama)

Get API keys from:
- [Google AI Studio](https://aistudio.google.com/app/apikey) for Gemini
- [OpenAI Platform](https://platform.openai.com/api-keys) for OpenAI

### How do I set up my API key?

**Recommended**: Use environment variable
```bash
export LANGEXTRACT_API_KEY="your-key-here"
```

**Alternative**: Use `.env` file
```bash
echo "LANGEXTRACT_API_KEY=your-key" > .env
```

**Development only**: Pass directly
```python
lx.extract(..., api_key="your-key")  # Not recommended for production
```

### What's the minimum working example?

```python
import langextract as lx

prompt = "Extract person names"
examples = [
    lx.data.ExampleData(
        text="John works here",
        extractions=[lx.data.Extraction("person", "John")]
    )
]

result = lx.extract(
    text_or_documents="Alice and Bob went to the store",
    prompt_description=prompt,
    examples=examples,
)

for extraction in result.extractions:
    print(extraction.extraction_text)
```

---

## Models and Providers

### Which model should I use?

**For most tasks**: `gemini-2.5-flash`
- Fast, cost-effective, good quality
- Best starting point

**For complex tasks**: `gemini-2.5-pro`
- Better accuracy and reasoning
- Higher cost, slower

**For local/offline**: `gemma2:2b` (via Ollama)
- Free, private
- Requires local setup

**For OpenAI users**: `gpt-4o`
- If you're already using OpenAI

### How do I switch models?

Just change the `model_id` parameter:

```python
# Gemini
result = lx.extract(..., model_id="gemini-2.5-flash")

# OpenAI
result = lx.extract(..., model_id="gpt-4o", api_key=openai_key)

# Ollama
result = lx.extract(..., model_id="gemma2:2b")
```

### Can I use other LLM providers?

Yes! LangExtract supports custom providers through a plugin system. See [Provider System Documentation](../langextract/providers/README.md) for details.

### How do I use OpenAI models?

1. Install OpenAI support: `pip install langextract[openai]`
2. Get API key from [OpenAI Platform](https://platform.openai.com/api-keys)
3. Use in code:

```python
result = lx.extract(
    text_or_documents=text,
    prompt_description=prompt,
    examples=examples,
    model_id="gpt-4o",
    api_key=os.environ["OPENAI_API_KEY"],
    fence_output=True,
    use_schema_constraints=False,
)
```

### What's the difference between Gemini models?

| Model | Speed | Cost | Quality | Best For |
|-------|-------|------|---------|----------|
| gemini-2.5-flash | Very Fast | Low | Good | Most tasks, high volume |
| gemini-2.5-pro | Medium | Higher | Excellent | Complex reasoning, accuracy critical |
| gemini-1.5-flash | Fast | Low | Good | Legacy compatibility |
| gemini-1.5-pro | Medium | Medium | Very Good | Legacy complex tasks |

---

## Extraction Quality

### How do I improve extraction accuracy?

1. **Provide better examples**
   - Use realistic text similar to your data
   - Include edge cases
   - Show all extraction types you want

2. **Refine your prompt**
   - Be specific about what to extract
   - Clarify whether to use exact text or allow inference
   - Provide domain context

3. **Adjust parameters**
   - Use smaller chunks: `max_char_buffer=800`
   - Set deterministic output: `temperature=0.0`
   - Use better model: `model_id="gemini-2.5-pro"`

4. **Use multiple passes**
   ```python
   result = lx.extract(..., extraction_passes=3)
   ```

### How many examples should I provide?

**Start with 1-2 good examples**, then add more if needed:
- **1 example**: Often sufficient for simple tasks
- **2-3 examples**: Good for most tasks
- **3-5 examples**: For complex or ambiguous tasks
- **More than 5**: Rarely needed, may slow down

Quality matters more than quantity!

### Why are some extractions missing?

Common causes:
1. **Document too long**: Try smaller `max_char_buffer` or multiple passes
2. **Ambiguous text**: Improve examples or prompt clarity
3. **Model limitations**: Try a more capable model
4. **Examples don't match**: Ensure examples are representative

**Solution**: Use multiple passes
```python
result = lx.extract(
    ...,
    extraction_passes=3,  # Process 3 times
    max_char_buffer=800,  # Smaller chunks
)
```

### Why are there false positives (wrong extractions)?

1. **Overly broad prompt**: Be more specific
2. **Misleading examples**: Ensure examples are representative
3. **Model hallucination**: Set `temperature=0.0` for deterministic output

**Example fix**:
```python
# Vague
prompt = "Extract things mentioned"

# Specific
prompt = "Extract only product names that are explicitly mentioned, not inferred"
```

### How do I extract relationships between entities?

Include relationship examples in your extraction classes:

```python
examples = [
    lx.data.ExampleData(
        text="Alice manages Bob. Carol reports to Alice.",
        extractions=[
            lx.data.Extraction(
                extraction_class="relationship",
                extraction_text="Alice manages Bob",
                attributes={
                    "subject": "Alice",
                    "predicate": "manages",
                    "object": "Bob"
                }
            ),
            lx.data.Extraction(
                extraction_class="relationship",
                extraction_text="Carol reports to Alice",
                attributes={
                    "subject": "Carol",
                    "predicate": "reports to",
                    "object": "Alice"
                }
            ),
        ]
    )
]
```

---

## Performance and Cost

### How much does it cost to use LangExtract?

Costs depend on your LLM provider:

**Gemini** (as of 2024):
- gemini-2.5-flash: ~$0.075 per 1M input tokens
- gemini-2.5-pro: ~$1.25 per 1M input tokens

**OpenAI** (as of 2024):
- gpt-4o: ~$2.50 per 1M input tokens

**Ollama**: Free (runs locally)

**Cost factors**:
- Document length
- `max_char_buffer` (smaller = more API calls)
- `extraction_passes` (multiplies cost)
- Model choice

**Estimate costs**: Test on small sample first!

### How can I reduce costs?

1. **Use cheaper models**: `gemini-2.5-flash` instead of Pro
2. **Larger chunks**: Increase `max_char_buffer` to 1500-2000
3. **Single pass**: Use `extraction_passes=1` (default)
4. **Test on samples**: Validate approach on small data first
5. **Use local models**: Ollama is free

### How can I speed up processing?

1. **Increase parallelism**:
   ```python
   result = lx.extract(
       ...,
       max_workers=30,      # More parallel workers
       batch_length=20,     # Larger batches
   )
   ```

2. **Use faster model**: `gemini-2.5-flash` is faster than Pro

3. **Optimize chunks**: Balance between speed and accuracy
   ```python
   max_char_buffer=1500  # Larger chunks = fewer API calls
   ```

4. **Batch documents**: Process multiple documents together

### How long does processing take?

Depends on:
- Document length
- Model speed
- Number of workers
- API rate limits

**Rough estimates** (with gemini-2.5-flash):
- Short text (1-2 paragraphs): 1-3 seconds
- Medium document (5-10 pages): 10-30 seconds
- Long document (100+ pages): 2-10 minutes

**Tip**: Use `show_progress=True` to see progress bar

### What are rate limits?

Cloud providers limit API requests:
- **Gemini Free tier**: 15 requests/minute
- **Gemini Tier 2**: Much higher limits

**Solutions**:
- Request higher quota from provider
- Reduce `max_workers` to slow down
- Add delays between batches
- Use local models (Ollama) - no limits!

---

## Advanced Usage

### Can I process multiple documents at once?

Yes! Pass a list of Document objects:

```python
documents = [
    lx.data.Document(text="Doc 1 text", document_id="doc1"),
    lx.data.Document(text="Doc 2 text", document_id="doc2"),
    lx.data.Document(text="Doc 3 text", document_id="doc3"),
]

results = lx.extract(
    text_or_documents=documents,
    prompt_description=prompt,
    examples=examples,
)

for doc in results:
    print(f"Document {doc.document_id}: {len(doc.extractions)} extractions")
```

### Can I extract from URLs?

Yes! LangExtract automatically downloads and processes URLs:

```python
result = lx.extract(
    text_or_documents="https://example.com/document.txt",
    prompt_description=prompt,
    examples=examples,
    fetch_urls=True,  # Default
)
```

Disable with `fetch_urls=False` to treat URLs as literal text.

### How do I save and load results?

**Save**:
```python
lx.io.save_annotated_documents([result], "output.jsonl", "./results")
```

**Load**:
```python
documents = lx.io.load_annotated_documents("output.jsonl")
```

**Visualize loaded data**:
```python
html = lx.visualize("output.jsonl")
```

### Can I customize the visualization?

Yes:

```python
html = lx.visualize(
    result,
    animation_speed=2.0,     # 2x faster animation
    show_legend=False,       # Hide legend
    gif_optimized=True,      # GIF-friendly styling
)
```

### How do I handle errors gracefully?

```python
from langextract.core.exceptions import InferenceConfigError
from langextract.resolver import ResolverParsingError

try:
    result = lx.extract(
        text_or_documents=text,
        prompt_description=prompt,
        examples=examples,
    )
except ValueError as e:
    print(f"Invalid input: {e}")
except InferenceConfigError as e:
    print(f"Model config error: {e}")
except ResolverParsingError as e:
    print(f"Failed to parse output: {e}")
    # Try with different settings
```

### Can I use LangExtract in production?

Yes! Recommendations:

1. **API key management**: Use environment variables or secret management
2. **Error handling**: Implement retry logic for API failures
3. **Cost monitoring**: Track API usage and costs
4. **Rate limits**: Handle rate limit errors gracefully
5. **Testing**: Validate on representative data before deployment
6. **Logging**: Enable debug mode during development

```python
import os
import logging

# Production configuration
result = lx.extract(
    text_or_documents=text,
    prompt_description=prompt,
    examples=examples,
    api_key=os.environ["LANGEXTRACT_API_KEY"],  # From secure source
    temperature=0.0,  # Deterministic
    debug=False,      # Disable in production
)
```

---

## Troubleshooting

### I'm getting "No provider registered for model_id"

**Cause**: Model ID not recognized

**Solutions**:
1. Check spelling: `"gemini-2.5-flash"` (with hyphens)
2. Install dependencies: `pip install langextract[openai]` for OpenAI
3. Use explicit provider:
   ```python
   config = lx.factory.ModelConfig(
       model_id="your-model",
       provider="YourProviderClass"
   )
   ```

### I'm getting "API key not found"

**Cause**: No API key provided

**Solutions**:
1. Set environment variable: `export LANGEXTRACT_API_KEY="your-key"`
2. Create `.env` file with key
3. Pass directly: `lx.extract(..., api_key="your-key")`

### Results are empty or incorrect

**Debug steps**:
1. Enable debug mode: `debug=True`
2. Check examples match your data
3. Try smaller chunks: `max_char_buffer=500`
4. Use multiple passes: `extraction_passes=3`
5. Try better model: `model_id="gemini-2.5-pro"`

### Visualization not showing in Jupyter

**Solution**: Install Jupyter dependencies:
```bash
pip install ipython
```

Or save to file:
```python
html = lx.visualize(result)
with open("viz.html", "w") as f:
    f.write(html if isinstance(html, str) else html.data)
```

### Processing is very slow

**Solutions**:
1. Increase parallelism: `max_workers=30`
2. Larger batches: `batch_length=20`
3. Use faster model: `gemini-2.5-flash`
4. Larger chunks: `max_char_buffer=1500`

### I'm hitting rate limits

**Solutions**:
1. Reduce workers: `max_workers=5`
2. Request higher quota from provider
3. Add delays between calls
4. Use local model (Ollama)

### Installation fails on Windows

**Common issue**: Build tools missing

**Solution**:
```bash
# Install Visual C++ Build Tools
# Then try again
pip install langextract
```

---

## Still Need Help?

- **[Troubleshooting Guide](troubleshooting.md)** - Detailed solutions
- **[GitHub Issues](https://github.com/google/langextract/issues)** - Report bugs
- **[Examples](examples/)** - Working code samples
- **[API Reference](api-reference.md)** - Complete documentation

---

**Can't find your question?** [Open an issue](https://github.com/google/langextract/issues) on GitHub!
