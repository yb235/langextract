# Troubleshooting Guide

Solutions to common problems when using LangExtract.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Configuration Problems](#configuration-problems)
- [API and Authentication Errors](#api-and-authentication-errors)
- [Extraction Quality Issues](#extraction-quality-issues)
- [Performance Problems](#performance-problems)
- [Visualization Issues](#visualization-issues)
- [Provider-Specific Issues](#provider-specific-issues)

---

## Installation Issues

### Problem: `pip install langextract` fails

**Symptoms:**
```
ERROR: Could not find a version that satisfies the requirement langextract
```

**Solutions:**

1. **Check Python version**:
   ```bash
   python --version  # Should be 3.10 or higher
   ```
   
   If too old, upgrade Python or use `python3.10`, `python3.11`, etc.

2. **Upgrade pip**:
   ```bash
   pip install --upgrade pip
   ```

3. **Use Python 3.10+ explicitly**:
   ```bash
   python3.11 -m pip install langextract
   ```

### Problem: OpenAI support not working

**Symptoms:**
```
InferenceConfigError: OpenAI provider requires openai package
```

**Solution:**
```bash
pip install langextract[openai]
```

### Problem: Build errors on Windows

**Symptoms:**
```
error: Microsoft Visual C++ 14.0 or greater is required
```

**Solution:**

1. Install [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. Or use pre-built wheels:
   ```bash
   pip install --only-binary :all: langextract
   ```

### Problem: Import error after installation

**Symptoms:**
```python
ImportError: No module named 'langextract'
```

**Solutions:**

1. **Check virtual environment**:
   ```bash
   which python  # Should point to venv
   ```

2. **Reinstall in correct environment**:
   ```bash
   pip uninstall langextract
   pip install langextract
   ```

3. **Check installation**:
   ```bash
   pip show langextract
   ```

---

## Configuration Problems

### Problem: "No provider registered for model_id"

**Symptoms:**
```
ValueError: No provider registered for model_id='gpt-4o'
```

**Cause:** Missing optional dependencies or typo in model ID

**Solutions:**

1. **For OpenAI models**:
   ```bash
   pip install langextract[openai]
   ```

2. **Check model ID spelling**:
   ```python
   # Correct
   model_id="gemini-2.5-flash"  # Note: hyphens, not underscores
   
   # Wrong
   model_id="gemini_2_5_flash"
   ```

3. **Use explicit provider**:
   ```python
   config = lx.factory.ModelConfig(
       model_id="gpt-4o",
       provider="OpenAILanguageModel"
   )
   model = lx.factory.create_model(config)
   ```

4. **List available providers**:
   ```python
   import langextract as lx
   lx.providers.registry.list_entries()
   ```

### Problem: Examples validation fails

**Symptoms:**
```
ValueError: examples cannot be None or empty
```

**Solution:**

Provide at least one example:
```python
examples = [
    lx.data.ExampleData(
        text="Sample text",
        extractions=[
            lx.data.Extraction(
                extraction_class="type",
                extraction_text="text"
            )
        ]
    )
]
```

### Problem: Invalid parameter error

**Symptoms:**
```
TypeError: extract() got an unexpected keyword argument 'xxx'
```

**Solution:**

Check parameter spelling in [API Reference](api-reference.md). Common mistakes:
```python
# Wrong
lx.extract(..., max_workers=10)  # Correct parameter name

# Common typos
lx.extract(..., max_worker=10)    # Wrong (missing 's')
lx.extract(..., maxworkers=10)    # Wrong (missing underscore)
```

---

## API and Authentication Errors

### Problem: "API key not found"

**Symptoms:**
```
ValueError: No API key provided and LANGEXTRACT_API_KEY environment variable not set
```

**Solutions:**

1. **Set environment variable**:
   ```bash
   export LANGEXTRACT_API_KEY="your-key-here"
   ```

2. **Create .env file**:
   ```bash
   echo "LANGEXTRACT_API_KEY=your-key" > .env
   ```

3. **Pass directly** (development only):
   ```python
   lx.extract(..., api_key="your-key")
   ```

4. **Check variable is set**:
   ```bash
   echo $LANGEXTRACT_API_KEY
   ```

5. **For Vertex AI** (service accounts):
   ```python
   lx.extract(
       ...,
       language_model_params={
           "vertexai": True,
           "project": "your-project-id",
           "location": "us-central1"
       }
   )
   ```

### Problem: Authentication failed

**Symptoms:**
```
google.api_core.exceptions.Unauthenticated: 401 Invalid API key
```

**Solutions:**

1. **Verify API key is valid**:
   - Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
   - Generate new key if needed

2. **Check for extra spaces**:
   ```bash
   # Remove any quotes/spaces
   export LANGEXTRACT_API_KEY=AIzaSy...  # No quotes
   ```

3. **Use correct environment variable**:
   ```python
   # Gemini accepts both
   GEMINI_API_KEY="your-key"
   # or
   LANGEXTRACT_API_KEY="your-key"
   ```

### Problem: Rate limit exceeded

**Symptoms:**
```
google.api_core.exceptions.ResourceExhausted: 429 Quota exceeded
```

**Solutions:**

1. **Reduce parallelism**:
   ```python
   lx.extract(
       ...,
       max_workers=5,      # Lower from default 10
       batch_length=5,     # Lower from default 10
   )
   ```

2. **Request higher quota**:
   - Visit [Google AI Studio](https://aistudio.google.com/)
   - Request Tier 2 quota increase

3. **Add delays** (if needed):
   ```python
   import time
   
   for doc in documents:
       result = lx.extract(...)
       time.sleep(1)  # Wait between calls
   ```

4. **Use local model** (no limits):
   ```python
   lx.extract(..., model_id="gemma2:2b")
   ```

### Problem: Network/timeout errors

**Symptoms:**
```
requests.exceptions.Timeout: Request timed out
```

**Solutions:**

1. **Check internet connection**

2. **Retry with exponential backoff**:
   ```python
   import time
   
   for attempt in range(3):
       try:
           result = lx.extract(...)
           break
       except Exception as e:
           if attempt < 2:
               time.sleep(2 ** attempt)  # 1s, 2s, 4s
           else:
               raise
   ```

3. **Use local model** (offline):
   ```bash
   ollama serve
   ```
   ```python
   lx.extract(..., model_id="gemma2:2b")
   ```

---

## Extraction Quality Issues

### Problem: No extractions found

**Symptoms:**
```python
len(result.extractions) == 0
```

**Debug steps:**

1. **Enable debug mode**:
   ```python
   result = lx.extract(..., debug=True)
   ```

2. **Check examples match your data**:
   ```python
   # Examples should be similar to input
   examples = [
       lx.data.ExampleData(
           text="Similar to your actual input",
           extractions=[...]
       )
   ]
   ```

3. **Try smaller chunks**:
   ```python
   lx.extract(
       ...,
       max_char_buffer=500,  # Smaller than default 1000
   )
   ```

4. **Use multiple passes**:
   ```python
   lx.extract(
       ...,
       extraction_passes=3,
   )
   ```

5. **Try better model**:
   ```python
   lx.extract(..., model_id="gemini-2.5-pro")
   ```

### Problem: Missing extractions (low recall)

**Symptoms:** Some entities not extracted

**Solutions:**

1. **Multiple passes** (best solution):
   ```python
   lx.extract(
       ...,
       extraction_passes=3,  # Process 3 times
   )
   ```

2. **Smaller chunks**:
   ```python
   lx.extract(
       ...,
       max_char_buffer=800,  # From 1000
   )
   ```

3. **Add more diverse examples**:
   ```python
   examples = [
       # Example 1: Simple case
       lx.data.ExampleData(...),
       # Example 2: Complex case
       lx.data.ExampleData(...),
       # Example 3: Edge case
       lx.data.ExampleData(...),
   ]
   ```

4. **Clarify prompt**:
   ```python
   # Vague
   prompt = "Extract things"
   
   # Specific
   prompt = "Extract all person names, including nicknames and titles"
   ```

### Problem: False positives (wrong extractions)

**Symptoms:** Extracting things that shouldn't be extracted

**Solutions:**

1. **Make prompt more specific**:
   ```python
   prompt = """
   Extract ONLY medications that are explicitly prescribed.
   Do NOT extract:
   - Over-the-counter drugs mentioned casually
   - Medical procedures
   - Conditions or symptoms
   """
   ```

2. **Set deterministic output**:
   ```python
   lx.extract(..., temperature=0.0)
   ```

3. **Add negative examples** (show what NOT to extract):
   ```python
   prompt = """
   Extract medication names.
   
   CORRECT: "Lisinopril 10mg"
   INCORRECT: "feeling dizzy" (symptom, not medication)
   """
   ```

4. **Review and refine examples**:
   - Ensure examples only show what you want
   - Remove ambiguous cases

### Problem: Alignment failures

**Symptoms:**
```python
extraction.alignment_status == "MATCH_FUZZY"
# or extractions have no char_interval
```

**Causes:**
- Model paraphrasing instead of using exact text
- OCR or encoding issues in source text

**Solutions:**

1. **Emphasize exact text in prompt**:
   ```python
   prompt = "Extract entities using EXACT text from the document. Do not paraphrase."
   ```

2. **Enable fuzzy alignment**:
   ```python
   lx.extract(
       ...,
       resolver_params={
           "enable_fuzzy_alignment": True,
           "fuzzy_alignment_threshold": 0.8,
       }
   )
   ```

3. **Check source text encoding**:
   ```python
   # Ensure UTF-8
   text = text.encode('utf-8', errors='ignore').decode('utf-8')
   ```

### Problem: Inconsistent results

**Symptoms:** Different results on same input

**Cause:** Non-deterministic model output

**Solution:**

Set temperature to 0:
```python
lx.extract(
    ...,
    temperature=0.0,  # Deterministic
)
```

---

## Performance Problems

### Problem: Processing is very slow

**Symptoms:** Takes minutes for small documents

**Solutions:**

1. **Increase parallelism**:
   ```python
   lx.extract(
       ...,
       max_workers=30,      # From 10
       batch_length=20,     # From 10
   )
   ```

2. **Use faster model**:
   ```python
   # gemini-2.5-flash is faster than gemini-2.5-pro
   lx.extract(..., model_id="gemini-2.5-flash")
   ```

3. **Larger chunks** (if accuracy allows):
   ```python
   lx.extract(
       ...,
       max_char_buffer=1500,  # From 1000
   )
   ```

4. **Single pass**:
   ```python
   lx.extract(
       ...,
       extraction_passes=1,  # Default
   )
   ```

5. **Check network latency**:
   - Use regional endpoint if available
   - Consider local model (Ollama)

### Problem: High memory usage

**Symptoms:** Out of memory errors

**Solutions:**

1. **Process documents one at a time**:
   ```python
   for doc in large_document_list:
       result = lx.extract(text_or_documents=[doc], ...)
       # Process result
       del result  # Free memory
   ```

2. **Reduce batch size**:
   ```python
   lx.extract(
       ...,
       batch_length=5,  # From 10
   )
   ```

3. **Smaller chunks**:
   ```python
   lx.extract(
       ...,
       max_char_buffer=500,  # From 1000
   )
   ```

### Problem: High costs

**Symptoms:** API bills higher than expected

**Solutions:**

1. **Estimate costs first**:
   ```python
   # Test on small sample
   sample = text[:1000]
   result = lx.extract(text_or_documents=sample, ...)
   # Extrapolate cost
   ```

2. **Use cheaper model**:
   ```python
   # gemini-2.5-flash is ~16x cheaper than Pro
   lx.extract(..., model_id="gemini-2.5-flash")
   ```

3. **Larger chunks**:
   ```python
   lx.extract(
       ...,
       max_char_buffer=1500,  # Fewer API calls
   )
   ```

4. **Single pass**:
   ```python
   lx.extract(
       ...,
       extraction_passes=1,  # Don't use 3
   )
   ```

5. **Use local model** (free):
   ```bash
   ollama serve
   ```
   ```python
   lx.extract(..., model_id="gemma2:2b")
   ```

---

## Visualization Issues

### Problem: Visualization not displaying in Jupyter

**Symptoms:** Nothing shown when running in notebook

**Solution:**

1. **Install Jupyter dependencies**:
   ```bash
   pip install ipython jupyter
   ```

2. **Save to file instead**:
   ```python
   html = lx.visualize(result)
   with open("viz.html", "w") as f:
       if hasattr(html, 'data'):
           f.write(html.data)
       else:
           f.write(html)
   ```

### Problem: Empty visualization

**Symptoms:** Visualization loads but shows "No extractions to animate"

**Cause:** No extractions or all failed alignment

**Solutions:**

1. **Check extractions exist**:
   ```python
   print(f"Found {len(result.extractions)} extractions")
   for e in result.extractions:
       print(e.extraction_text, e.char_interval)
   ```

2. **Enable fuzzy alignment**:
   ```python
   lx.extract(
       ...,
       resolver_params={"enable_fuzzy_alignment": True}
   )
   ```

### Problem: Visualization file too large

**Symptoms:** Browser struggles to load HTML file

**Solution:**

Filter extractions before visualizing:
```python
# Take first 100 extractions
filtered = lx.data.AnnotatedDocument(
    document_id=result.document_id,
    text=result.text,
    extractions=result.extractions[:100]
)
html = lx.visualize(filtered)
```

---

## Provider-Specific Issues

### Gemini Issues

**Problem: "Model not found"**

**Symptoms:**
```
google.api_core.exceptions.NotFound: 404 Model not found
```

**Solutions:**

1. **Check model name**:
   ```python
   # Correct
   model_id="gemini-2.5-flash"
   
   # Wrong
   model_id="gemini-flash-2.5"
   ```

2. **Model may be deprecated**:
   - Check [official model versions](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions)
   - Use current stable version

### OpenAI Issues

**Problem: "Invalid API key"**

**Solution:**

Use correct environment variable:
```bash
export OPENAI_API_KEY="sk-..."
```

**Problem: Schema constraints not working**

**Symptoms:** Unexpected output format

**Solution:**

Disable schema constraints for OpenAI:
```python
lx.extract(
    ...,
    model_id="gpt-4o",
    fence_output=True,
    use_schema_constraints=False,  # Important for OpenAI
)
```

### Ollama Issues

**Problem: "Connection refused"**

**Symptoms:**
```
requests.exceptions.ConnectionError: Connection refused
```

**Solutions:**

1. **Start Ollama**:
   ```bash
   ollama serve
   ```

2. **Check URL**:
   ```python
   lx.extract(
       ...,
       model_id="gemma2:2b",
       model_url="http://localhost:11434",  # Default
   )
   ```

3. **Verify Ollama is running**:
   ```bash
   curl http://localhost:11434/api/tags
   ```

**Problem: "Model not found"**

**Solution:**

Pull model first:
```bash
ollama pull gemma2:2b
ollama list  # Verify it's installed
```

**Problem: Low quality results with Ollama**

**Solutions:**

1. **Use better model**:
   ```bash
   ollama pull llama3  # Larger model
   ```

2. **Provide more examples**:
   ```python
   examples = [ex1, ex2, ex3, ex4]  # 3-4 examples
   ```

3. **Consider using cloud model for critical tasks**

---

## Getting More Help

### Enable Debug Mode

Get detailed information:
```python
result = lx.extract(
    ...,
    debug=True,
)
```

### Check Logs

```python
import logging
logging.basicConfig(level=logging.DEBUG)

result = lx.extract(...)
```

### Minimal Reproducible Example

When reporting issues, provide:

```python
import langextract as lx

# Minimal example
prompt = "Extract names"
examples = [
    lx.data.ExampleData(
        text="John works here",
        extractions=[lx.data.Extraction("name", "John")]
    )
]

result = lx.extract(
    text_or_documents="Alice went home",
    prompt_description=prompt,
    examples=examples,
    model_id="gemini-2.5-flash",
)

print(f"Result: {len(result.extractions)} extractions")
```

### Report Issues

- **[GitHub Issues](https://github.com/google/langextract/issues)** - Report bugs
- Include:
  - LangExtract version: `pip show langextract`
  - Python version: `python --version`
  - Operating system
  - Complete error traceback
  - Minimal reproducible example

---

## See Also

- **[FAQ](faq.md)** - Common questions
- **[API Reference](api-reference.md)** - Complete documentation
- **[Getting Started](getting-started.md)** - Setup guide
- **[Tutorial](tutorial.md)** - Step-by-step examples
