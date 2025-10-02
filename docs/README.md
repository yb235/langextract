# LangExtract Documentation

Welcome to the LangExtract documentation! This guide will help you learn how to extract structured information from unstructured text using Large Language Models.

## 📚 Documentation Overview

### For First-Time Users

Start here if you're new to LangExtract:

1. **[Getting Started Guide](getting-started.md)** ⭐ **START HERE**
   - What is LangExtract?
   - Installation and setup
   - Your first extraction
   - Basic concepts explained
   - Quick reference card

2. **[Tutorial](tutorial.md)**
   - Step-by-step walkthroughs
   - 8 hands-on examples from basic to advanced
   - Best practices
   - Real-world scenarios

3. **[FAQ](faq.md)**
   - Common questions and answers
   - Quick solutions to frequent issues
   - Tips and tricks

### Reference Documentation

Complete technical documentation:

4. **[API Reference](api-reference.md)**
   - Complete function signatures
   - All parameters explained
   - Data classes and types
   - Examples for every feature

5. **[Architecture Documentation](architecture.md)**
   - System overview
   - Core components explained
   - Workflow and data flow
   - How LangExtract works internally

6. **[Deep Dive Architecture](deepdivearchitecture.md)**
   - Detailed technical walkthrough
   - Component interactions
   - Data transformations
   - Advanced internals

### Troubleshooting

7. **[Troubleshooting Guide](troubleshooting.md)**
   - Solutions to common problems
   - Installation issues
   - API errors
   - Performance optimization
   - Provider-specific issues

### Examples

8. **[Examples Directory](examples/)**
   - Medication extraction (healthcare)
   - Long document processing (Romeo & Juliet)
   - Real-world use cases

## 🚀 Quick Start Path

**Never used LangExtract before?** Follow this learning path:

```
┌─────────────────────────────────────────────┐
│  1. Getting Started Guide (15-20 min)       │
│     - Installation                          │
│     - First extraction                      │
│     - Basic concepts                        │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│  2. Tutorial - First 3 Sections (30 min)    │
│     - Basic extraction                      │
│     - Adding attributes                     │
│     - Multiple classes                      │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│  3. Try Your Own Data! (hands-on)           │
│     - Start with simple examples            │
│     - Gradually increase complexity         │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│  4. Advanced Topics (as needed)             │
│     - Remaining tutorial sections           │
│     - Architecture docs                     │
│     - API reference                         │
└─────────────────────────────────────────────┘
```

## 📖 Documentation by Use Case

### I want to...

#### Learn the Basics
- **Install LangExtract** → [Getting Started - Installation](getting-started.md#installation)
- **Understand core concepts** → [Getting Started - Understanding the Basics](getting-started.md#understanding-the-basics)
- **Run my first extraction** → [Getting Started - Your First Extraction](getting-started.md#your-first-extraction)

#### Extract Information
- **Extract named entities** → [Tutorial - Basic Extraction](tutorial.md#1-basic-extraction)
- **Add attributes to extractions** → [Tutorial - Adding Attributes](tutorial.md#2-adding-attributes)
- **Extract multiple types** → [Tutorial - Multiple Extraction Classes](tutorial.md#3-multiple-extraction-classes)
- **Process long documents** → [Tutorial - Processing Long Documents](tutorial.md#4-processing-long-documents)

#### Work with Different Models
- **Use Gemini (recommended)** → [Getting Started - API Key Setup](getting-started.md#step-3-get-an-api-key)
- **Use OpenAI** → [FAQ - OpenAI Models](faq.md#how-do-i-use-openai-models)
- **Use local models (Ollama)** → [Tutorial - Using Local Models](tutorial.md#6-using-local-models)
- **Choose the right model** → [Getting Started - Choosing a Model](getting-started.md#choosing-a-model)

#### Handle Issues
- **Fix extraction quality** → [FAQ - Extraction Quality](faq.md#extraction-quality)
- **Improve performance** → [FAQ - Performance and Cost](faq.md#performance-and-cost)
- **Debug errors** → [Troubleshooting Guide](troubleshooting.md)
- **Reduce costs** → [FAQ - Reduce Costs](faq.md#how-can-i-reduce-costs)

#### Advanced Usage
- **Batch process documents** → [Tutorial - Batch Processing](tutorial.md#7-batch-processing-multiple-documents)
- **Improve recall** → [Tutorial - Multiple Passes](tutorial.md#8-improving-recall-with-multiple-passes)
- **Understand architecture** → [Architecture Documentation](architecture.md)
- **Create custom providers** → [Provider System README](../langextract/providers/README.md)

#### Reference and API
- **Look up a parameter** → [API Reference](api-reference.md)
- **Understand data structures** → [API Reference - Data Classes](api-reference.md#data-classes)
- **Configure extraction** → [API Reference - Core Functions](api-reference.md#core-functions)

## 🎯 Key Concepts

Before diving in, understand these core concepts:

### 1. Examples Drive Extraction
LangExtract uses **few-shot learning**. You provide examples of what you want to extract, and the AI learns from them:

```python
examples = [
    lx.data.ExampleData(
        text="Input example text",
        extractions=[
            lx.data.Extraction(
                extraction_class="what_type",
                extraction_text="what_to_extract",
                attributes={"extra": "info"}
            )
        ]
    )
]
```

### 2. Source Grounding
Every extraction is mapped back to its exact position in the source text:

```python
extraction.extraction_text  # "Aspirin"
extraction.char_interval    # CharInterval(45, 52)
source_text[45:52]          # "Aspirin"
```

### 3. Automatic Chunking
Long documents are automatically split into manageable pieces and processed in parallel.

### 4. Multiple Passes
For better recall, you can process documents multiple times:

```python
lx.extract(..., extraction_passes=3)  # Process 3 times
```

### 5. Provider Flexibility
Use different LLM providers based on your needs:
- **Cloud**: Gemini, OpenAI
- **Local**: Ollama (free, private, offline)
- **Custom**: Create your own provider plugin

## 🔗 External Resources

- **[GitHub Repository](https://github.com/google/langextract)** - Source code and issues
- **[PyPI Package](https://pypi.org/project/langextract/)** - Installation
- **[Main README](../README.md)** - Project overview
- **[Contributing Guide](../CONTRIBUTING.md)** - How to contribute
- **[License](../LICENSE)** - Apache 2.0

## 📝 Documentation Structure

```
docs/
├── README.md                    # This file - Documentation index
├── getting-started.md          # First steps and installation
├── tutorial.md                 # Step-by-step examples
├── api-reference.md            # Complete API documentation
├── faq.md                      # Frequently asked questions
├── troubleshooting.md          # Problem solving guide
├── architecture.md             # System architecture overview
├── deepdivearchitecture.md     # Detailed technical deep dive
└── examples/                   # Example use cases
    ├── longer_text_example.md  # Romeo & Juliet full text
    ├── medication_examples.md  # Healthcare extraction
    └── ...
```

## 🆘 Getting Help

### Quick Help
1. Check [FAQ](faq.md) for common questions
2. Review [Troubleshooting Guide](troubleshooting.md) for specific errors
3. Look at [Examples](examples/) for working code

### Reporting Issues
- **Bug reports**: [GitHub Issues](https://github.com/google/langextract/issues)
- **Feature requests**: [GitHub Issues](https://github.com/google/langextract/issues)
- **Questions**: [GitHub Discussions](https://github.com/google/langextract/discussions) (if available)

When reporting issues, include:
- LangExtract version (`pip show langextract`)
- Python version (`python --version`)
- Minimal reproducible example
- Complete error traceback

## 📊 Documentation Status

| Document | Status | Audience |
|----------|--------|----------|
| Getting Started | ✅ Complete | Beginners |
| Tutorial | ✅ Complete | All users |
| API Reference | ✅ Complete | Developers |
| FAQ | ✅ Complete | All users |
| Troubleshooting | ✅ Complete | All users |
| Architecture | ✅ Complete | Advanced users |
| Deep Dive | ✅ Complete | Contributors |

## 🤝 Contributing to Documentation

Found an error or want to improve the docs?

1. **Small fixes**: Edit directly on GitHub and submit a PR
2. **Large changes**: Open an issue first to discuss
3. **Follow style**: Match the tone and format of existing docs
4. **Test examples**: Ensure all code examples work

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

---

## Quick Links Summary

### For Learning
- [Getting Started](getting-started.md) - Start here!
- [Tutorial](tutorial.md) - Hands-on examples
- [Examples](examples/) - Real use cases

### For Reference
- [API Reference](api-reference.md) - Complete API docs
- [Architecture](architecture.md) - How it works
- [FAQ](faq.md) - Quick answers

### For Help
- [Troubleshooting](troubleshooting.md) - Fix issues
- [GitHub Issues](https://github.com/google/langextract/issues) - Report bugs
- [Contributing](../CONTRIBUTING.md) - Get involved

---

**Ready to start?** Go to the [Getting Started Guide](getting-started.md)! 🚀
