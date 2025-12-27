# RKLLM Integration Session State

**Last Updated:** 2025-12-27

## Current Status

### Working Components
- **Exo with RKLLM engine** running on node1 (10.10.88.73:52415)
- **RKLLAMA server** running on node1 (10.10.88.73:8080) - OLD VERSION (d0392d7)
- **DeepSeek-R1-1.5B** model loaded and functional (~8 tok/s)
- **Qwen2.5-1.5B-Instruct** model loaded and functional (~7.8 tok/s)
- **Token caching mechanism** working
- **Prompt template extraction** working

### Available Models
| Model ID | Directory | Status |
|----------|-----------|--------|
| `deepseek-r1-1.5b-rkllm` | DeepSeek-R1-1.5B | Working |
| `qwen2.5-1.5b-instruct-rkllm` | Qwen2.5-1.5B-Instruct | Working |
| `qwen2.5-1.5b-rkllm` | Qwen2.5-1.5B | Not installed |

## Benchmark Results

### Performance Comparison (RK3588 NPU)

#### Qwen2.5-1.5B-Instruct (w8a8)
| Test            | Tokens/sec | Total Tokens | Completion Tokens |
|-----------------|------------|--------------|-------------------|
| Math            | 7.23       | 56           | 23                |
| Explanation     | 7.81       | 57           | 29                |
| Code Generation | 8.21       | 189          | 156               |
| Reasoning       | 8.09       | 253          | 222               |
| Creative        | 7.48       | 54           | 26                |
| **AVERAGE**     | **7.76**   | **122**      | **91**            |

#### DeepSeek-R1-1.5B (chain-of-thought)
| Test            | Tokens/sec | Total Tokens | Completion Tokens |
|-----------------|------------|--------------|-------------------|
| Math            | 7.95       | 534          | 501               |
| Explanation     | 7.75       | 1027         | 999               |
| Code Generation | 7.92       | 674          | 641               |
| Reasoning       | 7.86       | 659          | 628               |
| Creative        | 8.07       | 240          | 212               |
| **AVERAGE**     | **7.91**   | **627**      | **596**           |

### Summary Comparison
| Metric | Qwen2.5-1.5B-Instruct | DeepSeek-R1-1.5B |
|--------|----------------------|------------------|
| **Avg Speed** | 7.76 tok/s | 7.91 tok/s |
| **Avg Completion Tokens** | 91 | 596 |
| **Token Efficiency** | 6.5x fewer tokens | Verbose |
| **Response Style** | Concise, direct | Chain-of-thought reasoning |
| **Best For** | Quick answers, APIs | Explanations, reasoning tasks |

### Key Findings
1. **Speed**: Both models perform similarly at ~7.8-8.0 tokens/sec
2. **Token Efficiency**: Qwen generates 6.5x fewer tokens for same prompts
3. **Response Time**: Qwen ~3-32s, DeepSeek ~30-130s (more tokens)
4. **Use Cases**:
   - Qwen: Chatbots, APIs, quick Q&A, production
   - DeepSeek: Educational content, complex problem-solving

### Sample Responses

**Math (145 + 278):**
- Qwen: `Solution: $$ 145 + 278 = \boxed{423} $$` (23 tokens)
- DeepSeek: Full step-by-step carry-over explanation (501 tokens)

**Haiku about AI:**
- Qwen: `Code writes, learns, / Silent hands of data flow, / Future in its grasp.`
- DeepSeek: Extended explanation with interpretation of each line

## Resolved Issues

1. **Qwen2.5-1.5B model now working**
   - Downloaded pre-converted model from HuggingFace (c01zaut/Qwen2.5-1.5B-Instruct-RK3588-1.1.4)
   - Model converted with RKLLM toolkit 1.1.4, matching node runtime
   - Works correctly through both rkllama and exo

## Known Issues

1. **New rkllama (upstream) incompatible**
   - Updated to upstream (004dc88) - 142 commits ahead
   - New runtime 1.2.3 causes DeepSeek to output garbage (&&&&&)
   - Reverted to old version (d0392d7) which works

## Node Configuration

### Node 1 (10.10.88.73)
- **SSH Access:** `ssh -i ~/.ssh/workbench root@10.10.88.73`
- **Exo:** `/opt/exo` (jfreed-dev/exo-rkllama)
- **RKLLAMA:** `/opt/rkllama` (reverted to d0392d7, runtime 1.1.4)
- **Models:** `/root/RKLLAMA/models/`
  - `DeepSeek-R1-1.5B/` - Working
  - `Qwen2.5-1.5B-Instruct/` - Working (1.9GB, w8a8 quantized)

### Starting Services
```bash
# On node (via SSH)
# RKLLAMA (old version)
cd /opt/rkllama && source venv/bin/activate
python server.py --target_platform rk3588 --port 8080

# EXO
cd /opt/exo && source venv/bin/activate
RKLLM_SERVER_HOST=localhost RKLLM_SERVER_PORT=8080 \
  python -m exo.main --inference-engine rkllm --disable-tui
```

### Testing
```bash
# Direct RKLLAMA test
curl -s http://10.10.88.73:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is 2+2?"}], "stream": false}'

# Through Exo API (DeepSeek)
curl -s http://10.10.88.73:52415/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "deepseek-r1-1.5b-rkllm", "messages": [{"role": "user", "content": "What is 2+2?"}]}'

# Through Exo API (Qwen)
curl -s http://10.10.88.73:52415/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen2.5-1.5b-instruct-rkllm", "messages": [{"role": "user", "content": "What is 2+2?"}]}'
```

## Pending Tasks

1. **Fix rkllama upstream compatibility**
   - New version (1.2.3 runtime) causes DeepSeek output corruption
   - May need to report issue upstream or find compatible model

2. **Convert more models**
   - Can use RKLLM-Toolkit 1.1.4 on x86_64 to convert additional models
   - Toolkit available at: https://github.com/airockchip/rknn-llm (release-v1.1.4 tag)

## Quick Resume Commands

```bash
# Check if services are running
curl -s http://10.10.88.73:8080/models
curl -s http://10.10.88.73:52415/v1/models | head -1

# Switch to Qwen model
curl -s -X POST http://10.10.88.73:8080/unload_model && \
curl -s -X POST http://10.10.88.73:8080/load_model \
  -H "Content-Type: application/json" \
  -d '{"model_name": "Qwen2.5-1.5B-Instruct"}'

# Switch to DeepSeek model
curl -s -X POST http://10.10.88.73:8080/unload_model && \
curl -s -X POST http://10.10.88.73:8080/load_model \
  -H "Content-Type: application/json" \
  -d '{"model_name": "DeepSeek-R1-1.5B"}'
```

## Model Sources

- **Qwen2.5-1.5B-Instruct**: https://huggingface.co/c01zaut/Qwen2.5-1.5B-Instruct-RK3588-1.1.4
- **DeepSeek-R1-1.5B**: Pre-installed with old rkllama
