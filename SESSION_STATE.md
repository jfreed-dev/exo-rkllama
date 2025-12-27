# RKLLM Integration Session State

**Last Updated:** 2025-12-27

## Current Status

### Working Components
- **Exo with RKLLM engine** running on node1 (10.10.88.73:52415)
- **RKLLAMA server** running on node1 (10.10.88.73:8080) - OLD VERSION (d0392d7)
- **DeepSeek-R1-1.5B** model loaded and functional (~8 tok/s)
- **Qwen2.5-1.5B-Instruct** model loaded and functional (~6.5 tok/s)
- **Token caching mechanism** working
- **Prompt template extraction** working

### Test Results (Latest)
```
Model: Qwen2.5-1.5B-Instruct (RKLLM 1.1.4)
Test: What is 25 + 17?
Answer: 42 (correct)
Speed: ~6.5 tokens/sec on RK3588 NPU

Model: DeepSeek-R1-1.5B
Test: What is 25 + 17?
Answer: 42 (correct with chain-of-thought reasoning)
Speed: ~8 tokens/sec on RK3588 NPU
```

### Available Models
| Model ID | Directory | Status |
|----------|-----------|--------|
| `deepseek-r1-1.5b-rkllm` | DeepSeek-R1-1.5B | Working |
| `qwen2.5-1.5b-instruct-rkllm` | Qwen2.5-1.5B-Instruct | Working |
| `qwen2.5-1.5b-rkllm` | Qwen2.5-1.5B | Not installed |

### Resolved Issues

1. **Qwen2.5-1.5B model now working**
   - Downloaded pre-converted model from HuggingFace (c01zaut/Qwen2.5-1.5B-Instruct-RK3588-1.1.4)
   - Model converted with RKLLM toolkit 1.1.4, matching node runtime
   - Works correctly through both rkllama and exo

### Known Issues

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
