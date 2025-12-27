# RKLLM Integration Session State

**Last Updated:** 2025-12-27

## Current Status

### Working Components
- **Exo with RKLLM engine** running on node1 (10.10.88.73:52415)
- **RKLLAMA server** running on node1 (10.10.88.73:8080) - OLD VERSION (d0392d7)
- **DeepSeek-R1-1.5B** model loaded and functional (~8 tok/s)
- **Token caching mechanism** working
- **Prompt template extraction** working

### Test Results (Latest)
```
Test: What is 25 + 17?
Answer: 42 (correct with chain-of-thought reasoning)
Speed: ~8 tokens/sec on RK3588 NPU
```

### Known Issues

1. **Qwen2.5-1.5B model incompatible**
   - Downloaded from HuggingFace (Pelochus/qwen2-1_5B-rk3588)
   - Converted with RKLLM runtime 1.0.1
   - Node has runtime 1.2.3 in new rkllama, but old rkllama works with DeepSeek

2. **New rkllama (upstream) incompatible**
   - Updated to upstream (004dc88) - 142 commits ahead
   - New runtime 1.2.3 causes DeepSeek to output garbage (&&&&&)
   - Reverted to old version (d0392d7) which works

## Node Configuration

### Node 1 (10.10.88.73)
- **SSH Access:** `ssh -i ~/.ssh/workbench root@10.10.88.73`
- **Exo:** `/opt/exo` (jfreed-dev/exo-rkllama)
- **RKLLAMA:** `/opt/rkllama` (reverted to d0392d7)
- **Models:** `/root/RKLLAMA/models/`
  - `DeepSeek-R1-1.5B/` - Working
  - `Qwen2.5-1.5B/` - Downloaded but incompatible

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

# Through Exo API
curl -s http://10.10.88.73:52415/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "deepseek-r1-1.5b-rkllm", "messages": [{"role": "user", "content": "What is 2+2?"}]}'
```

## Pending Tasks

1. **Get Qwen model working**
   - Need a Qwen model converted with matching RKLLM toolkit version
   - Options:
     - Find pre-converted model for older runtime
     - Convert model ourselves using RKLLM-Toolkit on x86
     - Wait for rkllama fix for newer runtime

2. **Fix rkllama upstream compatibility**
   - New version (1.2.3 runtime) causes DeepSeek output corruption
   - May need to report issue upstream or find compatible model

## Git Status
```
Latest commit: 7d89a1c Revert HTTP client to use /generate endpoint (old rkllama)
```

## Quick Resume Commands

```bash
# Check if services are running
curl -s http://10.10.88.73:8080/models
curl -s http://10.10.88.73:52415/v1/models | head -1

# Reload model if needed
curl -s -X POST http://10.10.88.73:8080/load_model \
  -H "Content-Type: application/json" \
  -d '{"model_name": "DeepSeek-R1-1.5B"}'
```
