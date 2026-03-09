# CosyVoice3 vLLM Zero-Shot Adapter

This adapter exposes a low-latency JSON TTS API for realtime callers such as Dynamic_MCP.

It is designed for CosyVoice3 and uses zero-shot speaker profiles instead of SFT speaker ids.

## Endpoints

- `GET /healthz`
- `GET /readyz`
- `GET /v1/voices`
- `POST /v1/tts`
- `POST /v1/tts/pcm16`

Both TTS endpoints return `audio/pcm` with signed 16-bit mono PCM at the model sample rate.

## Request Example

```json
{
  "text": "您好，这里是智能客服。",
  "voice": "customer_service_cn_female",
  "model": "Fun-CosyVoice3-0.5B-2512",
  "sample_rate": 22050,
  "stream": true
}
```

## Speaker Profiles

Create a JSON file such as `runtime/python/vllm_adapter/speakers.json`:

```json
{
  "voices": {
    "customer_service_cn_female": {
      "prompt_text": "You are a helpful assistant.<|endofprompt|>希望以后你能做的比我还好哦。",
      "prompt_wav": "/models/prompts/customer_service_cn_female.wav"
    },
    "customer_service_cn_male": {
      "prompt_text": "You are a helpful assistant.<|endofprompt|>您好，这里是智能客服，很高兴为您服务。",
      "prompt_wav": "/models/prompts/customer_service_cn_male.wav"
    }
  }
}
```

Each voice must provide:
- `prompt_text`
- `prompt_wav`

## Environment Variables

- `COSYVOICE_MODEL_DIR`: local model directory
- `COSYVOICE_API_TOKEN`: optional bearer token
- `COSYVOICE_LOAD_VLLM`: defaults to `true`
- `COSYVOICE_LOAD_TRT`: defaults to `false`
- `COSYVOICE_FP16`: defaults to `false`
- `COSYVOICE_DEFAULT_SPEED`: defaults to `1.0`
- `COSYVOICE_DEFAULT_VOICE`: optional fallback voice key
- `COSYVOICE_WARMUP_TEXT`: warmup text run once at startup
- `COSYVOICE_SPEAKERS_JSON`: path to your speaker profile JSON file

## Run Locally

```bash
pip install -r requirements.txt
pip install -r runtime/python/vllm_adapter/requirements.txt

export COSYVOICE_MODEL_DIR=/models/Fun-CosyVoice3-0.5B
export COSYVOICE_API_TOKEN=replace-me
export COSYVOICE_DEFAULT_VOICE=customer_service_cn_female
export COSYVOICE_SPEAKERS_JSON=/opt/CosyVoice/runtime/python/vllm_adapter/speakers.json

uvicorn runtime.python.vllm_adapter.app:app --host 0.0.0.0 --port 8001
```

## Dynamic_MCP Configuration

Use the project voice config below:

- `tts_provider=cosyvoice`
- `tts_api_url=http://cosyvoice3-vllm-adapter:8001`
- `tts_model=Fun-CosyVoice3-0.5B-2512`
- `tts_voice=customer_service_cn_female`

The Dynamic_MCP voice agent will call `/v1/tts` for connection tests and `/v1/tts/pcm16` for realtime streaming.
