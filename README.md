# Models API Inference

A comprehensive inference platform for deploying multiple ASR (Automatic Speech Recognition) models using vLLM, supporting various state-of-the-art models like Whisper, Whisper-X, and Qwen ASR.

## Overview

This project provides:
- **Multi-model ASR support**: Whisper, Whisper-X, Qwen ASR, and more
- **vLLM inference backend**: High-performance model serving with vLLM
- **Docker Compose setup**: Easy containerized deployment
- **LiteLLM integration**: Unified API for model serving
- **Configuration-driven**: YAML-based model and service configuration

## Supported Models

- **Whisper MS Precise**: Malaysian Whisper Large v3 Turbo
- **WhisperX**: Large v3 model with cross-attention
- **Qwen ASR**: Qwen3-ASR-1.7B
- **FireRed ASR**: Specialized ASR model

## Installation

### Prerequisites
- Python 3.9+
- Docker and Docker Compose (for containerized deployment)
- CUDA-capable GPU (recommended for optimal performance)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/models-api-inference.git
cd models-api-inference
```

2. **Create a virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure models**
Edit `config.yaml` to specify which models to deploy and their configurations:
```yaml
model_list:
  - model_name: whisperx
    litellm_params:
      model: hosted_vllm/large-v3
      api_base: http://whisperx-srv:7090/v1
      api_key: "sk-1234"
```

## Usage

### Starting Models with Docker Compose

```bash
python start_models.py --gpu-ids 1,2,3
```

This will:
- Start specified ASR models in Docker containers
- Configure vLLM inference backends
- Set up communication networks between services

### Running Inference Requests

```bash
# Test with LiteLLM
python test_litellm.py

# Custom requests
python qwen_requests.py
```

### API Configuration

Models are served via LiteLLM with REST API endpoints. Each model gets:
- Dedicated vLLM server instance
- Custom API base endpoint
- Model-specific parameters

## Project Structure

```
models-api-inference/
├── models/                    # Model implementations
│   ├── qwen_asr.py
│   ├── whisper_x.py
│   ├── whisper_lev.py
│   ├── gemini.py
│   ├── pyannote_vad.py
│   └── constructor/           # Model constructors
├── models_benchmark/          # Benchmarking scripts
├── vllm_inference_models/     # Downloaded model weights
├── config.yaml               # Model configuration
├── config.json               # Alternative JSON config
├── docker-compose.yml        # Docker services
├── start_models.py           # Model startup script
├── test_litellm.py          # LiteLLM test client
└── README.md                # This file
```

## Configuration

### config.yaml

Main configuration file for model deployment:

```yaml
model_list:
  - model_name: <name>
    litellm_params:
      model: <model_identifier>
      api_base: <vllm_server_url>
      api_key: <api_key>

general_settings:
  disable_auth: <true/false>
  disable_database: <true/false>
```

### Environment Variables

Create a `.env` file for sensitive configuration:
```
API_KEY=your_api_key_here
CUDA_VISIBLE_DEVICES=0,1,2
LOG_LEVEL=INFO
```

## Development

### Running Benchmarks

```bash
cd models_benchmark
python qwen_benchmark_requests.py
```

### Testing

```bash
python test_litellm.py
```

## Docker Deployment

The project uses Docker Compose for containerized model serving:

```bash
# Build and start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Performance Optimization

- **GPU Allocation**: Specify GPU IDs in `start_models.py` for optimal resource utilization
- **Batch Size**: Configure via model parameters for throughput optimization
- **Caching**: Enable KV-cache optimization in vLLM configs
- **Quantization**: Support for model quantization (if enabled)

## Troubleshooting

### Model Won't Start
- Check Docker is running and GPU access is available
- Verify `config.yaml` syntax
- Check logs: `docker-compose logs model-name`

### API Connection Errors
- Ensure vLLM server is running on the configured port
- Verify network connectivity between services
- Check firewall rules for port access

### Out of Memory
- Reduce batch size in configuration
- Use smaller model variants
- Enable quantization if available

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License

[MIT License](LICENSE) - See LICENSE file for details

## Citation

If you use this project in your research, please cite:

```bibtex
@software{models_api_inference,
  title={Models API Inference: Multi-Model ASR Inference Platform},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/models-api-inference}
}
```

## Acknowledgments

- vLLM inference engine
- LiteLLM API standardization
- Hugging Face model hub
- Whisper, Whisper-X, and Qwen model creators

## Support

For issues, questions, or feedback:
- Open an issue on GitHub
- Check existing documentation
- Review benchmarks and configurations

---

**Last Updated**: March 2026
