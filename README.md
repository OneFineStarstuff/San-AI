# AGI Pipeline

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14504697.svg)](https://doi.org/10.5281/zenodo.14504697)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

The AGI Pipeline is an enterprise-grade multimodal AGI system integrating Natural Language Processing (NLP), Computer Vision (CV), and Speech Processing. It is built on FastAPI and adheres to the Enterprise AI Agent Interoperability Protocol (EAIP).

## Features

- **NLP**: Text generation and conditional responses using Google's FLAN-T5.
- **Computer Vision**: Real-time object detection using YOLOv8.
- **Speech-to-Text**: High-accuracy audio transcription using OpenAI's Whisper.
- **Text-to-Speech**: Offline speech synthesis using Pyttsx3.
- **Security**: Robust JWT-based authentication for all API endpoints.
- **EAIP Compliant**: Implements gRPC over HTTP/2 and SPIFFE/SPIRE for identity management.

## Installation

### Prerequisites

- Python 3.10+
- FFmpeg (for speech processing)
- espeak-ng (for text-to-speech on Linux)

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/OneFineStarstuff/AGI-Pipeline.git
   cd AGI-Pipeline
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Variables**:
   Set a secure secret key for JWT:
   ```bash
   export SECRET_KEY="your-very-secure-secret-key"
   ```

## Usage

1. **Run the FastAPI application**:
   ```bash
   uvicorn main:app --reload
   ```

2. **Access the API Documentation**:
   Navigate to `http://localhost:8000/docs` for the interactive Swagger UI.

### Using Docker

1. **Build the image**:
   ```bash
   docker build -t agi-pipeline .
   ```

2. **Run the container**:
   ```bash
   docker run -p 8000:8000 -e SECRET_KEY="your-secure-key" agi-pipeline
   ```

## API Endpoints

- `POST /process-nlp/`: Generate text responses.
- `POST /process-cv-detection/`: Detect objects in an uploaded image.
- `POST /speech-to-text/`: Transcribe uploaded audio files.
- `POST /text-to-speech/`: Synthesize text into speech.

*Note: All endpoints require a valid JWT bearer token.*

## Citation

If you use this software in your research, please cite it as follows:

```bibtex
@software{Tun_AGI-Pipeline_2024,
  author = {Tun, Kyaw T.},
  doi = {10.5281/zenodo.14504697},
  month = dec,
  title = {{AGI-Pipeline}},
  url = {https://github.com/OneFineStarstuff/AGI-Pipeline},
  version = {1.0.1},
  year = {2024}
}
```

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
