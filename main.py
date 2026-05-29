"""
Main application module for the AGI Pipeline.
Provides NLP, Computer Vision, and Speech processing capabilities
via a FastAPI.
"""

import os
import io
import signal
import sys
import asyncio
from typing import List

import torch
import jwt
import pyttsx3
import uvicorn
import whisper
from PIL import Image
from fastapi import FastAPI, UploadFile, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from loguru import logger
from transformers import T5Tokenizer, T5ForConditionalGeneration
from ultralytics import YOLO


# === Configuration and Logging Setup ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.add(
    "pipeline_{time}.log",
    rotation="1 MB",
    level="DEBUG",
    enqueue=True,
    backtrace=True,
    diagnose=True
)
logger.info("Application startup")

# === Security Setup ===
SECRET_KEY = os.getenv(
    "SECRET_KEY",
    "YvZz9Hni0hWJPh_UWW4dQYf9rhIe9nNYcC5ZQTTZz0Q"
)
ALGORITHM = "HS256"
OAUTH2_SCHEME = OAuth2PasswordBearer(tokenUrl="token")


def create_access_token(data: dict):
    """
    Creates a JWT access token for a given data payload.
    """
    to_encode = data.copy()
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def authenticate_user(token: str = Depends(OAUTH2_SCHEME)):
    """
    Authenticates a user based on the provided JWT token.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.PyJWTError:
        logger.warning("Authentication failed.")
        raise HTTPException(status_code=401, detail="Invalid token") from None
    return payload


# === Pydantic Models ===
class TextRequest(BaseModel):
    """Request model for text-based endpoints."""
    text: str


class TextResponse(BaseModel):
    """Response model for text-based endpoints."""
    response: str


# === NLP Module (T5 Transformer) ===
class NLPModule:
    """
    Module for Natural Language Processing using T5 models.
    """
    # pylint: disable=too-few-public-methods

    def __init__(self):
        """Initializes the NLP module with a pre-trained T5 model."""
        model_name = "google/flan-t5-small"
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        logger.info("NLP model loaded successfully.")

    def generate_text(self, prompt: str) -> str:
        """Generates text based on the provided prompt."""
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")
        logger.debug(f"Generating text for prompt: {prompt}")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = self.model.generate(inputs["input_ids"], max_length=100)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Generated response: {response}")
        return response


# === CV Module (YOLOv8 for Object Detection) ===
class CVModule:
    """
    Module for Computer Vision using YOLOv8 for object detection.
    """
    # pylint: disable=too-few-public-methods

    def __init__(self):
        """Initializes the CV module with the YOLOv8 model."""
        self.model = YOLO('yolov8n.pt').to(DEVICE)
        logger.info("CV model loaded successfully.")

    def detect_objects(self, image: Image.Image) -> str:
        """
        Detects objects within the provided image and returns results as JSON.
        """
        if image is None:
            raise ValueError("Image cannot be None")
        logger.debug("Detecting objects in the image.")
        results = self.model(image)
        return results[0].tojson()


# === Speech Processor (Whisper for Speech-to-Text) ===
class SpeechProcessor:
    """
    Module for speech processing, including transcription and synthesis.
    """

    def __init__(self):
        """Initializes the speech processor with Whisper and pyttsx3."""
        self.whisper_model = whisper.load_model("base")
        self.tts = pyttsx3.init()
        logger.info("Speech processor initialized successfully.")

    def speech_to_text(self, audio_file: UploadFile) -> str:
        """Transcribes audio from an uploaded file."""
        with audio_file.file as audio_data:
            result = self.whisper_model.transcribe(audio_data)
            return result['text']

    def text_to_speech(self, text: str) -> None:
        """Synthesizes speech from the provided text."""
        if not text.strip():
            raise ValueError("Text cannot be empty.")
        self.tts.say(text)
        self.tts.runAndWait()

    def __del__(self):
        """Cleans up the TTS engine."""
        if hasattr(self, 'tts'):
            self.tts.stop()


# === Enhanced AGI Pipeline ===
class EnhancedAGIPipeline:
    """
    Integrated pipeline coordinating NLP, CV, and Speech modules.
    """

    def __init__(self):
        """Initializes the integrated pipeline."""
        self.nlp = NLPModule()
        self.cv = CVModule()
        self.speech_processor = SpeechProcessor()

    async def process_nlp(self, text: str) -> str:
        """Asynchronously processes text using the NLP module."""
        return await asyncio.to_thread(self.nlp.generate_text, text)

    async def process_cv(self, image: Image.Image) -> str:
        """Asynchronously processes an image using the CV module."""
        return await asyncio.to_thread(self.cv.detect_objects, image)

    async def process_speech_to_text(self, audio_file: UploadFile) -> str:
        """Asynchronously transcribes speech from a file."""
        return await asyncio.to_thread(
            self.speech_processor.speech_to_text,
            audio_file
        )

    async def process_text_to_speech(self, text: str) -> None:
        """Asynchronously synthesizes speech from text."""
        await asyncio.to_thread(self.speech_processor.text_to_speech, text)


# === FastAPI Application ===
app = FastAPI()

pipeline = EnhancedAGIPipeline()


# === Graceful Shutdown ===
def shutdown_signal_handler(sig, _frame):
    """Handles termination signals for a graceful shutdown."""
    logger.info(f"Received signal {sig}. Shutting down gracefully...")
    sys.exit(0)


signal.signal(signal.SIGINT, shutdown_signal_handler)
signal.signal(signal.SIGTERM, shutdown_signal_handler)


# === Endpoints ===
@app.post(
    "/process-nlp/",
    response_model=TextResponse,
    dependencies=[Depends(authenticate_user)]
)
async def process_nlp_endpoint(request: TextRequest):
    """Endpoint for NLP text generation."""
    response = await pipeline.process_nlp(request.text)
    return {"response": response}


@app.post("/process-cv-detection/", dependencies=[Depends(authenticate_user)])
async def process_cv_detection(file: UploadFile):
    """Endpoint for object detection in a single image."""
    image = Image.open(io.BytesIO(await file.read()))
    response = await pipeline.process_cv(image)
    return {"detections": response}


@app.post("/batch-cv-detection/", dependencies=[Depends(authenticate_user)])
async def batch_cv_detection(files: List[UploadFile]):
    """Endpoint for batch object detection in multiple images."""
    tasks = [
        pipeline.process_cv(Image.open(io.BytesIO(await file.read())))
        for file in files
    ]
    responses = await asyncio.gather(*tasks)
    return {"batch_detections": responses}


@app.post(
    "/speech-to-text/",
    response_model=TextResponse,
    dependencies=[Depends(authenticate_user)]
)
async def speech_to_text_endpoint(file: UploadFile):
    """Endpoint for speech-to-text transcription."""
    response = await pipeline.process_speech_to_text(file)
    return {"response": response}


@app.post("/text-to-speech/", dependencies=[Depends(authenticate_user)])
async def text_to_speech_endpoint(request: TextRequest):
    """Endpoint for text-to-speech synthesis."""
    await pipeline.process_text_to_speech(request.text)
    return {"response": "Speech synthesis complete."}


# === Run the Application ===
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
