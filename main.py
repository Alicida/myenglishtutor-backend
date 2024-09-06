from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import texttospeech
from google.api_core.client_options import ClientOptions
import os

app = FastAPI()

# ----> CONFIGURACIÓN DE CORS <----
origins = [
    "https://opulent-cod-pjrj47rx47pc7w79-3000.app.github.dev",
    "https://bookish-barnacle-v6v69pvx7ww2p6r6-8000.app.github.dev"  # Agrega este origen
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ----> FIN DE LA CONFIGURACIÓN DE CORS <---

# ----> CONFIGURACIÓN DE PALM 2 <----
endpoint = "us-central1-aiplatform.googleapis.com"
location = "us-central1"
api_endpoint = f"{location}-aiplatform.googleapis.com"
client_options = ClientOptions(api_endpoint=api_endpoint)

# Reemplaza 'tu-proyecto' con el ID de tu proyecto de Google Cloud
parent = f"projects/hotline-434020/locations/{location}"

# ----> FIN DE LA CONFIGURACIÓN DE PALM 2 <----

@app.post("/transcribe/")
async def transcribe_audio(audio_file: UploadFile = File(...)):
    # Convierte el audio a texto usando Google Cloud Speech-to-Text
    client = speech.SpeechClient()
    content = await audio_file.read()
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=44100,
        language_code="en-US",
    )

    response = client.recognize(config=config, audio=audio)
    transcript = ""
    for result in response.results:
        transcript += result.alternatives[0].transcript

    # ---->  AQUÍ VA LA LÓGICA PARA GENERAR LA RESPUESTA CON PALM 2 <----
    from google.cloud import aiplatform
    aiplatform.init(client_options=client_options)

    # Reemplaza 'text-bison@001' con el nombre del modelo de PaLM 2 que deseas usar
    response = aiplatform.gapic.PredictionServiceClient(
        client_options=client_options
    ).predict(
        endpoint=f"{parent}/endpoints/text-bison@001",
        instances=[{"content": transcript}],
        parameters={},
    )
    response_text = response.predictions[0]['text']
    # ----> FIN DE LA LÓGICA DE PALM 2 <----

    # Convierte la respuesta de texto a audio usando Google Cloud Text-to-Speech
    synthesis_input = texttospeech.SynthesisInput(text=response_text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16
    )

    client = texttospeech.TextToSpeechClient()
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    # Guarda el audio generado en un archivo temporal
    with open("response.wav", "wb") as out:
        out.write(response.audio_content)

    return {"audio_path": "response.wav"}