import os
import base64
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")


def get_model_from_gcp():
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        project=PROJECT_ID,
        location=LOCATION,
    )
    return model


def get_embeddings_from_gcp():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        project=PROJECT_ID,
        location=LOCATION,
    )
    return embeddings


def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")