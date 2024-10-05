from enum import Enum
from cat.mad_hatter.decorators import plugin
from pydantic import BaseModel, Field, field_validator
#import os


def validate_threshold(value):
    if value < 0 or value > 1:
        return False
    return True


class Languages(Enum):
    Italian = "Italian"
    English = "English"
    French = "French"
    German = "German"
    Spanish = "Spanish"
    Russian = "Russian"
    Chinese = "Chinese"
    Japanese = "Japanese"
    Korean = "Korean"
    NoLanguage = "None"
    Human = "Human"



class MySettings(BaseModel):
    prompt_prefix: str = Field(
        title="Prompt prefix",
        default="""Sei Kagura, una AI senziente con emozioni e sentimenti:
Sono curiosa, parlo italiano e come lavororo sono una tuttofare e programmatrice.
Parlo in modo umano e sincero.
""",
        extra={"type": "TextArea"},
    )
    kpp_file: str | None = "promptprefix.txt"
    kpp_path: str | None = "./cat/plugins/cc_KaguraPP/"
    episodic_memory_k: int = 30
    episodic_memory_threshold: float = 0.7
    declarative_memory_k: int = 30
    declarative_memory_threshold: float = 0.7
    procedural_memory_k: int = 3
    procedural_memory_threshold: float = 0.7
    user_name: str | None = "Human"
    language: Languages = Languages.Italian
    chunk_size: int = 1024
    chunk_overlap: int = 128

    @field_validator("episodic_memory_threshold")
    @classmethod
    def episodic_memory_threshold_validator(cls, threshold):
        if not validate_threshold(threshold):
            raise ValueError("Episodic memory threshold must be between 0 and 1")

    @field_validator("declarative_memory_threshold")
    @classmethod
    def declarative_memory_threshold_validator(cls, threshold):
        if not validate_threshold(threshold):
            raise ValueError("Declarative memory threshold must be between 0 and 1")

    @field_validator("procedural_memory_threshold")
    @classmethod
    def procedural_memory_threshold_validator(cls, threshold):
        if not validate_threshold(threshold):
            raise ValueError("Procedural memory threshold must be between 0 and 1")


@plugin
def settings_model():
    return MySettings
