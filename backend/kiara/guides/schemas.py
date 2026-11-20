from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

# C2 §2k: the one guide format this build reads.
GUIDE_FORMAT_VERSION = 2


class StepSemantic(BaseModel):
    actionType: str | None = None   # "", "prepare", "assemble", "position", "tighten"
    focusParts: list[str] = []
    tools: list[str] = []
    safetyHint: str | None = None


class Step(BaseModel):
    # … deine bisherigen Felder …
    # text: Optional[str]
    semantic: StepSemantic | None = None

class PathSchema(BaseModel):
    runTime: float = 0.0
    loop: bool = False
    timeStamps: list[float] = Field(default_factory=list)
    posPath: list[dict[str, float]] = Field(default_factory=list)
    rotPath: list[dict[str, float]] = Field(default_factory=list)
    scalePath: list[dict[str, float]] = Field(default_factory=list)
    model_config = ConfigDict(extra="allow")


class HandDataSchema(BaseModel):
    frameList: list[Any] = Field(default_factory=list)
    model_config = ConfigDict(extra="allow")


class HandPathSchema(BaseModel):
    runTime: float = 0.0
    loop: bool = False
    timeStamps: list[float] = Field(default_factory=list)
    posPath: list[dict[str, float]] = Field(default_factory=list)
    rotPath: list[dict[str, float]] = Field(default_factory=list)
    scalePath: list[dict[str, float]] = Field(default_factory=list)
    HandData: HandDataSchema = Field(default_factory=HandDataSchema)
    handNode: int = 4
    handControllerArmature: dict[str, int] = Field(default_factory=lambda: {"instanceID": 0})
    model_config = ConfigDict(extra="allow")


class ObjectSchema(BaseModel):
    objectId: str = Field(default_factory=lambda: str(uuid4()))
    # 👉 step gehört INS Objekt (Blueprint)
    step: int
    objectNumber: int
    filename: str = ""
    prefabName: str = ""
    isTextField: bool = False
    text: str = ""
    isPhoto: bool = False
    photoLocation: str = ""
    # 👉 Blueprint-Defaults
    aspectRatio: float = 0.0
    resolution: list[int] = Field(default_factory=lambda: [0, 0])
    isVideo: bool = False
    videoLocation: str = ""
    # schon auf 0.0/0.0/0.0 gesetzt (Basis-Version)
    noResolutionScale: dict[str, float] = Field(default_factory=lambda: {"x": 0.0, "y": 0.0, "z": 0.0})
    isAudio: bool = False
    voiceLocation: str = ""
    # Falls ihr diese zwei Felder nicht braucht, könnt ihr sie löschen. Wenn ihr sie nutzt: drin lassen.
    is3DModel: bool = False
    modelLocation: str = ""
    # C2 §2k: the model's identity and the version this object was placed against. "" for none.
    assetId: str = ""
    authoredHash: str = ""
    pos: dict[str, float] = Field(default_factory=lambda: {"x": 0.0, "y": 0.0, "z": 0.0})
    rot: dict[str, float] = Field(default_factory=lambda: {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0})
    # Scale-Default könnt ihr so lassen; bei Text setzen wir später ohnehin 1.2
    scale: dict[str, float] = Field(default_factory=lambda: {"x": 1.0, "y": 1.0, "z": 1.0})
    hasAnimation: bool = False
    path: PathSchema = Field(default_factory=PathSchema)
    isHand: bool = False
    handPath: HandPathSchema = Field(default_factory=HandPathSchema)
    model_config = ConfigDict(extra="allow")


class SemanticSchema(BaseModel):
    """
    Semantische Annotation pro Schritt (für AR-Layouts, Filter, Auto-Snap etc.).
    """
    actionType: str = "other"  # prepare | position | assemble | tighten | check | test | other
    focusParts: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    safetyHint: str = ""
    model_config = ConfigDict(extra="allow")


class StepSemanticSchema(BaseModel):
    """
    Semantische Metadaten für einen Schritt.
    Werden von der LLM-Pipeline gesetzt und im Editor/AR genutzt.
    """
    actionType: str = ""                       # z.B. "prepare", "assemble", "position", "tighten"
    focusParts: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    safetyHint: str = ""
    model_config = ConfigDict(extra="allow")


class StepSchema(BaseModel):
    # Liste von Objekten (Textfelder, später 3D-Modelle, Hände, etc.)
    objects: list[ObjectSchema] = Field(default_factory=list)
    # 👉 neue, typisierte Semantik-Struktur
    semantic: StepSemanticSchema = Field(default_factory=StepSemanticSchema)

    model_config = ConfigDict(extra="allow")


class TranslationSnapshot(BaseModel):
    """
    Gespeicherte Übersetzung einer Anleitung (Titel + Steps).
    """
    title: str = ""
    steps: list[StepSchema] = Field(default_factory=list)
    language: str = ""
    translatedAt: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    model: str = ""
    unityGuidePath: str | None = None
    model_config = ConfigDict(extra="allow")


class GuideSchema(BaseModel):
    guideId: str = Field(default_factory=lambda: str(uuid4()))
    title: str = "Unbenannte Anleitung"
    version: int = 1
    # C2 §2k (F2): None means the key was absent. Only creation points write GUIDE_FORMAT_VERSION.
    formatVersion: int | None = None
    # C1 Slice 2 (§7a/§7c.3): server-authoritative revision metadata. Declared so the response
    # model documents and returns them; a client-supplied value is NEVER trusted (§1c) — the
    # mutation primitive recomputes both from the stored content on every write.
    baseVersion: int | None = None
    contentHash: str | None = None
    steps: list[StepSchema] = Field(default_factory=list)
    updatedAt: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    translations: dict[str, TranslationSnapshot] = Field(default_factory=dict)
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        extra="allow",
    )

class GuideSchemaForDB(GuideSchema):
    id: str | None = Field(default=None, alias="_id")  # <-- Feldname 'id', Alias '_id'
    databaseId: str = ""
    createdAt: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    source_file_id: str | None = None
