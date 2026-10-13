# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Validated browser control protocol."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class Command(BaseModel):
  model_config = ConfigDict(extra="forbid")
  type: str


class TransportCommand(Command):
  type: Literal["transport"]
  playing: bool


class ParamsCommand(Command):
  type: Literal["params"]
  temperature: float = Field(default=1.3, ge=0.0, le=3.0)
  top_k: int = Field(default=40, ge=1, le=1024)
  cfg_musiccoca: float = Field(default=3.0, ge=-1.0, le=7.0)
  cfg_notes: float = Field(default=1.0, ge=-1.0, le=7.0)
  cfg_drums: float = Field(default=1.0, ge=-1.0, le=7.0)
  drumless: bool = False
  onset_mode: bool = False
  unmask_width: int = Field(default=0, ge=0, le=127)


class Prompt(BaseModel):
  model_config = ConfigDict(extra="forbid")
  text: str = Field(max_length=500)
  weight: float = Field(default=1.0, ge=0.0, le=1.0)
  source: str | None = Field(default=None, max_length=64)


class PromptsCommand(Command):
  type: Literal["prompts"]
  prompts: list[Prompt] = Field(max_length=6)


class MidiCommand(Command):
  type: Literal["midi"]
  note: int = Field(ge=0, le=127)
  on: bool


class ResetCommand(Command):
  type: Literal["reset"]


class ModelCommand(Command):
  type: Literal["model"]
  model: Literal["mrt2_base", "mrt2_small"]


class RecordingCommand(Command):
  type: Literal["recording"]
  action: Literal["start", "stop"]


class BankCommand(Command):
  type: Literal["bank"]
  action: Literal["save", "load"]
  index: int = Field(ge=0, le=2)


ClientCommand = Annotated[
    TransportCommand
    | ParamsCommand
    | PromptsCommand
    | MidiCommand
    | ResetCommand
    | ModelCommand
    | RecordingCommand
    | BankCommand,
    Field(discriminator="type"),
]
COMMAND_ADAPTER = TypeAdapter(ClientCommand)
