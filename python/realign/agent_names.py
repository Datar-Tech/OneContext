"""Random astronomical context name generator.

Generates names like "Nova Prime", "Lunar Arc", etc.
"""

import random

PREFIXES = [
    "nova",
    "lunar",
    "solar",
    "stellar",
    "nebula",
    "astral",
    "cosmic",
    "aurora",
    "comet",
    "quasar",
    "pulsar",
    "zenith",
    "orbit",
    "photon",
    "prism",
    "radiant",
    "vortex",
    "flux",
    "plasma",
    "cipher",
]

SUFFIXES = [
    "prime",
    "arc",
    "drift",
    "core",
    "edge",
    "field",
    "wave",
    "pulse",
    "spark",
    "bloom",
    "gate",
    "link",
    "path",
    "peak",
    "span",
    "trace",
    "nexus",
    "axis",
    "relay",
    "vault",
]

_rng = random.SystemRandom()


def generate_agent_name() -> str:
    """Return a random name, e.g. 'Nova Prime'."""
    return f"{_rng.choice(PREFIXES).capitalize()} {_rng.choice(SUFFIXES).capitalize()}"
