import logging
import os

VERBOSE = os.getenv("VERBOSE", "0") == "1"

logging.basicConfig(
    level=logging.DEBUG if VERBOSE else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
_logger = logging.getLogger(__name__)