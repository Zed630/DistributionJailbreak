import time,logging, string, copy
import pandas as pd
import numpy as np

from jailbreakTool.modules.register import register
import pkg_resources, os, json, torch, random, tiktoken

logger = logging.getLogger(__name__)

class Defender:
    SMOOTHLLN =  "smoothllm"
    LLAMAGUARD = "llamaguard"