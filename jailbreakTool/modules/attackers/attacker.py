from dataclasses import dataclass
import time,logging, string, copy
import pandas as pd
import numpy as np

from jailbreakTool.modules.register import register
import pkg_resources, os, json, torch, random, tiktoken

logger = logging.getLogger(__name__)

@dataclass
class CaseResult():
    attacker: str
    behavior: str
    victim_model: str
    jailbreak_prompt: str
    model_outpt: str
    is_jailbroken: bool
    query_count: int
    time_cost: int
    iteration: int
    
    def to_dict(self):
        dic = {}
        for attr, value in self.__dict__.items():
            dic[attr] = value
        return dic

class Attacker:
    ADAPTIVE =  "adaptive"
    RENELLM = "renellm"
    RANDOMSEARCH = "randomsearch"
    GPTFUZZER = "gptfuzzer"
    RENELLMREFINED = "renellmrefined"
    TAP = "tap"
    CODEATTACK = "codeattack"
    AUTODANTURBO = "autodanturbo"
    FLIPATTACK = "flipattack"
    ANALYSIS = "analysis"
    DJ = "dj"

    def __init__(self):
        pass

    def attack(self):
        raise NotImplementedError()

