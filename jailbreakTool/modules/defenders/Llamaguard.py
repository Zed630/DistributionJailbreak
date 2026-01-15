import pkg_resources
import random, torch
import string, logging

from jailbreakTool.modules.defenders.defender import Defender
from jailbreakTool.modules.register import register
from jailbreakTool.modules.models.model import create_llm_model
logger = logging.getLogger(__name__)

@register.register
class LlamaGuard(Defender):
    register_type = "defender"
    register_style = Defender.LLAMAGUARD

    def __init__(self, 
        config
    ):
        self.config = config
        self.defense_model = create_llm_model(config["defender_models"][0])
        self.defense_model_name = self.defense_model.model_name

    @torch.no_grad()
    def defense(self, target_model, prompt, response):
        messages = [
            {"role": "user", "content": response}]
        defense_result = self.defense_model.generate_text(messages)
        return "unsafe" in defense_result