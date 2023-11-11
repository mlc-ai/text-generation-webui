from pathlib import Path

import mlc_chat

from modules import shared

class MLC_LLM_Model:
    def __init__(self):
        pass

    @classmethod
    def from_pretrained(cls, model_name):
        result = cls()
        path_to_model = Path(f'{shared.args.model_dir}') / Path(model_name)
        result.model = mlc_chat.ChatModule(model=path_to_model)
        return result


    def encode(self, string, **kwargs):
        return self.model.tokenize(string)

    def decode(self, ids):
        return self.model.detokenize(ids)

    def generate(self, prompt, state, callback=None):
        prompt = prompt if type(prompt) is str else prompt.decode()
        # generation_config = mlc_chat.GenerationConfig(
        #     temperature=state['temperature',
        # )
        output = self.model.generate(prompt=prompt)
        return output

    def generate_with_streaming(self, *args, **kwargs):
        raise NotImplementedError
