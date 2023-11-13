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

    def generate_with_streaming(self, prompt, state, callback=None):
        prompt = prompt if type(prompt) is str else prompt.decode()
        self.model._prefill(input=prompt)

        while not self.model._stopped():
            self.model._decode()
            content = self.model._get_message()
            if content:
                # Remove the replacement character (U+FFFD) from the response
                # This is to handle emojis. An emoji might be made up of multiple tokens.
                # In the Rest streaming setting, if an emoji gets truncated in the middle of
                # its encoded byte sequence, a replacement character will appear.
                valid_content = content.replace("�", "")
                yield valid_content
        