from pathlib import Path

import mlc_chat

from modules import shared

class MLCChatModel:
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
        generation_config = mlc_chat.GenerationConfig(
            temperature=state['temperature'],
            top_p=state['top_p'],
            repetition_penalty=state['repetition_penalty'],
            max_gen_len=state['max_new_tokens'],
            presence_penalty=state['presence_penalty'],
            frequency_penalty=state['frequency_penalty'],
            stop=state['custom_stopping_strings'],
            #TODO: n (num_return_sequence)    
        )
        output = self.model.generate(prompt=prompt, generation_config=generation_config)
        return output

    def generate_with_streaming(self, prompt, state, callback=None):
        prompt = prompt if type(prompt) is str else prompt.decode()
        self.model._prefill(input=prompt)
        generation_config = mlc_chat.GenerationConfig(
            temperature=state['temperature'],
            top_p=state['top_p'],
            repetition_penalty=state['repetition_penalty'],
            max_gen_len=state['max_new_tokens'],
            presence_penalty=state['presence_penalty'],
            frequency_penalty=state['frequency_penalty'],
            stop=state['custom_stopping_strings'],
            #TODO: n (num_return_sequence)    
        )
        while not self.model._stopped():
            self.model._decode(generation_config=generation_config)
            content = self.model._get_message()
            if content:
                # Remove the replacement character (U+FFFD) from the response
                # This is to handle emojis. An emoji might be made up of multiple tokens.
                # In the Rest streaming setting, if an emoji gets truncated in the middle of
                # its encoded byte sequence, a replacement character will appear.
                valid_content = content.replace("�", "")
                yield valid_content
        