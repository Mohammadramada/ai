from cog import BasePredictor, Input
import torch
from transformers import pipeline

class Predictor(BasePredictor):
    def setup(self):
        model_id = "Mo0189/iCreative.ai"

        self.pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

    def predict(self, prompt: str = Input(description="أدخل طلبك هنا")) -> str:
        outputs = self.pipe(prompt, max_new_tokens=512)
        return outputs[0]["generated_text"]
