from dataclasses import dataclass
from keras import Input, Model
from keras.layers import Dense, Dropout
from vit_keras import vit


@dataclass(frozen=True)
class ViTSkinLesionModel:
    """Factory for a ViT-B16 binary classifier (sigmoid head)."""
    img_size: int
    dropout: float

    def build(self) -> Model:
        backbone = vit.vit_b16(
            classes=2,
            include_top=False,
            pretrained_top=False,
        )
        backbone.trainable = False

        inputs = Input(shape=(self.img_size, self.img_size, 3))
        x = backbone(inputs, training=False)  # (batch, 768)
        x = Dropout(self.dropout)(x)
        outputs = Dense(1, activation="sigmoid")(x)
        model = Model(inputs, outputs, name="vit_b16_binary")
        return model