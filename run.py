from vit_keras import vit

model = vit.vit_b16(
    classes=2,
    include_top=False,
    pretrained_top=False
)
print(model.summary())
