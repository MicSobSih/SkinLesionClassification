import pandas as pd

meta = pd.read_csv("dataverse_files/HAM10000_metadata")

malignant_dx = {"mel", "bcc", "akiec"}
meta["benign_malignant"] = meta["dx"].apply(lambda d: "malignant" if d in malignant_dx else "benign")

# Example lookup for one image file
image_file = "ISIC_0027419.jpg"
image_id = image_file.removesuffix(".jpg")
print(meta.loc[meta["image_id"] == image_id, ["image_id", "dx", "benign_malignant"]])