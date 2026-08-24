from flask import Flask, render_template, request, jsonify
import os

import torch
import torch.nn as nn

from PIL import Image

from torchvision import transforms
from torchvision.models import mobilenet_v3_small


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device("cpu")

MODEL_PATH = "mobilenet_alzheimer.pth"


# Les classes sont celles détectées par ImageFolder
CLASS_NAMES = [
    "MildDemented",
    "ModerateDemented",
    "NonDemented",
    "VeryMildDemented"
]


# ============================================================
# TRANSFORMATION IMAGE
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# CHARGEMENT DU MODÈLE
# ============================================================

print("=" * 60)
print("CHARGEMENT DE MOBILENETV3-SMALL")
print("=" * 60)

print("Device :", DEVICE)
print("Model  :", MODEL_PATH)

model = mobilenet_v3_small(
    weights=None
)

number_features = model.classifier[-1].in_features

model.classifier[-1] = nn.Linear(
    number_features,
    4
)


# Charger les poids entraînés
checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

# Notre fichier contient model_state_dict
model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)

model.eval()

print("✓ Modèle chargé avec succès")
print("✓ 4 classes Alzheimer")
print()


# ============================================================
# FONCTION DE PRÉDICTION
# ============================================================

def predict_image(image_path):

    image = Image.open(image_path).convert("RGB")

    image_tensor = transform(image)

    # Ajouter la dimension batch
    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(DEVICE)

    # Pas de calcul de gradient
    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

    probabilities = probabilities[0]

    # Classe avec la probabilité maximale
    predicted_index = torch.argmax(
        probabilities
    ).item()

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    predicted_probability = (
        probabilities[predicted_index].item()
        * 100
    )

    # Toutes les probabilités
    results = {}

    for i, class_name in enumerate(CLASS_NAMES):

        results[class_name] = round(
            probabilities[i].item() * 100,
            2
        )

    return (
        predicted_class,
        round(predicted_probability, 2),
        results
    )


# ============================================================
# PAGE PRINCIPALE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "neurodetect.html"
    )


# ============================================================
# ANALYSE
# ============================================================

@app.route("/analyze", methods=["POST"])
def analyze():

    # --------------------------------------------------------
    # Vérifier l'image
    # --------------------------------------------------------

    if "image" not in request.files:

        return jsonify({
            "success": False,
            "error": "No image uploaded."
        })

    file = request.files["image"]

    if file.filename == "":

        return jsonify({
            "success": False,
            "error": "No image selected."
        })


    # --------------------------------------------------------
    # Sauvegarde
    # --------------------------------------------------------

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(filepath)


    # --------------------------------------------------------
    # VRAIE PRÉDICTION
    # --------------------------------------------------------

    try:

        (
            predicted_class,
            predicted_probability,
            probabilities
        ) = predict_image(filepath)

    except Exception as e:

        print("ERROR :", e)

        return jsonify({
            "success": False,
            "error": str(e)
        })


    # --------------------------------------------------------
    # Pour le moment :
    #
    # Seul MobileNet est réellement entraîné.
    #
    # LightAlzNet et Model 3 seront ajoutés plus tard.
    # --------------------------------------------------------

    mobilenet_score = predicted_probability


    # --------------------------------------------------------
    # Explication temporaire
    # --------------------------------------------------------

    explanation = (
        "The MobileNetV3-Small model analyzed the "
        "uploaded image and classified it as "
        f"{predicted_class} with an estimated "
        f"confidence of {predicted_probability:.2f}%. "
        "This result is experimental and is not "
        "intended for medical diagnosis."
    )


    # --------------------------------------------------------
    # Réponse JSON
    # --------------------------------------------------------

    return jsonify({

        "success": True,

        "prediction": {

            "class":
                predicted_class,

            "confidence":
                predicted_probability
        },

        "models": {

            "MobileNetV3-Small":
                round(mobilenet_score, 2),

            "LightAlzNet":
                None,

            "Model 3":
                None
        },

        "probabilities":
            probabilities,

        # Pour l'instant on ne calcule PAS
        # un vrai MMSE à partir de cette prédiction.
        "mmse":
            None,

        "mmse_percentage":
            None,

        "explanation":
            explanation
    })


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("NEURO AI ANALYSIS - PFE PROTOTYPE")
    print("=" * 60)

    print()
    print("Server starting...")
    print()
    print("Open your browser at:")
    print("http://127.0.0.1:5000")
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )