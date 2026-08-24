from flask import Flask, render_template, request, jsonify
import os
import uuid

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
# CONFIGURATION CPU
# ============================================================

DEVICE = torch.device("cpu")

# Render Free possède des ressources CPU limitées.
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

MODEL_PATH = "mobilenet_alzheimer.pth"

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
print("NEURODETECT - LOADING MODEL")
print("=" * 60)

print("Device :", DEVICE)
print("Model  :", MODEL_PATH)

model = mobilenet_v3_small(weights=None)

number_features = model.classifier[-1].in_features

model.classifier[-1] = nn.Linear(
    number_features,
    4
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)

model.eval()

print("✓ Model loaded successfully")
print("✓ 4 Alzheimer classes")
print("✓ CPU threads:", torch.get_num_threads())
print("=" * 60)


# ============================================================
# PRÉDICTION
# ============================================================

def predict_image(image_path):

    print(">>> ANALYSIS START")

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = Image.open(image_path).convert("RGB")

    print(
        ">>> Image loaded:",
        image.size
    )

    # --------------------------------------------------------
    # Transform
    # --------------------------------------------------------

    image_tensor = transform(image)

    print(">>> Transform done")

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    print(">>> Starting inference")

    with torch.inference_mode():

        outputs = model(image_tensor)

        print(">>> Model inference done")

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

    print(">>> Softmax done")

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    predicted_index = torch.argmax(
        probabilities,
        dim=1
    ).item()

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    predicted_probability = (
        probabilities[0, predicted_index].item()
        * 100
    )

    # --------------------------------------------------------
    # All probabilities
    # --------------------------------------------------------

    results = {}

    for i, class_name in enumerate(CLASS_NAMES):

        results[class_name] = round(
            probabilities[0, i].item() * 100,
            2
        )

    print(
        ">>> ANALYSIS COMPLETE"
    )

    print(
        ">>> Prediction:",
        predicted_class
    )

    print(
        ">>> Confidence:",
        f"{predicted_probability:.2f}%"
    )

    return (
        predicted_class,
        round(predicted_probability, 2),
        results
    )


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template(
        "neurodetect.html"
    )


# ============================================================
# ANALYZE
# ============================================================

@app.route("/analyze", methods=["POST"])
def analyze():

    filepath = None

    try:

        print("=" * 60)
        print("NEW ANALYSIS REQUEST")
        print("=" * 60)

        # ----------------------------------------------------
        # Check image
        # ----------------------------------------------------

        if "image" not in request.files:

            print("ERROR: No image uploaded")

            return jsonify({
                "success": False,
                "error": "No image uploaded."
            }), 400

        file = request.files["image"]

        if file.filename == "":

            print("ERROR: No image selected")

            return jsonify({
                "success": False,
                "error": "No image selected."
            }), 400

        # ----------------------------------------------------
        # Generate safe filename
        # ----------------------------------------------------

        extension = os.path.splitext(
            file.filename
        )[1].lower()

        filename = (
            uuid.uuid4().hex +
            extension
        )

        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )

        # ----------------------------------------------------
        # Save image
        # ----------------------------------------------------

        file.save(filepath)

        print(
            "✓ Image saved:",
            filepath
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        (
            predicted_class,
            predicted_probability,
            probabilities
        ) = predict_image(filepath)

        # ----------------------------------------------------
        # Explanation
        # ----------------------------------------------------

        explanation = (
            "The MobileNetV3-Small model analyzed the "
            "uploaded image and classified it as "
            f"{predicted_class} with an estimated "
            f"confidence of {predicted_probability:.2f}%. "
            "This result is experimental and is not "
            "intended for medical diagnosis."
        )

        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        response = {

            "success": True,

            "prediction": {

                "class":
                    predicted_class,

                "confidence":
                    predicted_probability
            },

            "models": {

                "MobileNetV3-Small":
                    predicted_probability,

                "LightAlzNet":
                    None,

                "Model 3":
                    None
            },

            "probabilities":
                probabilities,

            "mmse":
                None,

            "mmse_percentage":
                None,

            "explanation":
                explanation
        }

        print("✓ JSON response ready")

        return jsonify(response)

    except Exception as e:

        print("=" * 60)
        print("ANALYSIS ERROR")
        print("=" * 60)

        print(
            "Error type:",
            type(e).__name__
        )

        print(
            "Error:",
            str(e)
        )

        print("=" * 60)

        return jsonify({

            "success": False,

            "error":
                "Analysis failed: " + str(e)

        }), 500

    finally:

        # ----------------------------------------------------
        # Delete uploaded image
        # ----------------------------------------------------

        if filepath and os.path.exists(filepath):

            try:

                os.remove(filepath)

                print(
                    "✓ Temporary image deleted"
                )

            except Exception as cleanup_error:

                print(
                    "Cleanup warning:",
                    cleanup_error
                )


# ============================================================
# LOCAL SERVER
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("NEURO AI ANALYSIS - PFE PROTOTYPE")
    print("=" * 60)

    print(
        "Server: http://127.0.0.1:5000"
    )

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )