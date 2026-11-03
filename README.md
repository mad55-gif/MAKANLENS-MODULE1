# MakanLens-MH -- Malaysian Food Recognition Module

## 1. Overview

This is **Student 1's food image recognition component** for
MakanLens-MH. It uses a trained **EfficientNetB0** model to predict
which supported Malaysian food class appears in an image.

The predicted food name and confidence score can be passed to the other
modules. This module recognises food; it does **not** calculate
calories, estimate portion size, or provide health recommendations.

## 2. What other students can use this module for

### Student 2 -- Portion and Nutrition Estimation

Use the predicted food class as the input to the nutrition module.

Workflow: 1. A user uploads or captures a food image. 2. EfficientNetB0
predicts the food class and returns a confidence score. 3. Student 2
maps the predicted class to the Malaysian food nutrition database. 4.
The user selects a portion category such as Small, Medium, or Large. 5.
Student 2's module estimates calories, protein, carbohydrates, and fat.

**Important:** The recognition model does not estimate portion size or
nutritional values. Those calculations belong to Student 2's module.

### Student 3 -- Personal Health and Wellness Monitoring

Use the nutrition results produced by Student 2 to record a user's meal
and update daily nutrition intake. The recognised food name can be used
as the meal label, together with the calories and macronutrients
returned by Student 2.

### Student 4 -- Dashboard and User Evaluation

Display the predicted food name and confidence score on the
food-recognition screen. The dashboard can then display nutrition and
wellness information returned by the other modules.

## 3. Files provided

  File/folder                    Purpose
  ------------------------------ --------------------------------------------------
  `efficientnetb0_model.keras`   Trained EfficientNetB0 model
  `class_names.txt`              Food labels in the exact order used by the model
  `requirements.txt`             Python package dependencies
  `test_images/`                 Sample images for testing

Keep `class_names.txt` together with the model. Do not manually sort or
rearrange its labels: the order must match the model's output indexes.

## 4. Model information

  Item                  Information
  --------------------- ------------------
  Model                 EfficientNetB0
  Input image size      224 × 224 pixels
  Input colour format   RGB
  Test accuracy         83.85%
  Test loss             0.5312

These results are from the held-out test dataset. Performance on new
real-world images may differ.

## 5. Setup

Install the required packages:

``` bash
pip install -r requirements.txt
```

Example `requirements.txt`:

``` text
tensorflow
numpy
pillow
```

Use compatible package versions if the project environment requires
pinned versions.

## 6. Load the model and class labels

Place the model and `class_names.txt` in the same directory:

``` python
import tensorflow as tf

model = tf.keras.models.load_model("efficientnetb0_model.keras")

with open("class_names.txt", "r", encoding="utf-8") as file:
    class_names = [line.strip() for line in file if line.strip()]

print("Model loaded.")
print("Number of classes:", len(class_names))
```

## 7. Predict a food image

``` python
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array

def predict_food(image_path):
    image = load_img(
        image_path,
        target_size=(224, 224),
        color_mode="rgb"
    )

    image_array = img_to_array(image)
    input_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(input_array, verbose=0)[0]

    predicted_index = int(np.argmax(predictions))
    predicted_food = class_names[predicted_index]
    confidence = float(predictions[predicted_index])

    return {
        "food_name": predicted_food,
        "confidence": confidence
    }

result = predict_food("test_images/sample.jpg")

print("Food:", result["food_name"])
print(f"Confidence: {result['confidence'] * 100:.2f}%")
```

**Preprocessing note:** The preprocessing during integration must match
the preprocessing used to train this model. The example resizes the
image and converts it to RGB but does not divide pixel values by 255.
Check the training notebook and model preprocessing layers before adding
normalization. Do not add extra normalization unless it matches the
training pipeline.

## 8. Suggested integration output

The recognition module can return a simple result:

``` python
{
    "food_name": "nasi_lemak",
    "confidence": 0.92
}
```

These values are illustrative only. Actual predictions depend on the
input image.

Student 2 can use `food_name` to look up nutrition information. The
confidence score can be displayed or used to ask the user to confirm an
uncertain prediction.

## 9. Important integration notes

-   Use the exact labels in `class_names.txt` as the nutrition database
    keys.
-   Confirm that the number of labels matches the model's output count.
-   Keep the output-index-to-class-name mapping unchanged.
-   If confidence is low, consider asking the user to retake the photo
    or select the food manually.
-   The model only recognises classes it was trained on. It may
    incorrectly classify food outside its supported classes.
-   A confidence score is not a guarantee that a prediction is correct.
-   Nutrition values should be presented as estimates, not exact
    measurements.

## 10. Model validation

Final test-dataset evaluation:

-   **Test accuracy:** 83.85%
-   **Test loss:** 0.5312

Add precision, recall, F1-score, Top-1/Top-3/Top-5 accuracy, and the
final confusion matrix after their validation results have been
confirmed.

------------------------------------------------------------------------

**Module owner:** Student 1 -- AI Malaysian Food Recognition\
**Project:** MakanLens-MH
