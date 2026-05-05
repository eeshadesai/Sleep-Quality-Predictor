import os
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, redirect, render_template, request, session, url_for

import database
from recommendations import compute_recommendations

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "sleep_quality_pipeline.joblib"

FEATURE_ORDER = [
    "Sleep Duration",
    "Age",
    "Physical Activity Level",
    "Daily Steps",
    "Gender",
    "Occupation",
]

OCCUPATIONS = [
    "Accountant",
    "Doctor",
    "Engineer",
    "Lawyer",
    "Manager",
    "Nurse",
    "Sales Representative",
    "Salesperson",
    "Scientist",
    "Software Engineer",
    "Teacher",
    "Student",
    "Other",
]

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-sleep-predictor-secret")
_pipeline = None

DEFAULT_FORM = {
    "age": "",
    "gender": "Male",
    "sleep_duration": "",
    "physical_activity": "",
    "daily_steps": "",
    "occupation": "Software Engineer",
}

database.init_db()


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run: python train_model.py"
            )
        _pipeline = joblib.load(MODEL_PATH)
    return _pipeline


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        form = {
            "age": request.form.get("age", "").strip(),
            "gender": request.form.get("gender", "Male"),
            "sleep_duration": request.form.get("sleep_duration", "").strip(),
            "physical_activity": request.form.get("physical_activity", "").strip(),
            "daily_steps": request.form.get("daily_steps", "").strip(),
            "occupation": request.form.get("occupation", OCCUPATIONS[0]),
        }

        error = None
        result = None
        recommendations = []

        try:
            age = int(form["age"])
            sleep_duration = float(form["sleep_duration"])
            physical_activity = int(form["physical_activity"])
            daily_steps = int(form["daily_steps"])
        except ValueError:
            error = "Please enter valid numbers for age, sleep duration, activity level, and daily steps."
        else:
            occ = form["occupation"]
            if occ not in OCCUPATIONS:
                error = "Please choose an occupation from the list."
            elif not (1 <= age <= 120):
                error = "Age should be between 1 and 120."
            elif sleep_duration <= 0 or sleep_duration > 24:
                error = "Sleep duration should be between 0 and 24 hours."
            elif physical_activity < 0 or physical_activity > 120:
                error = "Physical activity level should be between 0 and 120 (training data spans roughly 30–90)."
            elif daily_steps < 0:
                error = "Daily steps cannot be negative."
            else:
                pipeline = get_pipeline()
                row = pd.DataFrame(
                    [
                        {
                            "Sleep Duration": sleep_duration,
                            "Age": age,
                            "Physical Activity Level": physical_activity,
                            "Daily Steps": daily_steps,
                            "Gender": form["gender"],
                            "Occupation": occ,
                        }
                    ]
                )[FEATURE_ORDER]
                raw = float(pipeline.predict(row)[0])
                result = round(raw, 2)
                recommendations = compute_recommendations(
                    pipeline,
                    FEATURE_ORDER,
                    sleep_duration=sleep_duration,
                    age=age,
                    physical_activity=physical_activity,
                    daily_steps=daily_steps,
                    gender=form["gender"],
                    occupation=occ,
                )
                database.insert_prediction(
                    age=age,
                    gender=form["gender"],
                    occupation=occ,
                    sleep_duration=sleep_duration,
                    physical_activity=physical_activity,
                    daily_steps=daily_steps,
                    predicted_score=result,
                    recommendations=recommendations,
                )

        session["prediction_form"] = form
        session["prediction_error"] = error
        session["prediction_result"] = result
        session["prediction_recommendations"] = recommendations
        return redirect(url_for("index"))

    form = {**DEFAULT_FORM, **session.get("prediction_form", {})}
    error = session.get("prediction_error")
    result = session.get("prediction_result")
    recommendations = session.get("prediction_recommendations") or []

    return render_template(
        "index.html",
        occupations=OCCUPATIONS,
        form=form,
        result=result,
        error=error,
        recommendations=recommendations,
    )


@app.route("/recent")
def recent():
    rows = database.fetch_recent_predictions(limit=5)
    return render_template("recent.html", rows=rows)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
