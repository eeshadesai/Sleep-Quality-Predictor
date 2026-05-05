
from __future__ import annotations

import pandas as pd

# Near observed ranges in Sleep_Data_Sampled.csv (slightly expanded for search)
SLEEP_MIN, SLEEP_MAX = 5.5, 9.0
PAL_MIN, PAL_MAX = 25, 120
STEPS_MIN, STEPS_MAX = 2000, 12000

MIN_GAIN = 0.04


def _row_df(
    feature_order,
    *,
    sleep_duration,
    age,
    physical_activity,
    daily_steps,
    gender,
    occupation,
):
    return pd.DataFrame(
        [
            {
                "Sleep Duration": sleep_duration,
                "Age": age,
                "Physical Activity Level": physical_activity,
                "Daily Steps": daily_steps,
                "Gender": gender,
                "Occupation": occupation,
            }
        ]
    )[feature_order]


def _predict(
    pipeline,
    feature_order,
    *,
    sleep_duration,
    age,
    physical_activity,
    daily_steps,
    gender,
    occupation,
) -> float:
    df = _row_df(
        feature_order,
        sleep_duration=sleep_duration,
        age=age,
        physical_activity=physical_activity,
        daily_steps=daily_steps,
        gender=gender,
        occupation=occupation,
    )
    return float(pipeline.predict(df)[0])


def compute_recommendations(
    pipeline,
    feature_order,
    *,
    sleep_duration: float,
    age: int,
    physical_activity: int,
    daily_steps: int,
    gender: str,
    occupation: str,
) -> list[str]:
    baseline = _predict(
        pipeline,
        feature_order,
        sleep_duration=sleep_duration,
        age=age,
        physical_activity=physical_activity,
        daily_steps=daily_steps,
        gender=gender,
        occupation=occupation,
    )

    candidates: list[tuple[float, str]] = []

    # Sleep duration — try hourly-ish steps from current
    for delta in (-1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5):
        new_sleep = round(sleep_duration + delta, 2)
        if new_sleep < SLEEP_MIN or new_sleep > SLEEP_MAX:
            continue
        if abs(new_sleep - sleep_duration) < 1e-6:
            continue
        s = _predict(
            pipeline,
            feature_order,
            sleep_duration=new_sleep,
            age=age,
            physical_activity=physical_activity,
            daily_steps=daily_steps,
            gender=gender,
            occupation=occupation,
        )
        gain = s - baseline
        if gain >= MIN_GAIN:
            candidates.append(
                (
                    gain,
                    f"Try aiming for about {new_sleep:g} hours of sleep each night "
                    f"instead of around {sleep_duration:g}.",
                )
            )

    # Physical activity level
    for delta in (-20, -15, -10, -5, 5, 10, 15, 20):
        new_pal = physical_activity + delta
        if new_pal < PAL_MIN or new_pal > PAL_MAX:
            continue
        if new_pal == physical_activity:
            continue
        s = _predict(
            pipeline,
            feature_order,
            sleep_duration=sleep_duration,
            age=age,
            physical_activity=new_pal,
            daily_steps=daily_steps,
            gender=gender,
            occupation=occupation,
        )
        gain = s - baseline
        if gain >= MIN_GAIN:
            candidates.append(
                (
                    gain,
                    f"Consider moving your physical activity level toward {new_pal} "
                    f"(you entered {physical_activity}).",
                )
            )

    # Daily steps
    for delta in (-2500, -2000, -1500, -1000, -500, 500, 1000, 1500, 2000, 2500):
        new_steps = daily_steps + delta
        if new_steps < STEPS_MIN or new_steps > STEPS_MAX:
            continue
        if new_steps == daily_steps:
            continue
        s = _predict(
            pipeline,
            feature_order,
            sleep_duration=sleep_duration,
            age=age,
            physical_activity=physical_activity,
            daily_steps=new_steps,
            gender=gender,
            occupation=occupation,
        )
        gain = s - baseline
        if gain >= MIN_GAIN:
            candidates.append(
                (
                    gain,
                    f"Try working toward about {new_steps:,} daily steps instead of {daily_steps:,}.",
                )
            )

    candidates.sort(key=lambda x: x[0], reverse=True)

    # Keep diverse tips: best per category keyword (sleep / activity / steps)
    picked: list[str] = []
    seen_kind = set()
    for gain, text in candidates:
        if "hours of sleep" in text:
            kind = "sleep"
        elif "physical activity level" in text:
            kind = "pal"
        elif "daily steps" in text:
            kind = "steps"
        else:
            kind = "other"
        if kind in seen_kind:
            continue
        seen_kind.add(kind)
        picked.append(text)
        if len(picked) >= 4:
            break

    if not picked:
        picked.append(
            "From here, small shifts in sleep, movement, or steps may not move your score "
            "much—a steady sleep schedule and gradual activity still tend to help how you feel."
        )

    if sleep_duration < 7:
        picked.append(
            "Many people feel best with about 7–9 hours of sleep per night; you're under 7 hours here."
        )
    if daily_steps < 5000:
        picked.append(
            "If your step count is on the lower side, short walks added over time can support energy and rest."
        )

    return picked
