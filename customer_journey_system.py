import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class CustomerJourneySystem:
    def __init__(self, file_path: str):
        self.df = None
        self.df_seq = None
        self.win_model = None

        self.load_data(file_path)
        self.build_sequences()

    # ======================================================
    # 1. LOAD & CLEAN DATA
    # ======================================================
    def load_data(self, file_path):
        self.df = pd.read_excel(file_path, engine="openpyxl")

        self.df.columns = (
            self.df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )

        self.df.rename(columns={"types": "action"}, inplace=True)

        self.df["action"] = self.df["action"].fillna("Unknown").astype(str)
        self.df["activity_date"] = pd.to_datetime(
            self.df["activity_date"], errors="coerce"
        )

        for col in ["country", "solution", "opportunity_stage"]:
            self.df[col] = self.df[col].fillna("Unknown").astype(str)

        if "is_lead" not in self.df.columns:
            self.df["is_lead"] = 0

        self.df = self.df.sort_values(
            ["opportunity_id", "activity_date"]
        )

    # ======================================================
    # 2. BUILD ACTION SEQUENCES
    # ======================================================
    def build_sequences(self):
        rows = []

        for opp_id, g in self.df.groupby("opportunity_id"):
            actions = g["action"].tolist()

            for i in range(len(actions) - 1):
                rows.append({
                    "opportunity_id": opp_id,
                    "action_now": actions[i],
                    "action_next": actions[i + 1],
                    "step": i + 1,
                    "total_steps": len(actions),
                    "country": g["country"].iloc[0],
                    "solution": g["solution"].iloc[0],
                    "result": g["opportunity_stage"].iloc[-1],
                    "is_lead": g["is_lead"].iloc[0]
                })

        self.df_seq = pd.DataFrame(rows)

    # ======================================================
    # 3. WIN PROBABILITY MODEL
    # ======================================================
    def _build_opportunity_features(self):
        df = self.df.copy()

        df["is_won"] = df["opportunity_stage"].str.contains(
            "won", case=False, na=False
        ).astype(int)

        agg = df.groupby("opportunity_id").agg(
            total_activities=("action", "count"),
            unique_activities=("action", "nunique"),
            last_action=("action", "last"),
            country=("country", "first"),
            solution=("solution", "first"),
            is_lead=("is_lead", "first"),
            is_won=("is_won", "max")
        ).reset_index()

        return agg

    def train_win_probability_model(self):
        df_feat = self._build_opportunity_features()

        X = df_feat.drop(["opportunity_id", "is_won"], axis=1)
        y = df_feat["is_won"]

        cat_cols = ["last_action", "country", "solution"]
        num_cols = ["total_activities", "unique_activities", "is_lead"]

        preprocessor = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols)
        ])

        self.win_model = Pipeline([
            ("prep", preprocessor),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                n_jobs=-1
            ))
        ])

        self.win_model.fit(X, y)

    def predict_win_probability(self, opportunity_snapshot: dict) -> float:
        df = pd.DataFrame([opportunity_snapshot])
        return self.win_model.predict_proba(df)[0][1]

    # ======================================================
    # 4. NEXT BEST ACTION (TOP 5)
    # ======================================================
    def get_best_next_actions(self, country, solution, action_now, top_n=5):
        df = self.df_seq.copy()
        df["is_won"] = df["result"].str.contains("won", case=False, na=False).astype(int)

        filt = df[
            (df["country"].str.lower() == country.lower()) &
            (df["solution"].str.lower() == solution.lower()) &
            (df["action_now"].str.lower() == action_now.lower())
        ]

        if filt.empty:
            return pd.DataFrame()

        scores = filt.groupby("action_next").agg(
            frequency=("action_next", "count"),
            win_rate=("is_won", "mean")
        )

        scores["score"] = scores["frequency"] * scores["win_rate"]

        return scores.sort_values("score", ascending=False).head(top_n)

    # ======================================================
    # 5. REMOVE CONSECUTIVE DUPLICATES
    # ======================================================
    @staticmethod
    def _remove_consecutive_duplicates(actions):
        if not actions:
            return actions

        cleaned = [actions[0]]
        for a in actions[1:]:
            if a != cleaned[-1]:
                cleaned.append(a)
        return cleaned

    # ======================================================
    # 6. BEST PATHS (TOP 5)
    # ======================================================
    def get_best_paths(self, min_occurrences=5, top_n=5, min_length=3):
        def clean_path(x):
            return self._remove_consecutive_duplicates(x.tolist())

        paths = (
            self.df
            .groupby("opportunity_id")
            .agg(
                actions=("action", clean_path),
                is_won=("opportunity_stage",
                        lambda x: any(x.astype(str).str.contains("won", case=False)))
            )
            .reset_index()
        )

        paths = paths[paths["is_won"]]
        paths["length"] = paths["actions"].apply(len)
        paths = paths[paths["length"] >= min_length]
        paths["path"] = paths["actions"].apply(lambda x: " → ".join(x))

        best = (
            paths.groupby("path")
            .size()
            .reset_index(name="count")
            .query("count >= @min_occurrences")
            .sort_values("count", ascending=False)
            .head(top_n)
        )

        return best

    # ======================================================
    # 7. TOP 4 ACTIONS DYNAMIC (NEW WEIGHT)
    # ======================================================
    def get_top_actions(self, by="country", top_n=4, last_touch_weight=0.3):
        """
        by: "country", "solution", "country_solution"
        last_touch_weight: factor for last step influence
        """
        df = self.df_seq.copy()
        df["is_won"] = df["result"].str.contains("won", case=False, na=False).astype(int)

        if by == "country_solution":
            group_cols = ["country", "solution", "action_now"]
        elif by == "solution":
            group_cols = ["solution", "action_now"]
        else:
            group_cols = ["country", "action_now"]

        scores = df.groupby(group_cols).agg(
            frequency=("action_now", "count"),
            win_rate=("is_won", "mean"),
            last_touch=("step", lambda x: (x == x.max()).mean())
        ).reset_index()

        # New weighted score: BaseWeight * (1 - last_touch_weight + last_touch * last_touch_weight)
        scores["score"] = (scores["frequency"] * scores["win_rate"]) * (1 - last_touch_weight + scores["last_touch"] * last_touch_weight)

        top_actions = (
            scores.sort_values("score", ascending=False)
            .groupby(group_cols[:-1])  
            .head(top_n)
            .reset_index(drop=True)
        )

        return top_actions

    # ======================================================
    # 8. ADD NEW ACTIVITY (UPDATES SEQUENCES AND TOP ACTIONS)
    # ======================================================
    def add_activity(self, opportunity_id, action, country, solution, step, total_steps, result, is_lead):
        new_row = {
            "opportunity_id": opportunity_id,
            "action_now": action,
            "action_next": None,  
            "step": step,
            "total_steps": total_steps,
            "country": country,
            "solution": solution,
            "result": result,
            "is_lead": is_lead
        }
        self.df_seq = pd.concat([self.df_seq, pd.DataFrame([new_row])], ignore_index=True)
        mask_prev = (self.df_seq["opportunity_id"] == opportunity_id) & (self.df_seq["step"] == step - 1)
        self.df_seq.loc[mask_prev, "action_next"] = action
