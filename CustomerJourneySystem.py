import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
import os

class CustomerJourneySystem:
    def __init__(self, file_path=None):
        self.df = None
        self.df_seq = None
        self.ohe = None
        self.clf = None
        self.action_weights = {} 
        
        self.top_by_country = pd.DataFrame()
        self.top_by_solution = pd.DataFrame()
        self.top_by_country_solution = pd.DataFrame()

        if file_path:
            self.load_data(file_path)

    def load_data(self, file_path):
        self.df = pd.read_excel(file_path, engine="openpyxl")
        self.df.columns = [c.strip().lower().replace(" ", "_") for c in self.df.columns]
        self.df.rename(columns={"types": "action"}, inplace=True)
        self.df["action"] = self.df["action"].fillna("Unknown").astype(str)
        self.df["activity_date"] = pd.to_datetime(self.df["activity_date"], errors="coerce")
        for col in ["country", "solution", "opportunity_stage"]:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna("Unknown").astype(str)
        self.df = self.df.sort_values(["account_id", "activity_date"])

    def build_sequences(self):
        sequence_rows = []
        for acc, g in self.df.groupby("account_id"):
            actions = g["action"].tolist()
            for i in range(len(actions) - 1):
                sequence_rows.append({
                    "account_id": acc,
                    "action_now": str(actions[i]),
                    "action_next": str(actions[i+1]),
                    "step": i + 1,
                    "total_steps": len(actions),
                    "country": g["country"].iloc[0],
                    "solution": g["solution"].iloc[0],
                    "result": g["opportunity_stage"].iloc[-1],
                    "is_lead": g["is_lead"].iloc[0] if "is_lead" in g.columns else 0
                })
        self.df_seq = pd.DataFrame(sequence_rows)

    def train_decision_tree(self):
        cat_cols = ["action_now", "country", "solution", "result"]
        self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        X_cat = self.ohe.fit_transform(self.df_seq[cat_cols])
        X_num = self.df_seq[["step", "total_steps", "is_lead"]].fillna(0).values
        X = np.hstack([X_cat, X_num])
        y = self.df_seq["action_next"]
        self.clf = DecisionTreeClassifier(max_depth=10, random_state=42)
        self.clf.fit(X, y)
        
        features = list(self.ohe.get_feature_names_out()) + ["step", "total_steps", "is_lead"]
        importances = pd.Series(self.clf.feature_importances_, index=features).sort_values(ascending=False)
        print("\n--- Feature Importance Report ---")
        print(importances.head(5))

    def apply_dynamic_weight(self, action_name, is_first_touch, last_touch_weight=0.1):
        base_weight = self.action_weights.get(action_name, 1.0)
        if is_first_touch:
            adjusted_weight = base_weight
        else:
            adjusted_weight = base_weight * (1 - last_touch_weight)
        self.action_weights[action_name] = adjusted_weight
        return adjusted_weight

    def precompute_top_actions(self):
        def get_top_4(df, groups):
            counts = df.groupby(groups + ["action_next"]).size().reset_index(name='f')
            counts['score'] = counts.apply(lambda r: r['f'] * self.action_weights.get(r['action_next'], 1.0), axis=1)
            return counts.sort_values(groups + ['score'], ascending=False).groupby(groups).head(4)

        self.top_by_country = get_top_4(self.df_seq, ["country"])
        self.top_by_solution = get_top_4(self.df_seq, ["solution"])
        self.top_by_country_solution = get_top_4(self.df_seq, ["country", "solution"])
        
        return {
            "country": self.top_by_country,
            "solution": self.top_by_solution,
            "country_solution": self.top_by_country_solution
        }

    def add_action_and_recalculate(self, account_id, action_now, action_next, country, solution, result, 
                                   sourcesystem="Unknown", who_id="Unknown", opportunity_id="Unknown", is_lead=0):
        
        is_first = not (self.df_seq['account_id'] == account_id).any()
        self.apply_dynamic_weight(action_now, is_first)
        
        new_data = {
            "account_id": account_id, "action_now": action_now, "action_next": action_next,
            "country": country, "solution": solution, "result": result, 
            "step": 1, "total_steps": 2, "is_lead": is_lead
        }
        self.df_seq = pd.concat([self.df_seq, pd.DataFrame([new_data])], ignore_index=True)
        return self.precompute_top_actions()

    def build_best_trip(self, country, solution, initial_action, result, **kwargs):
        # تمثيل بسيط للمسار المتوقع بناءً على معطيات الإدخال
        trip = [initial_action, "Predicting Next Step...", result]
        return trip
    