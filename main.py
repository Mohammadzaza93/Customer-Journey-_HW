from customer_journey_system import CustomerJourneySystem

# ======================================================
# 1️⃣ Load the Customer Journey System
# ======================================================
file_path = input("Enter the path to your Excel file: ").strip()
system = CustomerJourneySystem(file_path)

# ======================================================
# 2️⃣ Train the Win Probability Model
# ======================================================
system.train_win_probability_model()

# ======================================================
# 3️⃣ Ask the user to input the opportunity snapshot
# ======================================================
print("\n--- Enter Opportunity Snapshot ---")
total_activities = int(input("Total activities: "))
unique_activities = int(input("Unique activities: "))
last_action = input("Last action: ").strip()
country = input("Country: ").strip()
solution = input("Solution: ").strip()
is_lead = int(input("Is lead? (1 for Yes, 0 for No): "))

opportunity_snapshot = {
    "total_activities": total_activities,
    "unique_activities": unique_activities,
    "last_action": last_action,
    "country": country,
    "solution": solution,
    "is_lead": is_lead
}

# ======================================================
# 4️⃣ Calculate Win Probability
# ======================================================
prob = system.predict_win_probability(opportunity_snapshot)
print("\nWin Probability:", round(prob, 2))

# ======================================================
# 5️⃣ Display Top 5 Next Actions (case-insensitive)
# ======================================================
print("\n--- Top 5 Next Actions ---")
# Filter is case-insensitive
df = system.df_seq.copy()
df["is_won"] = df["result"].str.contains("won", case=False, na=False).astype(int)

filt = df[
    (df["country"].str.lower() == country.lower()) &
    (df["solution"].str.lower() == solution.lower()) &
    (df["action_now"].str.lower() == last_action.lower())
]

if filt.empty:
    print("No matching activities found.")
else:
    scores = filt.groupby("action_next").agg(
        frequency=("action_next", "count"),
        win_rate=("is_won", "mean")
    )
    scores["score"] = scores["frequency"] * scores["win_rate"]
    print(scores.sort_values("score", ascending=False).head(5))

# ======================================================
# 6️⃣ Display Top 5 Paths
# ======================================================
print("\n--- Top 5 Paths ---")
best_paths = system.get_best_paths()
print(best_paths)

# ======================================================
# 7️⃣ Display Top 4 Actions dynamically
# ======================================================
print("\n--- Top 4 Actions by Country ---")
top_by_country = system.get_top_actions(by="country", top_n=4)
print(top_by_country)

print("\n--- Top 4 Actions by Solution ---")
top_by_solution = system.get_top_actions(by="solution", top_n=4)
print(top_by_solution)

print("\n--- Top 4 Actions by Country and Solution ---")
top_by_country_solution = system.get_top_actions(by="country_solution", top_n=4)
print(top_by_country_solution)
