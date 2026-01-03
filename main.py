# Main.py ============================================================
from CustomerJourneySystem import CustomerJourneySystem
import pandas as pd

# ============================================================
# 1. Initialize the system and load data
# ============================================================
file_path = input("Enter the path to your Excel file: ").strip()
system = CustomerJourneySystem(file_path)
system.build_sequences()
system.train_decision_tree()
system.precompute_top_actions()

# ============================================================
# 2. Show current top actions before adding new ones
# ============================================================
print("\n--- Current Top Actions ---")
print("By Country:", system.top_by_country)
print("By Solution:", system.top_by_solution)
print("By Country + Solution:", system.top_by_country_solution)

# ============================================================
# 3. Allow user to add new actions:
# ============================================================
while True:
    add_more = input("\nDo you want to add a new action? (yes/no): ").strip().lower()
    if add_more != "yes":
        break

    account_id = input("Enter account_id: ").strip()
    action_now = input("Enter current action (action_now): ").strip()
    action_next = input("Enter next action (action_next): ").strip()
    country = input("Enter Country: ").strip()
    solution = input("Enter Solution: ").strip()
    result = input("Enter Result (e.g., Won, Loss, Unknown): ").strip()
    sourcesystem = input("Enter SourceSystem (optional, default='Unknown'): ").strip() or "Unknown"
    who_id = input("Enter Who_id (optional, default='Unknown'): ").strip() or "Unknown"
    opportunity_id = input("Enter Opportunity_id (optional, default='Unknown'): ").strip() or "Unknown"
    is_lead = input("Enter Is_lead (optional, default=0): ").strip()
    is_lead = int(is_lead) if is_lead.isdigit() else 0

    # Add new action and recalc top actions
    top_actions = system.add_action_and_recalculate(
        account_id, action_now, action_next, country, solution, result,
        sourcesystem, who_id, opportunity_id, is_lead
    )

    print("\n--- Top Actions After Adding ---")
    print("By Country:", top_actions["country"])
    print("By Solution:", top_actions["solution"])
    print("By Country + Solution:", top_actions["country_solution"])

    # Build best trip
    print("\n--- Building Best Trip for the Opportunity ---")
    best_trip = system.build_best_trip(
        country, solution, initial_action=action_now, result=result,
        sourcesystem=sourcesystem, who_id=who_id, opportunity_id=opportunity_id, is_lead=is_lead
    )
    print(" -> ".join(best_trip))

    # Save updated sequences to Excel
    output_file = "updated_data.xlsx"
    system.df_seq.to_excel(output_file, index=False)
    print(f"\nAll actions saved to '{output_file}'")
