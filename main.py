import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

class ReturnRxSimple:
    """A simpler version of ReturnRx that works without Streamlit.
    This class provides core functionality for return analysis with console interface.
    """
    
    def __init__(self):
        """Initialize the ReturnRx application."""
        # Initialize data storage
        self.scenarios = pd.DataFrame(columns=[
            'scenario_name', 'sku', 'sales_30', 'avg_sale_price', 
            'sales_channel', 'returns_30', 'solution', 'solution_cost',
            'additional_cost_per_item', 'current_unit_cost', 'reduction_rate',
            'return_cost_30', 'return_cost_annual', 'revenue_impact_30',
            'revenue_impact_annual', 'new_unit_cost', 'savings_30',
            'annual_savings', 'break_even_days', 'break_even_months',
            'roi', 'score', 'timestamp'
        ])
        
        self.analytics = pd.DataFrame(columns=[
            'scenario_name', 'sku', 'solution', 'reduction_rate',
            'roi', 'annual_savings', 'solution_cost', 'break_even_days',
            'score', 'timestamp'
        ])
        
        # Define maximum number of scenarios
        self.MAX_SCENARIOS = 4
    
    def run(self):
        """Run the ReturnRx application in console mode."""
        print("\n" + "=" * 60)
        print("  RETURNRX ENTERPRISE - PRODUCT RETURN ANALYSIS TOOL")
        print("=" * 60)
        
        while True:
            print("\nMAIN MENU:")
            print("1. Add Scenario")
            print("2. View Scenarios")
            print("3. Clear Scenarios")
            print("4. Save to Analytics")
            print("5. View Analytics")
            print("6. Export Data")
            print("7. Help")
            print("8. Exit")
            
            choice = input("\nEnter your choice (1-8): ")
            
            if choice == '1':
                self.add_scenario_input()
            elif choice == '2':
                self.view_scenarios()
            elif choice == '3':
                self.clear_scenarios()
            elif choice == '4':
                self.save_to_analytics()
            elif choice == '5':
                self.view_analytics()
            elif choice == '6':
                self.export_data()
            elif choice == '7':
                self.show_help()
            elif choice == '8':
                print("\nThank you for using ReturnRx Enterprise. Goodbye!")
                break
            else:
                print("\nInvalid choice. Please try again.")
    
    def add_scenario_input(self):
        """Get scenario input from user."""
        print("\n" + "=" * 60)
        print("  ADD NEW SCENARIO")
        print("=" * 60)
        
        # Check if maximum scenarios reached
        if len(self.scenarios) >= self.MAX_SCENARIOS:
            print(f"\nMaximum of {self.MAX_SCENARIOS} scenarios reached. Please clear scenarios first.")
            return
        
        try:
            # Get scenario inputs
            scenario_name = input("\nScenario Name (or press Enter for default): ")
            sku = input("SKU: ")
            if not sku:
                print("SKU is required.")
                return
                
            sales_30 = float(input("Sales (30 days): "))
            avg_sale_price = float(input("Average Sale Price: "))
            sales_channel = input("Top Sales Channel: ")
            returns_30 = float(input("Returns (30 days): "))
            solution = input("Suggested Solution: ")
            solution_cost = float(input("Solution Total Cost: "))
            additional_cost_per_item = float(input("Additional Cost per Item (can be negative): "))
            current_unit_cost = float(input("Current Unit Cost: "))
            
            reduction_rate = input("Est. Return Rate Reduction % (0-100): ")
            reduction_rate = float(reduction_rate) if reduction_rate else 0
            
            if reduction_rate < 0 or reduction_rate > 100:
                print("Return Rate Reduction must be between 0 and 100.")
                return
            
            # Add scenario with validated inputs
            self.add_scenario(scenario_name, sku, sales_30, avg_sale_price, sales_channel, 
                             returns_30, solution, solution_cost, additional_cost_per_item, 
                             current_unit_cost, reduction_rate)
            
        except ValueError:
            print("\nError: Please enter valid numeric values.")
        except Exception as e:
            print(f"\nError: {str(e)}")
    
    def add_scenario(self, scenario_name, sku, sales_30, avg_sale_price, sales_channel, 
                 returns_30, solution, solution_cost, additional_cost_per_item, 
                 current_unit_cost, reduction_rate):
    """Calculate metrics and add a scenario to the data."""
    
    # Generate a default scenario name if not provided
    if not scenario_name:
        scenario_name = f"Scenario {len(self.scenarios) + 1}"

    # Calculate costs and impacts
    return_cost_30 = returns_30 * current_unit_cost
    return_cost_annual = return_cost_30 * 12
    revenue_impact_30 = returns_30 * avg_sale_price
    revenue_impact_annual = revenue_impact_30 * 12
    new_unit_cost = current_unit_cost + additional_cost_per_item

    # Calculate benefits and ROI
    savings_30 = returns_30 * (reduction_rate / 100) * avg_sale_price
    annual_savings = savings_30 * 12

    roi = None
    break_even_days = None
    break_even_months = None
    score = None

    if solution_cost > 0 and annual_savings > 0:
        roi = annual_savings / solution_cost

        # Calculate break-even with additional item costs factored in
        annual_additional_costs = additional_cost_per_item * sales_30 * 12

        if (annual_savings - annual_additional_costs) > 0:
            break_even_days = solution_cost / (annual_savings - annual_additional_costs)
            break_even_months = break_even_days / 30
            score = roi * 100 - break_even_days

    # Create the scenario row
    new_row = {
        'scenario_name': scenario_name,
        'sku': sku,
        'sales_30': sales_30,
        'avg_sale_price': avg_sale_price,
        'sales_channel': sales_channel,
        'returns_30': returns_30,
        'solution': solution,
        'solution_cost': solution_cost,
        'additional_cost_per_item': additional_cost_per_item,
        'current_unit_cost': current_unit_cost,
        'reduction_rate': reduction_rate,
        'return_cost_30': return_cost_30,
        'return_cost_annual': return_cost_annual,
        'revenue_impact_30': revenue_impact_30,
        'revenue_impact_annual': revenue_impact_annual,
        'new_unit_cost': new_unit_cost,
        'savings_30': savings_30,
        'annual_savings': annual_savings,
        'break_even_days': break_even_days,
        'break_even_months': break_even_months,
        'roi': roi,
        'score': score,
        'timestamp': datetime.now()
    }

    # Append to the DataFrame
    self.scenarios = pd.concat([self.scenarios, pd.DataFrame([new_row])], ignore_index=True)
    print(f"\n✅ Scenario '{scenario_name}' added successfully.")
