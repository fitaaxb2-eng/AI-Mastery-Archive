from manager import ExpenseManager
from expense import Expense
from visualizer import ExpenseVisualizer


def main():
    """
    Main function to run the Expense Tracker CLI application.
    """
    manager = ExpenseManager()

    while True:
        # Display the main menu
        print("\n=== Expense Tracker Menu ===")
        print("1. Add New Expense")
        print("2. View Expense Statistics (Total & Average)")
        print("3. Visualize Data (Graph)")
        print("4. Exit")

        choice = input("\nPlease enter your choice (1-4): ")

        if choice == "1":
            # Collect user input for new expense
            date = input("Enter Date (YYYY-MM-DD): ")
            category = input("Enter Category (e.g., Food, Transport): ")

            try:
                # Validate that amount is a number
                amount = float(input("Enter Amount: "))

                # Create Expense object and add to manager
                new_exp = Expense(date, category, amount)
                manager.add_expense(new_exp)  # Updated method name to match manager.py
                manager.save_expenses()  # Updated method name to match manager.py

                print("\n[Success] Expense added successfully!")
            except ValueError:
                print("\n[Error] Invalid amount. Please enter a numeric value.")

        elif choice == "2":
            # Display statistics
            total = manager.get_total()
            average = manager.get_average()

            print('\n--- Expense Statistics ---')
            print(f"Total Expenses:   ${total:,.2f}")  # Formatted with commas and 2 decimals
            print(f"Average Expense:  ${average:,.2f}")

        elif choice == "3":
            # Visualize data using the Visualizer class
                viz = ExpenseVisualizer(manager.expenses)
                viz.plot_expenses()
        elif choice == "4":
            # Exit the application
            print("Exiting... Thank you for using Expense Tracker!")
            break
        else:
            print("\n[Invalid] Please choose a valid option (1-4).")


if __name__ == "__main__":
    main()