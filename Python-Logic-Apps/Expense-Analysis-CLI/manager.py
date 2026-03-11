from expense import Expense
import os

class ExpenseManager:
    """
    Manages the collection of expenses.
    Handles adding, saving, loading, and calculating expense statistics.
    """

    def __init__(self):
        # Initialize an empty list and load existing data from file
        self.expenses = []
        self.load_expenses()

    def add_expense(self, expense):
        # Adds a new expense object to the list
        self.expenses.append(expense)

    def get_total(self):
        # Calculates the sum of all expense amounts
        total = 0
        for expense in self.expenses:
            total += expense.amount
        return total

    def get_average(self):
        # Calculates the average expense amount
        if not self.expenses:
            return 0
        return self.get_total() / len(self.expenses)

    def save_expenses(self):
        # Ensures the data directory exists
        if not os.path.exists("data"):
            os.mkdir("data")

        # Saves all expenses to a text file (overwrites existing file)
        with open("data/expenses.txt", "w") as wf:
            for expense in self.expenses:
                # Format: Date, Category, Amount
                line = f'{expense.date},{expense.category},{expense.amount}\n'
                wf.write(line)

    def load_expenses(self):
        # Loads expenses from the text file into memory
        self.expenses = []  # Clear list to avoid duplicates
        try:
            if os.path.exists("data/expenses.txt"):
                with open("data/expenses.txt", "r") as rf:
                    lines = rf.readlines()
                    for line in lines:
                        # Parse the CSV line
                        data = line.strip().split(',')
                        if len(data) == 3:
                            # Create Expense object and append to list
                            load_exp = Expense(
                                date=data[0].strip(),
                                category=data[1].strip(),
                                amount=float(data[2].strip())
                            )
                            self.expenses.append(load_exp)
        except Exception as e:
            print(f"Error loading expenses: {e}")