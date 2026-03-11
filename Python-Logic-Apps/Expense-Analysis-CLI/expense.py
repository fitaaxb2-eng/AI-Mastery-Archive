class Expense:
    """
    Represents a single financial expense record.
    """
    def __init__(self, date, category, amount):
        # Initialize the expense object with date, category, and amount
        self.date = date
        self.category = category
        self.amount = amount

    def __str__(self):
        # Returns a string representation of the expense for display
        return f'{self.date} {self.category} {self.amount}'

# Example usage: Creating an instance of the Expense class
exp_1 = Expense(date='2020-01-10', category='Income', amount=20000)