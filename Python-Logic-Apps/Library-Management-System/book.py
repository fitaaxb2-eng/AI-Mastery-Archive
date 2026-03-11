class Book:
    # Constructor method to initialize the object's attributes
    def __init__(self, title, author, year):
        self.title = title
        self.author = author
        self.year = year

    # Method to return a readable string representation of the object
    def __str__(self):
        return f'{self.title} by {self.author}, {self.year}'

# Using try-except block to handle potential runtime errors
try:
    # Creating an instance of the Book class with corrected capitalization
    book1 = Book('Geele Book', 'Axmed Xaashi', '1991')

    # Printing the book object (this calls the __str__ method automatically)
    print(book1)

except Exception as e:
    # If an error occurs, print a friendly error message
    print(f"Something went wrong: {e}")