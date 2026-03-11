from book import Book
import os

class Library:
    # Initialize the Library: create list and load existing data
    def __init__(self):
        self.books = []
        self.load_books()

    # Method to add a new book to the library
    def add_books(self, book):
        try:
            self.books.append(book)
            self.save_books()  # Save changes to file immediately
            print(f"Success: '{book.title}' has been added to the library.")
        except Exception as e:
            print(f"Error adding book: {e}")

    # Method to remove a book by its title
    def remove_books(self, title_to_remove):
        found_book = False
        try:
            for book in self.books:
                # Comparing titles (stripped of whitespace)
                if book.title.lower() == title_to_remove.strip().lower():
                    self.books.remove(book)
                    found_book = True
                    break

            if found_book:
                self.save_books()  # Update the file after removal
                print(f"Success: Book '{title_to_remove}' removed.")
            else:
                print(f"Error: Book '{title_to_remove}' not found.")
        except Exception as e:
            print(f"An error occurred while removing: {e}")

    # Method to list all books currently in the library
    def list_books(self):
        if not self.books:
            print('No books available in the library.')
        else:
            print(f"Found {len(self.books)} Books:")
            print("-" * 30)  # Prints a separator line
            for book in self.books:
                print(book)
            print("-" * 30)

    # Method to save the current list of books to a text file
    def save_books(self):
        try:
            # Create directory if it doesn't exist
            if not os.path.exists('data_test'):
                os.mkdir('data_test')

            with open('data_test/books.txt', 'w') as f:
                for book in self.books:
                    # Format: Title,Author,Year (Added \n for new line)
                    line = f'{book.title},{book.author},{book.year}\n'
                    f.write(line)
        except Exception as e:
            print(f"Error saving data: {e}")

    # Method to load books from the text file into the program
    def load_books(self):
        self.books = []  # Clear list to avoid duplicates
        try:
            with open('data_test/books.txt', 'r') as f:
                lines = f.readlines()
                for line in lines:
                    data = line.strip().split(',')
                    # Ensure the data line has exactly 3 parts (Title, Author, Year)
                    if len(data) == 3:
                        book_load = Book(title=data[0].strip(), author=data[1].strip(), year=data[2].strip())
                        # Appending directly to list to avoid recursive saving
                        self.books.append(book_load)
        except FileNotFoundError:
            # If file doesn't exist yet, just ignore (it will be created later)
            pass
        except Exception as e:
            print(f"Error loading data: {e}")