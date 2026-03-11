from library import Library
from book import Book

# Initialize the Library system (loads existing books automatically)
my_library = Library()

while True:
    print('\n___ Library Management System ___')
    print('1. Add Book')
    print('2. Remove Book')
    print('3. List Books')
    print('4. Exit')

    choice = input('\nEnter your choice (1-4): ')

    if choice == '1':
        # Adding a new book
        # .title() makes the first letter capital automatically (e.g., "ali" -> "Ali")
        title = input('Enter title: ').strip().title()
        author = input('Enter author: ').strip().title()

        try:
            # Validating that year is a number
            year_input = int(input('Enter year: '))

            # Create Book object
            new_book = Book(title, author, str(year_input))

            # Add to library (Auto-save handles the saving part)
            my_library.add_books(new_book)

        except ValueError:
            print('Error: Invalid year! Please enter a number (e.g., 2023).')

    elif choice == '2':
        # Removing a book
        del_title = input('Enter Book Title to remove: ').strip()
        my_library.remove_books(del_title)

    elif choice == '3':
        # Listing all books
        my_library.list_books()

    elif choice == '4':
        # Exiting the program
        print('Exiting system. Thank you!')
        break

    else:
        # Handling invalid menu choices
        print('Invalid choice. Please enter a number between 1 and 4.')