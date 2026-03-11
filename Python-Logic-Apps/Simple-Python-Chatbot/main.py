from responses import get_response

print('If you want to Exit this program, type (1)')

first_time = True

while True:
    if first_time:
        prompt_text = "\nWhat would you like to ask me? "
    else:
        prompt_text = "\nWhat else would you like to ask? "

    ask = input(prompt_text).lower()

    first_time = False

    if ask == '1':
        print('Thank you for using this program. Bye!')
        break

    response = get_response(ask)
    print(response)