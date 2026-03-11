import datetime

def get_response(user_input):

    user_input = user_input.lower()

    current_time = datetime.datetime.now().strftime("%I:%M %p")

    if "hello" in user_input or "hi" in user_input:
        return "Hi"

    elif "how are you" in user_input or "how are you doing" in user_input:
        return "I'm fine, thank you."

    elif "name" in user_input:
        return "My name is HelpBot"

    elif "age" in user_input or "how old are you" in user_input:
        return "I don't Have any age, I am a robot"

    elif "time" in user_input:
        return f"The time is: {current_time}"

    else:
        return "I don't know how to respond to your input"