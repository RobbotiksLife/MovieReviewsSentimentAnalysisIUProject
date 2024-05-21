from colorama import init as init_colorama, Fore
# Initialize colorama
init_colorama()


def print_colored_text(text, color: Fore = Fore.RED):
    print(color + text + Fore.RESET)


