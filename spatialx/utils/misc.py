import sys
from colorama import Fore, Style


def print_with_color(*values: object, color="", end="\n", reset=True):
    if color == "red":
        print(Fore.RED, *values, sep="", end=end)
    elif color == "green":
        print(Fore.GREEN, *values, sep="", end=end)
    elif color == "yellow":
        print(Fore.YELLOW, *values, sep="", end=end)
    elif color == "blue":
        print(Fore.BLUE, *values, sep="", end=end)
    elif color == "magenta":
        print(Fore.MAGENTA, *values, sep="", end=end)
    elif color == "cyan":
        print(Fore.CYAN, *values, sep="", end=end)
    elif color == "white":
        print(Fore.WHITE, *values, sep="", end=end)
    elif color == "black":
        print(Fore.BLACK, *values, sep="", end=end)
    else:
        print(*values, end=end)
    
    if reset: print(Style.RESET_ALL, end="")
    sys.stdout.flush()
