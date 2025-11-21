import argparse

print("Imports done.")

parser = argparse.ArgumentParser(description="main MASA Tracker. Typical usage: python main.py -camera 20936 -rtsp <rtsp_url> -gui -tracking -kalman_only")

# Definisco gli argomenti obbligatori
parser.add_argument("-gui", action="store_true", help="Enable GUI")
parser.add_argument("-no_tracking", action="store_false", help="Disable tracking")

args = parser.parse_args()  

GUI = args.gui
TRACKING = args.no_tracking


print(GUI)
print(TRACKING)