import ctypes

# Load SecuGen Fingerprint Library
sgfplib = ctypes.WinDLL("C:\\Program Files\\SecuGen\\SDK\\Inc\\SGFPLib.dll")

# Initialize library
sgfplib.Create()
print("SecuGen SDK initialized successfully!")
