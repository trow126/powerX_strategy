my_list = [0.2, 0.5, 0.8, 0.3, 0.7, 0.9]

if any(x < 0.1 or x >= 0.9 for x in my_list):
    print("NGです")
else:
    print("OKです")
