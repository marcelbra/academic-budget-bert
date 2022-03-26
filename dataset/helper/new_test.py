def main():
    global test_object
    test_object = {i:str(i) for i in range(10)}

def foo():
    print(test_object[3])

main()
foo()