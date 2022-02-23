from os import path
if __name__ == '__main__':
    print(path.dirname( path.dirname( path.abspath(__file__) ) ))