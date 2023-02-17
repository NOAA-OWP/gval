import os

# Get absolute path of file
abs_path = os.path.dirname(os.path.abspath(__file__))

TEST_DATA = os.path.join(abs_path, "data", "data")
AWS_KEYS = os.path.join(abs_path, "data", "access_keys.csv")


if __name__ == "__main__":
    print("Docker works from cmd")
