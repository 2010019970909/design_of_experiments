import random
# import string

chars = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~" #string.printable.replace('\n', '').replace('\r', '').replace('\t', '').replace(' ', '')

def pwd(length=10):
    """Generate a random string of fixed length """    
    return ''.join(random.choice(chars) for i in range(length))

if __name__ == "__main__":
    for i in range(1000):
        print(pwd(26))
