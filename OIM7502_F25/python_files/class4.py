import hashlib
class User:

    def __init__(self,Username=None ,password=None ):
        self.Username=Username
        self.password = self._encrypt_password(password)

    def __str__(self):
        return f'Username: {self.Username}\nPassword: {self.password}'

    def __repr__(self):
        return f"{self.__class__}({self.__dict__})"

    def __eq__(self,other):
        return self.Username == other.Username and self.password == other.password
# an underscore before the method
    def _encrypt_password(self,password):
        password=password.encode('utf-8')
        return hashlib.sha256(password).hexdigest()

    def _check_password(self,password):
        password=password.encode('utf-8')


#intialiser

if  __name__=="__main__":
    user = User("Maria", "password")
    other=User("Maria", "password")
    print(user)
    print(User)
    print(dir(User))
    print(repr(user))
    print(user==other)
    print ()


