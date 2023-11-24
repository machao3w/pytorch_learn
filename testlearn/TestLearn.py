class Person(object):
    def __init__(self, first, last, age):
        self.first = first
        self.last = last
        self.age = age
        self.__idCard = '123'

    def getIdCard(self):
        return self.__idCard

onePerson = Person('Mertle', 'Sedgewick', 52)
print(onePerson.first)
print(onePerson.getIdCard())

print("1 2 3 4  5".split())
name,age = 'John',10
f'{name} is {age} years old'

a = [1001, 'a','b','c']
a.extend([1,2,3])
b=(10)
c=(10,1)
type(b)
type(c)





