class  Employee:
    #Constructor for the class
    increment = 1.5
    num_Employee = 0
    def __init__(self,fname,lname,salary):
        self.fname = fname
        self.lname = lname
        self.salary = salary
        Employee.num_Employee += 1

    def increase(self):
        self.salary=  self.salary*Employee.increment
    @classmethod
    def change_increment(cls,amount):
        cls.increment = amount
    @classmethod
    def from_str(cls,emp_string):
        fname,lname,salary=emp_string.split("-")
        return cls(fname,lname,salary)
    @staticmethod
    def isopen(day):
        if(day=="Sunday"):
            print("THe Office is closed today")
        else:
            print("Office is open today")

class Programmer(Employee):
    pass
harry = Employee("harry","Jackson",40000)
rohan = Employee("Rohan","Das",50000)

harry.increase()  #by defauklt one positional argument is passed whenever the objectr calls a fucntion

print(harry.fname,harry.salary)
print(Employee.num_Employee)
Employee.change_increment(2)
harry.increase()
print("Updated Salary : ",harry.salary)

hrishikesh = Programmer.from_str("hrishikesh-karande-45000")

hrishikesh.increase()
print(hrishikesh.fname,hrishikesh.lname,hrishikesh.salary)
