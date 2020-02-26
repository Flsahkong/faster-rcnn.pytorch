class Father():
    def __init__(self):
        print("我是爸爸")

    def a(self):
        print("爸爸，a")

class Me(Father):
    def __init__(self):
        print("me")

        Father.__init__(self)
        Father.a(self)
        print("--------")
        self.a()
        print("--------")

    def a(self):
        print("me,a")

if __name__ == '__main__':
    me = Me()
    me.a()
