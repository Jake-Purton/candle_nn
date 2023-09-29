from guizero import App, PushButton, Text
import random

with open("Michael.txt", "r") as f:
    global names
    names = f.readlines()

colour = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
string = ""

def not_purple ():

    global textbox
    global app
    global colour
    global string

    string += f"{colour[0]} {colour[1]} {colour[2]} 0\n"

    colour = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    app.bg = colour
    textbox.value = "Name: " + names[random.randint(0, len(names))]
    

def purple():
    global textbox
    global app
    global colour
    global string

    string += f"{colour[0]} {colour[1]} {colour[2]} 1\n"

    colour = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    app.bg = colour
    textbox.value = "Name: " + names[random.randint(0, len(names))]

def printem():
    global string
    print(string)

app = App("Enter your name")

global textbox
textbox = Text(app, text="Name: ")

global button
button = PushButton(app, command=not_purple)
button.text = "not purple"

global button1
button1 = PushButton(app, command=purple)
button1.text = "purple"

global button2
button2 = PushButton(app, command=printem)
button2.text = "print"


app.display()