import tkinter as tk

Screen = tk.Tk()
Screen.title("User_Screen")
Screen.geometry("200x200")
Screen.config(bg="blue")

Predict_Button = tk.Button(Screen,text="Predict",bg="orange",fg="green")
Predict_Button.place(x=150,y=0)

Delete_Button = tk.Button(Screen,text="Delete",bg="orange",fg="green")
Delete_Button.place(x=100,y=0)

Canvas = tk.Canvas(Screen,width=100,height=100,background="cyan")
Canvas.place(x=0,y=0)
def Draw(event):
    Canvas.create_oval(50,50,80,80)
Canvas.bind("<B1-Motion>",Draw)

Screen.mainloop()