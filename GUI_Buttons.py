from tkinter import *

class Window(Frame):


    def __init__(self, master=None):
        Frame.__init__(self, master)                 
        self.master = master
        self.init_window()

    #Creation of window
    def init_window(self):

        # changing the title of Window      
        self.master.title("GUI")

        self.pack(fill=BOTH, expand=1)
        
        w = Label(self, text="Mini Spotify",height = 2, width = 30)
        w.place(x=100,y=40)

        # creating a button B1
        B1 = Button(self, text="Song A",height = 2, width = 30)
        # placing the button 
        B1.place(x=90, y=100)
        
        # creating a button B2
        B2 = Button(self, text="Song B",height = 2, width = 30)
        # placing the button 
        B2.place(x=90, y=180)
        
        # creating a button B2
        B3 = Button(self, text="Song C",height = 2, width = 30)
        # placing the button
        B3.place(x=90, y=260)

root = Tk()

#size of the window
root.geometry("400x300")

app = Window(root)
root.mainloop()  