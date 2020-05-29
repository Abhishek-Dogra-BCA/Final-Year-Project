import tkinter as tk
from tkinter import *
import subprocess

win=tk.Tk()
win.geometry("500x500")
win.configure(bg="white")


def expressionRecognition():
    subprocess.call([r'D:\Media\Education\Project\opencv-face-recognition\video_recognition.bat'])
    
def trainModel():
    subprocess.call([r'D:\Media\Education\Project\opencv-face-recognition\train_model_expressions.bat'])

def extractEmbeddings():
    subprocess.call([r'D:\Media\Education\Project\opencv-face-recognition\extract_embeddings_expressions.bat'])

a = Button(win, text="Extract Embeddings", command=extractEmbeddings)
b = Button(win, text="Train Model", command=trainModel)
c = Button(win, text="Expression Recognition", command=expressionRecognition)

a.configure(font=('Calibri','14','bold'),background = 'skyblue', foreground = 'white')
b.configure(font=('Calibri','14','bold'),background = 'skyblue', foreground = 'white')
c.configure(font=('Calibri','14','bold'),background = 'skyblue', foreground = 'white')

a.pack(side=TOP, pady=50)
b.pack(side=TOP, pady=50)
c.pack(side=TOP, pady=50)


win.mainloop()
