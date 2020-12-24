import Lab2_num as ln
# import Stiff linear system
import matplotlib
import matplotlib.pyplot as plt
# from itertools import count, izip
import pandas as pd

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import *
from tkinter import messagebox as mb
from tkinter import ttk
import pylab
from tkinter import filedialog as fd
from PIL import ImageTk, Image


class RungeKuttaGUI:

    def __init__(self, window):
        self.window = window
        self.fig = plt.figure()
        self.graph_axes = self.fig.add_subplot(111)

        self.fig.subplots_adjust(left=0.10, right=0.95, top=0.95, bottom=0.2)

    def run(self):
        self.window.title("Лабораторная работа 2")
        self.window.geometry("600x300")
        self.mount_components()

    def mount_components(self):
        self.button = ttk.Button(master=self.window, command=self.on_click)
        self.button.configure(text="График")
        self.button.place(x=50, y=0)

        # self.button1 = ttk.Button(master=self.window, command=self.table)
        # self.button1.configure(text="Таблица")
        # self.button1.place(x=50, y=30)
        #
        # self.button2 = ttk.Button(master=self.window, command=self.insert_text)
        # self.button2.configure(text="Условие задачи")
        # self.button2.place(x=250, y=0)


        # self.r_var = IntVar()
        # self.r_var.set(0)
        # self.r2 = Radiobutton(tk, text="Func 1", variable=self.r_var, value=1)

        self.r_var = IntVar()
        self.r_var.set(1)
        self.r1 = Radiobutton(tk, text='Test Func', variable=self.r_var, value=1)
        self.r2 = Radiobutton(tk, text='Main Func', variable=self.r_var, value=0)
        # self.r3 = Radiobutton(tk, text='Func 2', variable=self.r_var, value=2, command = self.sel)
        self.r1.place(x = 200, y = 0)
        self.r2.place(x = 200, y = 25)
        # self.r3.place(x = 200, y = 50)



        self.iter_num_label = ttk.Label(self.window)
        self.iter_num_label.configure(text="Число разбиений равномерной сетки")
        self.iter_num_label.place(x=305, y=25)

        self.iter_num_entry = ttk.Entry(self.window, width=10)
        self.iter_num_entry.insert(END, 600)
        self.iter_num_entry.place(x=305, y=50)
        #



    def on_click(self):
        n = int(self.iter_num_entry.get())
        r_var = bool(self.r_var.get())

        y, res1, res2, e = ln.data(r_var, n)
        data = []
        for i in range(len(y)):
            point = {'xn': y[i], 'u': res1[i], 'v': res2[i], 'u - v': res1[i] - res2[i]}
            data.append(point)
        df = pd.DataFrame(data)
        # print(df.to_string())

        with open('results.txt', 'w') as f:
            f.write(df.to_string())
        max_e = max(e, key=lambda p: abs(p))
        point = e.index(max_e)
        self.output_values(max_e, y[point])

        ln.draw(y, res1, res2, int(r_var))


    def draw(self, xs, ys, us, clear=False):
        if clear:
            self.graph_axes.clear()
        self.graph_axes.plot(xs, ys, xs, us)
        self.graph_axes.set_xlabel('x')
        self.graph_axes.set_ylabel('I(x)')
        try:
            self.canvas.get_tk_widget().pack_forget()
        except AttributeError:
            pass
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas.draw()
        self.canvas.get_tk_widget().place(x=0, y=250)

    def insert_text(self):
        self.description_root = Tk()
        self.description_root.title("Условия задачи")
        self.description_canvas = Canvas(self.description_root, width=710, height=208)
        self.img = ImageTk.PhotoImage(Image.open("5.jpg"), master=self.description_root)
        self.description_canvas.create_image(20, 20, anchor=NW, image=self.img)
        self.description_canvas.pack()
        mainloop()

    def output_values(self, max_error, point):
        self.root = Tk()
        self.root.title("Выходные значения")
        text = f"Достигнутая точность = {max_error} \
                \n В точке - {point}"

        textline = Text(self.root, width=50, height=10)
        textline.insert(1.0, text)
        textline.pack()


    def table(self):
        n = int(self.iter_num_entry.get())
        r_var = bool(self.r_var.get())

        y, res1, res2, e = ln.data(r_var, n)

        # plot_data, data = ln.main(x0, y0, u0, right_limit, h)
        # n = len(data[0])
        numbers = [i for i in range(0, n)]
        diff = []
        for i in range(len(res1)):
            diff.append(res1[i] - res2[i])

        values = [
            numbers,
            y,
            res1,
            res2,
            diff

        ]

        headers = ["№", 'x', 'u', 'v', 'u - v']

        values_and_headers = [[headers[i]] + data for i, data in enumerate(values)]

        total_rows = len(values_and_headers)
        total_columns = len(values_and_headers[0])

        max_e = max(e, key=lambda p: abs(p))
        point = e.index(max_e)
        self.output_values(max_e, y[point])

        self.table_root = Tk()
        self.table_root.title("Таблица")
        self.canvas_table = Canvas(self.table_root, borderwidth=0, background="#ffffff")
        self.frame = Frame(self.canvas_table, background="#ffffff")
        self.scrollbar = Scrollbar(
            self.table_root, orient="vertical", command=self.canvas_table.yview
        )
        self.canvas_table.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.pack(side="right", fill="y")
        self.canvas_table.pack(side="left", fill="both", expand=True)
        self.canvas_table.create_window((8, 8), window=self.frame, anchor="nw")
        self.frame.bind("<Configure>", self.on_frame_configure)
        self.table = Table(
            self.table_root, self.frame, values_and_headers, total_rows, total_columns
        )

        # self.output_values(max(errors), max(H), min(H), n)

    def on_frame_configure(self, event):
        self.canvas_table.configure(scrollregion=self.canvas_table.bbox("all"))

class Table:
    def __init__(self, root, frame, lst, total_rows, total_columns):

        for i in range(total_rows):
            for j in range(total_columns):
                self.e = Entry(frame, width=25, fg="black", font=("Arial", 12))

                self.e.grid(row=j, column=i)
                self.e.insert(END, lst[i][j])

if __name__ == "__main__":
    tk = Tk()
    app = RungeKuttaGUI(window=tk)
    app.run()
    tk.mainloop()