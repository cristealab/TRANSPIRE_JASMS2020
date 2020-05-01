import numpy as np
import matplotlib.pyplot as plt

class ProgressTracker:
    def __init__(self, m, X, y):
        
        self.m = m
        self.X = X
        self.y = y
        
        self.lml = []
        self.acc = []
          
    def update(self):
        self.lml.append(self.m.compute_log_likelihood(self.X)*-1)
        means, _ = self.m.predict_y(self.X)
        self.acc.append((np.argmax(means, axis=1)==self.y.flatten()).sum()/len(means))
        
         
class ProgressPlotter:
    def __init__(self, fig, ax, ax_2, n_iter, lml, window = 50):
        
        self.n_iter = n_iter
        
        self.fig = fig
        self.ax = ax
        self.ax_2 = ax_2
        
        self.window = window

        self.ax.set_xlabel('iteration #')
        self.ax.set_ylabel('ELBO')
        self.ax_2.set_ylabel('accuracy')

        self.ax.set_ylim((0, lml))
        self.ax_2.set_ylim((0, 1))
        self.ax.set_xlim((0, n_iter+1))

        self.lml_line,  = self.ax.plot([0], [0], c='grey')
        self.acc_line,  = self.ax_2.plot([0], [0], c='steelblue')
        self.acc_txt = self.ax_2.text(0, 0, '')
        
        plt.ion()
        self.fig.tight_layout()
        self.fig.show()
        self.fig.canvas.draw()
        
    def update(self, lml, acc):
        
        self.ax.set_title('iteration # {}'.format(len(lml)))
        
        x = [int(self.window*i) for i in range(int(np.ceil(len(lml)/self.window)))]
        
        if len(x)>1:

            lml = [np.mean(np.array(lml)[x[i]:x[i+1]]) for i in range(len(x)-1)]
            acc = [np.mean(np.array(acc)[x[i]:x[i+1]]) for i in range(len(x)-1)]

            self.lml_line.set_xdata(x[:-1])
            self.lml_line.set_ydata(lml)

            self.acc_line.set_xdata(x[:-1])
            self.acc_line.set_ydata(acc)
            self.acc_txt.set_position((max(x)+self.n_iter/50, acc[-1]))
            self.acc_txt.set_text('{:.0f}%'.format(acc[-1]*100))    

            self.fig.canvas.draw()
    
    def close(self):
        plt.close(self.fig)

class CustomCallback:
    def __init__(self, tracker, plotter):
        self.tracker = tracker # ensure type
        self.plotter = plotter # ensure type
        self.interval = plotter.window
        self.i = 0
    
    def update(self, i):
        if type(i)!= int:
            self.i+=1
            i = self.i
            
        self.tracker.update()
    
        if (i % self.interval == 0) & (i != 0):
            self.plotter.update(self.tracker.lml, self.tracker.acc)
            