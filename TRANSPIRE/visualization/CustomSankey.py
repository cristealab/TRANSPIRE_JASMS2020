import matplotlib.patches
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class SankeyDiagram:
    def __init__(self, df, comparisons = None, normed = True, figsize = (10, 5), savepath=None, labels= None, **kwargs):
        self.data = df
        self.normed = normed
        
        if comparisons is not None:
            self.comparisons = comparisons
            if labels is None:
                labels = np.array(self.comparisons)[:, 0].tolist()+[np.array(self.comparisons)[-1, 1]]
            self.labels = labels

        else:
            self.comparisons = [['From', 'To']]
            self.labels = ['From', 'To']
        
        self.calculate_weights()
        self.generate_figure(figsize)
        self.plot_sankey(**kwargs)
        self.fig.tight_layout()

        if savepath is not None:
            self.fig.savefig(savepath, type='svg', dpi=150)

    def get_plot(self):
        return self.fig, self.ax

    def close(self):
        plt.close(self.fig)

    def generate_figure(self, figsize):
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.set_xticks([])
        ax.set_yticks([])
        
        for spine in ax.spines.values():
            spine.set_visible(False)
        self.fig = fig
        self.ax = ax
        
    def calculate_weights(self):
        def normed_weights(x):
            sizes = x.groupby(by=['From organelle', 'To organelle']).size()
            
            l = sizes.groupby(level=0).apply(lambda y: y.sum())
            r = sizes.groupby(level=1).apply(lambda y: y.sum())

            lw = sizes.groupby(level=0, group_keys=False).apply(lambda x: pd.Series(x/l[x.name], index=x.index))
            rw = sizes.groupby(level=1, group_keys=False).apply(lambda x: pd.Series(x/r[x.name], index=x.index))
            
            weights = pd.concat([lw, rw], keys = ['leftWeight', 'rightWeight'], axis=1)
            weights.index.names = ['leftLabel', 'rightLabel']
            
            return weights
            
        def raw_weights(x):
            sizes = x.groupby(by=['From organelle', 'To organelle']).size()
            weights = pd.concat([sizes, sizes], keys = ['leftWeight', 'rightWeight'], axis=1)
            weights.index.names = ['leftLabel', 'rightLabel']
            
            return weights
            
        if self.normed:
            if self.comparisons == [['From', 'To']]:
                weights = normed_weights(self.data)

            else:
                weights = self.data.groupby(level=['From', 'To']).apply(normed_weights)
        else:
            if self.comparisons == [['From', 'To']]:
                weights = raw_weights(self.data)
            else:
                weights = self.data.groupby(level=['From', 'To']).apply(raw_weights)
        
        self.weights = weights
   
    def plot_sankey(self, colors=None, spacing = 0.01, rect_width = 0.02, colorby='left', label_order = None, **kwargs):
        
        if self.comparisons == [['From', 'To']]:
            leftLabels = pd.Series(np.unique(self.weights.index.get_level_values('leftLabel')))
            rightLabels = pd.Series(np.unique(self.weights.index.get_level_values('rightLabel')))

            if label_order is not None:
                leftLabels = pd.Series([leftLabels[leftLabels==key].values[0] for key in label_order[::-1] if key in leftLabels.values])
                rightLabels = pd.Series([rightLabels[rightLabels==key].values[0] for key in label_order[::-1] if key in rightLabels.values])

            all_labels = np.unique(leftLabels.tolist()+rightLabels.tolist())
            
            leftFrac = 1-(spacing*(len(leftLabels)-1))
            rightFrac = 1-(spacing*(len(rightLabels)-1))
            
            leftHeights = self.weights.groupby('leftLabel').sum()/self.weights['leftWeight'].sum()
            leftHeights = leftHeights['leftWeight']*leftFrac

            rightHeights = self.weights.groupby('rightLabel').sum()/self.weights['rightWeight'].sum()
            rightHeights = rightHeights['rightWeight']*rightFrac
            
            for df in [leftLabels, rightLabels, leftHeights, rightHeights]:
                multi = pd.MultiIndex.from_tuples(list(zip(['From']*df.shape[0], ['To']*df.shape[0], df.index.values)))
                df.index = multi
            
        else:
            grouped = self.weights.groupby(level=['From', 'To'])
            leftLabels = grouped.apply(lambda x: pd.Series(np.unique(x.index.get_level_values('leftLabel'))))
            rightLabels = grouped.apply(lambda x: pd.Series(np.unique(x.index.get_level_values('rightLabel'))))
            
            all_labels = np.unique(leftLabels.values.tolist()+rightLabels.values.tolist())
            
            leftFrac = grouped.apply(lambda x: 1-spacing*len(np.unique(x.index.get_level_values('leftLabel'))))
            rightFrac = grouped.apply(lambda x: 1-spacing*len(np.unique(x.index.get_level_values('rightLabel'))))
        
            leftHeights = grouped.apply(lambda x: x.groupby('leftLabel').sum()/x['leftWeight'].sum())
            leftHeights = leftHeights['leftWeight']*leftFrac

            rightHeights = grouped.apply(lambda x: x.groupby('rightLabel').sum()/x['rightWeight'].sum())
            rightHeights = rightHeights['rightWeight']*rightFrac  
        
        leftLabels = leftLabels.unstack()
        rightLabels = rightLabels.unstack()
        leftHeights = leftHeights.unstack()
        rightHeights = rightHeights.unstack()
                
        if colors is None:
            colors = dict(zip(all_labels, plt.cm.Spectral(np.linspace(0, 1, len(all_labels)))))
        
        params = {'transform': self.ax.transAxes}
        
        xcenters = np.linspace(0, 1-rect_width, len(self.labels))
        xleft = [xcenters[0]] + (xcenters[1:-1]+spacing+rect_width).tolist()
        xright = (xcenters[1:-1]).tolist() + [xcenters[-1]]
        xcoords = list(zip(xleft, xright))
        
        xtitle = [xl-0.5*spacing for xl in xleft] + [xright[-1]+rect_width]
        
        for title, x in zip(self.labels, xtitle):
            self.ax.text(x, 1.05, title, horizontalalignment='center', verticalalignment = 'center', **params, **kwargs)
        
        for (x1, x2), comparison in zip(xcoords, self.comparisons): 
            
            left = {}
            right = {}
            
            start = 0
            
            for label in leftLabels.loc[tuple(comparison), :].dropna().values:
                
                params = {'transform': self.ax.transAxes}
                
                end = start + leftHeights.loc[tuple(comparison), label]
                center = start + leftHeights.loc[tuple(comparison), label]/2 

                left[label] = start
                if x1 == 0:
                    self.ax.text(-rect_width, center, label, horizontalalignment='right', verticalalignment = 'center',color = colors[label], **params, **kwargs)

                coord = (x1, start)

                rect = matplotlib.patches.Rectangle(coord, width= rect_width, height = leftHeights.loc[tuple(comparison), label], alpha = 0.9, color = colors[label], **params)
                self.ax.add_artist(rect)

                start = end + spacing

            start=0
            for label in rightLabels.loc[tuple(comparison), :].dropna().values:
                
                end = start + rightHeights.loc[tuple(comparison), label]
                center = start + rightHeights.loc[tuple(comparison), label]/2 

                right[label] = start

                if x2 == xcenters[-1]:
                    self.ax.text(1+rect_width, center, label, horizontalalignment='left', verticalalignment = 'center',color = colors[label], **params, **kwargs)

                coord = (x2, start)

                rect = matplotlib.patches.Rectangle(coord, width= rect_width, height = rightHeights.loc[tuple(comparison), label], alpha = 0.9, color = colors[label], **params)
                self.ax.add_artist(rect)

                start = end + spacing
        
            
            for l in leftLabels.loc[tuple(comparison), :].dropna().values:
                for r in rightLabels.loc[tuple(comparison), :].dropna().values:
                    
                    if self.labels == ['From', 'To']:
                        idx = (l, r)
                        rsel = idx[1]
                        lsel = idx[0]

                    else:
                        idx = (tuple(comparison)+(l, r))
                        rsel = (idx[0], idx[1], idx[-1])
                        lsel = idx[:-1]
                    
                    if idx in self.weights.index:

                        lh = leftHeights.loc[tuple(comparison), l]
                        rh = rightHeights.loc[tuple(comparison), r]

                        lcontribution = lh*self.weights.loc[idx, 'leftWeight']/self.weights.loc[lsel, 'leftWeight'].sum()
                        rcontribution = rh*self.weights.loc[idx, 'rightWeight']/self.weights.swaplevel().loc[rsel, 'rightWeight'].sum()

                        t = np.array(50*[left[l]] + 50*[right[r]])
                        t = np.convolve(t, 0.05 * np.ones(20), mode='valid')
                        t = np.convolve(t, 0.05 * np.ones(20), mode='valid')

                        b = np.array(50*[left[l]+lcontribution] + 50*[right[r]+rcontribution])
                        b = np.convolve(b, 0.05 * np.ones(20), mode='valid')
                        b = np.convolve(b, 0.05 * np.ones(20), mode='valid')

                        left[l] += lcontribution
                        right[r] += rcontribution

                        if colorby == 'left':
                            c = colors[l]
                        elif colorby == 'right':
                            c = colors[r]

                        self.ax.fill_between(np.linspace(x1+rect_width/2, x2+rect_width/2, len(t)), b, t, alpha=0.65, color=c, **params)