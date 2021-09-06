
def takeout_axis(ax):
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.set_xticks([])
    ax.set_yticks([])


def calculate_auc(xs, ys):
  length = xs[-1] - xs[0]
  area = 0
  for i in range(len(ys) - 1):
    area += (ys[i] + ys[i + 1]) * (xs[i + 1] - xs[i]) / 2 / length
  return area


def str2float(s):
    return [float(i) for i in s.split(' ')]