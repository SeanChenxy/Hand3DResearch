from benchmarks.rhd.rhd_results import RHD_Results, method_list
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


if __name__ == '__main__':
    fr = RHD_Results()
    fig = plt.figure(figsize=(6, 5))

    results = []
    for m in method_list:
        p = fr.get_properties(m)
        results.append(p)
    results = sorted(results, key=lambda x: x.auc, reverse=True)

    # plot joints
    ax = fig.add_subplot(1, 1, 1)
    for r in results:
        plt.plot(r.x, r.y, label='{}[{}]: {}'.format(r.name,  r.ref, r.auc), c=r.color, linewidth=0.8)
        plt.grid(True)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlabel('error (mm)', fontsize=11, fontname='Times New Roman')
    ax.set_ylabel('3D PCK of joint (%)', fontsize=11, fontname='Times New Roman')
    ax.set_title('RHD', fontsize=11, fontname='Times New Roman')
    ax.set_xlim([20, 50])
    ax.set_ylim([0.4, 1])

    plt.subplots_adjust(left=0.1, right=0.98, top=0.94, bottom=0.11, wspace=0.15, hspace=0.15)
    plt.savefig('RHD.png')
