from benchmarks.freihand.freihand_results import FreiHAND_Results, method_list
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


if __name__ == '__main__':
    fr = FreiHAND_Results()
    fig = plt.figure(figsize=(10, 5))

    results = []
    for m in method_list:
        p = fr.get_properties(m)
        results.append(p)
    results = sorted(results, key=lambda x: x.auc['aligned_joints'], reverse=True)

    # plot joints
    ax = fig.add_subplot(1, 2, 1)
    for r in results:
        plt.plot(fr.x, r.aligned_joints, label='{}[{}]: {}'.format(r.name,  r.ref, r.auc['aligned_joints']), c=r.color, linewidth=0.8)
        plt.grid(True)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlabel('error (mm)', fontsize=11, fontname='Times New Roman')
    ax.set_ylabel('3D PCK of vertex/joint (%)', fontsize=11, fontname='Times New Roman')
    ax.set_title('Aligned joints (FreiHAND)', fontsize=11, fontname='Times New Roman')
    ax.set_xlim([0, 50])
    ax.set_ylim([0, 1])

    # plot verts
    ax = fig.add_subplot(1, 2, 2)
    for r in results:
        plt.plot(fr.x, r.aligned_verts, label='{}[{}]: {}'.format(r.name,  r.ref, r.auc['aligned_verts']), c=r.color, linewidth=0.8)
        plt.grid(True)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlabel('error (mm)', fontsize=11, fontname='Times New Roman')
    ax.set_title('Aligned vertices (FreiHAND)', fontsize=11, fontname='Times New Roman')
    ax.set_xlim([0, 50])
    ax.set_ylim([0, 1])

    plt.subplots_adjust(left=0.06, right=0.98, top=0.94, bottom=0.11, wspace=0.15, hspace=0.15)
    plt.savefig('freihand/FreiHAND.png')
