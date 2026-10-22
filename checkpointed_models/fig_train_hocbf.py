import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

file_path = './model_for_DiffPPO-LSE/tb/PPO_1/events.out.tfevents.1782732264.TT.73862.0'

print(f"{file_path} ...")
ea = event_accumulator.EventAccumulator(file_path)
ea.Reload()

tags = ea.Tags()['scalars']

if 'CBF/h0' in tags:
    events_cbf = ea.Scalars('CBF/h0')
    cbf_h0 = pd.DataFrame([{'Step': e.step, 'Value': e.value} for e in events_cbf])
    print("✅ 'CBF/h0'")
else:
    cbf_h0 = None
    print("❌ 'CBF/h0'")

if cbf_h0 is not None:
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(cbf_h0['Step']*0.93, cbf_h0['Value'], color='#1f77b4', linewidth=2.5, label='Composite HOCBF Value (H(x))')

    #ax.set_title('Evolution of Control Barrier Function (h0) during Training', fontsize=14, fontweight='bold')
    ax.set_xlabel('Steps', fontsize=12)
    #ax.set_ylabel('Value', fontsize=12)

    ax.grid(True, linestyle='--', alpha=0.8)

    ax.legend(loc='upper left', fontsize=12)

    ax.margins(x=0)

    plt.tight_layout()
   # plt.savefig('Train-hocbf.pdf')
    plt.show()

else:
    print("\n❌ CBF/h0")