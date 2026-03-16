# dataset_replay

## Setup

Create a conda environment and install the dependencies. What I prefer to do is install Isaac Sim and all dependencies into a single conda environment. If you have a different installation method for Isaac Sim, you would need to link it somehow. For the dependencies in requirements.txt, I may have missed some out -- please add any additional dependencies needed, and install new dependencies as you go when you run into errors. 

```bash
# Create and activate conda environment
conda create -n 3dv python=3.11
conda activate 3dv

# Install Isaac Sim (https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html)
pip install isaacsim[all,extscache]==5.1.0 --extra-index-url https://pypi.nvidia.com

# Install other dependencies
pip install -r requirements.txt
```

Download h5 files from Euler cluster. I chose object_in_bowl_processed_50hz/20250804_104715.h5 bag_groceries/20250829_180500.h5 as they were the smallest files. You can try other files too. 

```bash
# Copy with scp
scp USERNAME@euler.ethz.ch:/cluster/work/cvg/data/Egoverse/raw_timesynced_h5/object_in_bowl_processed_50hz/20250804_104715.h5 .

scp USERNAME@euler.ethz.ch:/cluster/work/cvg/data/Egoverse/raw_timesynced_h5/bag_groceries/20250829_180500.h5 .
```

Now run the replay file. **Make sure that the h5 file paths and USD file paths are correct.**

```bash
# Run the replay script
python franka_orca_replay.py
```

## Inspection of Dataset

You can inspect the h5 files with `inspect_h5.ipynb`. Feel free to modify or add any scripts. You should choose the kernel to be the conda environment you created.