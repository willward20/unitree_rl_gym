# Installation and Usage Guide

## System Requirements

- **Operating System**: Recommended Ubuntu 18.04 or later  
- **GPU**: Nvidia GPU  
- **Driver Version**: Recommended version 525 or later  

---

## 1. Creating a Virtual Environment

It is recommended to run training or deployment programs in a virtual environment. Conda is recommended for creating virtual environments. If Conda is already installed on your system, you can skip step 1.1.

### 1.1 Download and Install MiniConda

MiniConda is a lightweight distribution of Conda, suitable for creating and managing virtual environments. Use the following commands to download and install:

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```

After installation, initialize Conda:

```bash
~/miniconda3/bin/conda init --all
source ~/.bashrc
```

### 1.2 Create a New Environment

Use the following command to create a virtual environment:

```bash
conda create -n unitree-rl python=3.8
```

### 1.3 Activate the Virtual Environment

```bash
conda activate unitree-rl
```

---

## 2. Installing Dependencies

### 2.1 Install PyTorch

PyTorch is a neural network computation framework used for model training and inference. Install it using the following command:

```bash
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 2.2 Install Isaac Gym

Isaac Gym is a rigid body simulation and training framework provided by Nvidia.

#### 2.2.1 Download

Download [Isaac Gym](https://developer.nvidia.com/isaac-gym) from Nvidia’s official website. On the website, click Download from Archive, agree to the license, and download. This will download a zip file. Note that the website says the software is for Ubuntu 18 and 20, but it works on 22 also. Extract the zip file. 

#### 2.2.2 Install

After extracting the package, navigate to the `IsaacGym_Preview_4_Package/isaacgym/python` folder install it using the following commands:

```bash
cd isaacgym/python
pip install -e .
```

#### 2.2.3 Verify Installation

Run the following command. (Note that `examples` is in the `isaacgym/python` directory). If a window opens displaying 1080 balls falling, the installation was successful:

```bash
cd examples
python 1080_balls_of_solitude.py
```

If you get an error about `libpython3.8.so.1.0`, you may need to export a python library to the path. In this case, use the following command, and then run the example script again:

```
export LD_LIBRARY_PATH=/home/arl2/miniconda3/envs/unitree-rl/lib:$LD_LIBRARY_PATH
```

### 2.3 Install rsl_rl

`rsl_rl` is a library implementing reinforcement learning algorithms.

#### 2.3.1 Download

Clone the repository into your home directory using Git:

```bash
cd
git clone https://github.com/leggedrobotics/rsl_rl.git
```

#### 2.3.2 Switch Branch

Switch to the v1.0.2 branch of rsl_rl:

```bash
cd rsl_rl
git checkout v1.0.2
```

#### 2.3.3 Install

```bash
pip install -e .
```

### 2.4 Install unitree_rl_gym

#### 2.4.1 Download

Clone the repository into your home directory using Git:

```bash
cd
git clone https://github.com/willward20/unitree_rl_gym.git
```

#### 2.4.2 Install

Navigate to the directory and install it:

```bash
cd unitree_rl_gym
pip install -e .
```

#### 2.4.3 Upgrade Numpy

Upgrade numpy so that it's compatible with matplotlib.

```bash
pip install numpy==1.21.6
```

### 2.5 Run the Code

Deploy 100 Go2 robot agents in Isaac Gym that were trained using policy gradient methods. The following command should load an Issac Gym window with the robots reployed. After 1001 steps in the simulation, a plot of the average rewards recieved at each step should pop up. 

```bash
python legged_gym/scripts/play-2.py --task=go2
```
