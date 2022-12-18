# Morton G-PCC
This script compresses vertices data with the Morton order within a given boundary.

## 0. Prerequisite
### 0.1. Environment
- Python3
- Ubuntu 20.04 LTS ( may be worked on other OS )

### 0.2. Dependencies

I abbreviate `sudo` comand.

```Bash
# Python3
apt -y update
apt -y install libssl-dev libffi-dev python3-dev python3-distutils
apt -y install python3-pip
# Python modules
python3 -m pip install -U pip setuptools
python3 -m pip install numpy matplotlib
```

### 0.3. Download
```Bash
git clone https://github.com/cetusk/MortonGPCC
```

Or for contribution such any pull request. I welcome that!

```Bash
git clone https://ghp_xxx@github.com/cetusk/MortonGPCC
```

The `xxx` is your git acces token.

### 0.3. Note
In current status, this script works on the Python environment and 2 dimensional space. I'm planning to expand it as C++ and 3 dimensional space.


## 1. Usage

### 1.1. Execute sample script

```Bash
python3 run.py
```

### 1.2. Parameters

|Variable|Context|Default|
|:--|:--|:--|
|`depth`|Level of detail such tree depth|`8` ( maximum )|
|`dim`|Spatial dimension|`2` ( only supported )|
|`x0`, `y0`|Minimum of boundary point|`0.0`|
|`x1`, `y1`|Maximum of boundary point|`100.0`|
|`numPoints`|Number of points sampled randomly|`100`|
