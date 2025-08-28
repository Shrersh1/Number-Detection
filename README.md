
# Number-Detection
Neural network for detection of numbers using mnist dataset as training set

## Setup (in bash)

### 1. Using Docker (recommended) :

#### Download Docker:
```bash
sudo apt-get update
sudo apt-get install -y docker.io
sudo systemctl enable --now docker
```
#### Build Docker: 
```bash
 docker build -t mnist-cnn .
```

##### run project:
```bash
docker build -t mnist-cnn . docker run -it --rm -v $(pwd):/app mnist-cnn
``` 


### 2. Using Virtual Environment (venv) :
```bash
python3 -m venv .venv 
source .venv/bin/activate
pip install -r requirements.txt
```

(On Windows)
```bash
python3 -m venv .venv 
.venv\Scripts\activate 
pip install -r requirements.txt
```


##### run project:
```bash
python main.py
```

