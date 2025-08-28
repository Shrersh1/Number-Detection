# Number-Detection
Neural network for detection of numbers using mnist dataset as training set

## Setup

### 1. Using Docker (recommended) :
<pre> ```bash docker build -t mnist-cnn . ``` </pre>

##### run project:
<pre> ```bash docker build -t mnist-cnn . docker run -it --rm -v $(pwd):/app mnist-cnn ``` </pre>


### 2. Using Virtual Environment (venv) :
<pre> ``` python3 -m venv .venv 
source .venv/bin/activate   # Windows: .venv\Scripts\activate 
pip install -r requirements.txt ``` </pre>

##### run project:
<pre> ``` python main.py ``` </pre>


