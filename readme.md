# Landmark Retrieval Pipeline

Este repositorio contiene un sistema completo de recuperación de imágenes diseñado para identificar lugares de interés turístico a partir de imágenes, utilizando técnicas modernas de visión por computadora. A continuación, se describen los pasos necesarios para preparar el entorno y ejecutar el pipeline.

---

## ⚙️ Requisitos del sistema

Antes de comenzar, asegúrate de contar con:

* Ubuntu o distribución basada en Debian
* Acceso de superusuario (`sudo`)
* Conexión a internet
* [pyenv](https://github.com/pyenv/pyenv) y [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv)

---

## 🧱 Preparar el entorno

### 1. Instalar dependencias del sistema

```bash
sudo apt update
sudo apt install -y make build-essential libssl-dev zlib1g-dev \
  libbz2-dev libreadline-dev libsqlite3-dev curl \
  llvm libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
```

### 2. Instalar pyenv

```bash
curl https://pyenv.run | bash
```

### 3. Configurar `onstart.sh`

Edita o crea el archivo `onstart.sh` y agrega:

```bash
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

Luego ejecuta:

```bash
source onstart.sh
```

### 4. Crear entorno virtual con Python 3.11.8

```bash
pyenv install 3.11.8
pyenv virtualenv 3.11.8 tfm
pyenv activate tfm
```

---

## 🐍 Instalar dependencias de Python

```bash
python -m pip install --upgrade pip
pip uninstall -y numpy
pip install numpy==1.26.4 --force-reinstall
pip install boto3 tqdm pandas wandb matplotlib ipykernel
```

> 💡 Recuerda iniciar sesión en Weights & Biases:

```bash
export WANDB_API_KEY=<your_api_key>
wandb login
```

Instalar PyTorch (CUDA 12.1):

```bash
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121 --resume-retries=5
```

Instalar el kernel de Jupyter:

```bash
python -m ipykernel install --user --name tfm --display-name "TFM (Python 3.11)"
```

---

## 📅 Clonar el repositorio

```bash
git clone https://github.com/theviderlab/landmark-retrieval-pipeline.git .
```

---

## 🎯 Descargar pesos del modelo

```bash
cd assets/weights
# Descargar manualmente desde:
# [Peso de CVNet en Google Drive](https://drive.google.com/uc?%20export=download&id=1JAFwsVUr5JpQo3_Rhxd-V9FGdN4j8el0)
# y colocar el archivo en este directorio
```

---

## 🗂️ Descargar dataset (Open Images)

```bash
cd assets/database/open-images
python downloader.py image_ids_touristic_val.csv --download_folder=validation --num_processes=5
python downloader.py image_ids_touristic_test.csv --download_folder=test --num_processes=5
python downloader.py image_ids_touristic.csv --download_folder=train --num_processes=5
```

## 📝 Licencia

Distribuido bajo los términos del MIT License.


